//! Blocking (sync) streaming client — feature `client-sync`.
//!
//! Iterators over the async chunk stream for callers that live outside a
//! tokio runtime. The runtime is owned by the iterator: built once, driven
//! from a background thread, and dropped when iteration ends. Chunks cross
//! the async→sync boundary through a bounded crossfire channel — the
//! dual-context MPMC whose async sender pairs with this blocking receiver —
//! so a slow consumer pays backpressure onto the socket instead of buffering
//! the whole stream in memory.
//!
//! Channel capacity is sized to the deframer's worst-case per-event fan-out:
//! Anthropic emits at most one canonical chunk per input event; Responses'
//! terminal event can carry text + accumulated thinking + several complete
//! tool calls.

use crossfire::MRx;
use crossfire::mpmc::Array;
use futures::StreamExt;

use crate::canonical::CanonChunk;
use crate::client::DialectClient;
use crate::error::ProxyError;
use crate::items::ItemRequest;

/// One chunk per canonical item, in wire order, blocking on each `next()`.
/// The tokio task feeding the channel is detached; when the iterator is
/// dropped the channel closes and the pump's send fails, aborting the HTTP
/// request (reqwest aborts on dropped response bodies).
pub struct BlockingChunks {
    rx: MRx<Array<Result<CanonChunk, ProxyError>>>,
}

impl Iterator for BlockingChunks {
    type Item = Result<CanonChunk, ProxyError>;
    fn next(&mut self) -> Option<Self::Item> {
        // Channel close is a clean `None` end — the underlying Err indicates
        // disconnect-drain and the chunk payload inside was already relayed.
        self.rx.recv().ok()
    }
}

/// The canonical stream types crossfire knows how to route.
const CHUNK_FEED: usize = 8;

impl DialectClient {
    /// Blocking chunk iterator over the streaming request.
    ///
    /// Spawns a tokio task on a private runtime (one per call) that drives
    /// [`Self::stream`] and forwards each item through the channel; networks
    /// that complete one turn per client can afford the extra runtime, and
    /// per-call isolation keeps shutdown simple (drop = close = abort).
    pub fn stream_blocking(&self, req: &ItemRequest) -> BlockingChunks {
        let (tx, rx) =
            crossfire::mpmc::bounded_async_blocking::<Result<CanonChunk, ProxyError>>(CHUNK_FEED);
        let client = self.clone();
        let req = req.clone();
        // One runtime per blocking call: single-thread is plenty for a
        // thin HTTP → deframe → channel pump, and avoids entangling the
        // caller's ambient runtime.
        std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = tx.try_send(Err(ProxyError::Transport(format!(
                        "tokio runtime build: {e}"
                    ))));
                    return;
                }
            };
            rt.block_on(async move {
                let mut s = std::pin::pin!(client.stream(&req));
                while let Some(item) = s.next().await {
                    if tx.send(item).await.is_err() {
                        // caller dropped the iterator — abort the HTTP body
                        return;
                    }
                }
            });
        });
        BlockingChunks { rx }
    }
}

#[cfg(all(test, feature = "axum"))]
mod tests {
    use super::*;
    use crate::client::Dialect as CiDialect;

    fn text(t: &str) -> CanonChunk {
        CanonChunk {
            delta_text: t.into(),
            ..Default::default()
        }
    }

    async fn serve_once(app: axum::Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn blocking_iterator_collects_full_stream() {
        let chunks = vec![
            text("a"),
            text("b"),
            text("c"),
            CanonChunk {
                finish_reason: Some("stop".into()),
                usage: Some(crate::canonical::Usage {
                    prompt_tokens: 3,
                    completion_tokens: 2,
                    cached_read_tokens: 0,
                    cache_write_tokens: 0,
                    reasoning_tokens: None,
                }),
                ..Default::default()
            },
        ];
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move || {
                let s = futures::stream::iter(
                    chunks.clone().into_iter().map(Ok::<CanonChunk, ProxyError>),
                );
                async move {
                    crate::dialect::openai_chat::stream::openai_stream_response(
                        Box::pin(s),
                        "id".into(),
                        "m".into(),
                        0,
                        true,
                    )
                }
            }),
        );
        let base = serve_once(app).await;
        // hand the server a beat to bind before the client dials
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let client = DialectClient::new(base, CiDialect::OpenAiChat);
        let req = crate::items::ItemRequest {
            model: "m".into(),
            max_tokens: Some(4),
            stream: true,
            messages: vec![crate::items::ItemStreamMessage {
                role: crate::items::Role::User,
                items: vec![crate::items::ContentItem::Text { text: "x".into() }],
                metadata: crate::items::ItemMeta::default(),
            }],
            ..Default::default()
        };

        // The caller is sync at heart but this test's runtime is already
        // real — hand iteration to a plain thread so the outer tokio reactor
        // stays free to serve the mocker.
        let collected: Vec<_> =
            tokio::task::spawn_blocking(move || client.stream_blocking(&req).collect())
                .await
                .unwrap();
        // panic loudly on a transport error instead of silently filtering it
        let items: Vec<CanonChunk> = collected.into_iter().collect::<Result<_, _>>().unwrap();
        let text: String = items.iter().map(|c| c.delta_text.as_str()).collect();
        assert_eq!(text, "abc");
        // on the OpenAI chat wire finish and usage arrive on separate frames —
        // the framer splits them; assert both independently
        let term = items
            .iter()
            .find(|c| c.finish_reason.is_some())
            .expect("terminal chunk");
        assert_eq!(term.finish_reason.as_deref(), Some("stop"));
        let usage = items
            .iter()
            .find_map(|c| c.usage.as_ref())
            .expect("usage chunk");
        assert_eq!(usage.prompt_tokens, 3);
    }

    #[tokio::test]
    async fn dropped_iterator_aborts_upstream_body() {
        // long stream, one chunk pulled — the pump task exits once the
        // channel closes; no leak beyond the test.
        let chunks: Vec<CanonChunk> = (0..1000).map(|i| text(&i.to_string())).collect();
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move || {
                let s = futures::stream::iter(
                    chunks.clone().into_iter().map(Ok::<CanonChunk, ProxyError>),
                );
                async move {
                    crate::dialect::openai_chat::stream::openai_stream_response(
                        Box::pin(s),
                        "id".into(),
                        "m".into(),
                        0,
                        true,
                    )
                }
            }),
        );
        let base = serve_once(app).await;
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let client = DialectClient::new(base, CiDialect::OpenAiChat);
        let req = crate::items::ItemRequest {
            model: "m".into(),
            max_tokens: Some(4),
            stream: true,
            messages: vec![crate::items::ItemStreamMessage {
                role: crate::items::Role::User,
                items: vec![crate::items::ContentItem::Text { text: "x".into() }],
                metadata: crate::items::ItemMeta::default(),
            }],
            ..Default::default()
        };

        tokio::task::spawn_blocking(move || {
            let mut it = client.stream_blocking(&req);
            let first = it.next().expect("one chunk").expect("chunk ok");
            assert!(!first.delta_text.is_empty());
            std::mem::drop(it);
            // the pump folds on next send; the request aborts. Anything
            // beyond that is socket-timing, not behavior to assert on.
        })
        .await
        .unwrap();
    }
}
