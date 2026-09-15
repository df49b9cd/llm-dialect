//! Canonical chunk stream → OpenAI chat-completions SSE framing.
//!
//! Emits the role preamble chunk, one data frame per canonical chunk
//! (skipping empties), the opt-in trailing usage chunk, a guaranteed
//! terminal `data: [DONE]`, and OpenAI-style out-of-band error frames.
//! The per-item framing runs inside the shared SSE driver (`dialect::sse`).

use crate::canonical::CanonChunk;
use crate::error::ProxyError;
#[cfg(feature = "axum")]
use futures::Stream;

// ---------- pure framer (no runtime types) ----------

/// OpenAI chat-completions SSE framer: role preamble on first chunk, one data
/// frame per canonical chunk (empties skipped), the opt-in trailing usage
/// chunk, and the guaranteed terminal `data: [DONE]` on `finish`.
pub struct OpenAiFramer {
    first_pending: bool,
    done_sent: bool,
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
}

impl OpenAiFramer {
    pub fn new(id: String, model: String, created: u64, include_usage: bool) -> Self {
        Self {
            first_pending: true,
            done_sent: false,
            id,
            model,
            created,
            include_usage,
        }
    }

    /// All segments for one chunk (already `data: …\n\n`-formatted), in order.
    pub fn frame_chunk(&mut self, chunk: &CanonChunk) -> Vec<String> {
        let mut bytes = String::new();
        // OpenAI clients expect the first chunk to carry delta.role
        if self.first_pending {
            self.first_pending = false;
            let role = serde_json::json!({
                "id": self.id, "object": "chat.completion.chunk", "created": self.created,
                "model": self.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
            });
            bytes.push_str(&format!("data: {role}\n\n"));
        }
        if let Some(json) = chunk.to_sse_json(&self.id, &self.model, self.created, false) {
            bytes.push_str(&format!("data: {json}\n\n"));
        }
        if self.include_usage
            && chunk.usage.is_some()
            && let Some(json) = chunk.to_sse_json(&self.id, &self.model, self.created, true)
        {
            bytes.push_str(&format!("data: {json}\n\n"));
        }
        if bytes.is_empty() {
            Vec::new()
        } else {
            vec![bytes]
        }
    }

    /// Out-of-band error frame; terminal (carries its own `[DONE]`).
    pub fn frame_error(&mut self, e: &ProxyError) -> Vec<String> {
        self.done_sent = true;
        // never echo upstream bodies into a client stream
        let msg = match e {
            ProxyError::Upstream { status, .. } => {
                format!("upstream returned status {status}")
            }
            other => other.to_string(),
        };
        let frame = serde_json::json!({"error": {"message": msg, "type": "server_error"}});
        vec![format!("data: {frame}\n\ndata: [DONE]\n\n")]
    }

    /// Guaranteed `[DONE]` unless an error frame already terminated.
    pub fn finish(&mut self) -> Vec<String> {
        if self.done_sent {
            Vec::new()
        } else {
            vec!["data: [DONE]\n\n".into()]
        }
    }
}

/// OpenAI SSE framing over the accounted canonical stream (idle timeout,
/// billing and circuit-breaker reporting all happen inside `LoggedStream`).
/// The framer above is pure; this shell only pumps.
#[cfg(feature = "axum")]
pub fn openai_stream_response<S>(
    inner: S,
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
) -> axum::response::Response
where
    S: Stream<Item = Result<CanonChunk, ProxyError>> + Unpin + Send + 'static,
{
    let framer = std::sync::Arc::new(std::sync::Mutex::new(OpenAiFramer::new(
        id,
        model,
        created,
        include_usage,
    )));
    let framer_done = framer.clone();
    crate::dialect::sse::sse_response(
        inner,
        move |out, item| {
            let mut f = framer.lock().unwrap();
            match item {
                Ok(chunk) => {
                    for seg in f.frame_chunk(&chunk) {
                        out.push(seg);
                    }
                }
                Err(e) => {
                    for seg in f.frame_error(&e) {
                        out.push(seg);
                    }
                }
            }
        },
        move |out| {
            let mut f = framer_done.lock().unwrap();
            for seg in f.finish() {
                out.push(seg);
            }
        },
    )
}

#[cfg(all(test, feature = "axum"))]
mod tests {
    use super::*;
    use crate::canonical::Usage;

    fn stream_chunks(
        chunks: Vec<Result<CanonChunk, ProxyError>>,
    ) -> futures::stream::BoxStream<'static, Result<CanonChunk, ProxyError>> {
        Box::pin(futures::stream::iter(chunks))
    }

    fn chunk(text: &str) -> CanonChunk {
        CanonChunk {
            delta_text: text.into(),
            ..Default::default()
        }
    }

    async fn collect(resp: axum::response::Response) -> String {
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn usage_chunk_only_when_client_opted_in() {
        let mk = || {
            vec![
                Ok(chunk("hi")),
                Ok(CanonChunk {
                    finish_reason: Some("stop".into()),
                    usage: Some(Usage {
                        prompt_tokens: 5,
                        completion_tokens: 2,
                        cached_read_tokens: 0,
                        cache_write_tokens: 0,
                        reasoning_tokens: None,
                    }),
                    ..chunk("")
                }),
            ]
        };
        let off = openai_stream_response(
            stream_chunks(mk()),
            "chatcmpl-1".into(),
            "m".into(),
            0,
            false,
        );
        let s = collect(off).await;
        assert!(
            !s.contains("\"usage\""),
            "usage leaked without include_usage: {s}"
        );
        assert!(s.contains("data: [DONE]"));
        let on = openai_stream_response(
            stream_chunks(mk()),
            "chatcmpl-1".into(),
            "m".into(),
            0,
            true,
        );
        let s = collect(on).await;
        assert!(
            s.contains("\"prompt_tokens\":5"),
            "usage missing with include_usage: {s}"
        );
    }

    #[tokio::test]
    async fn first_chunk_carries_role_and_error_frame_terminates() {
        let s = collect(openai_stream_response(
            stream_chunks(vec![Err(ProxyError::upstream(
                502,
                "secret upstream body".into(),
            ))]),
            "chatcmpl-1".into(),
            "m".into(),
            0,
            false,
        ))
        .await;
        assert!(s.contains("\"type\":\"server_error\""), "{s}");
        // upstream body text never leaks into client frames
        assert!(!s.contains("secret upstream body"), "{s}");
        assert!(s.ends_with("data: [DONE]\n\n"), "{s}");
    }

    #[tokio::test]
    async fn first_chunk_carries_role_preamble_before_data() {
        // the role preamble chunk must go out exactly once, before any data
        // frame, even when the first canonical chunk carries content
        let s = collect(openai_stream_response(
            stream_chunks(vec![Ok(chunk("hello"))]),
            "chatcmpl-1".into(),
            "m".into(),
            0,
            false,
        ))
        .await;
        let role = s
            .find("\"delta\":{\"role\":\"assistant\"}")
            .expect("role preamble missing");
        let data = s.find("hello").expect("data frame missing");
        assert!(role < data, "role preamble must precede data frames: {s}");
        assert_eq!(
            s.matches("\"delta\":{\"role\":\"assistant\"}").count(),
            1,
            "exactly one role preamble: {s}"
        );
    }
}
