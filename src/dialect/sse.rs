//! The one shared SSE response driver for all three client surfaces.
//!
//! Owns the pump pattern the dialect framers had triplicated: drive a
//! canonical-chunk stream through a per-item framing closure, then a
//! finalizer, emitting pre-formatted SSE segments as `text/event-stream`
//! bytes. All translation logic stays in the caller's closure (pure framer
//! state machines); this module only handles stream plumbing, headers, and
//! the guaranteed final frame — mirrors the OpenAI surface's guaranteed
//! `[DONE]` / Anthropic's `message_stop` even when the provider truncates.
//!
//! Closures are plain `FnMut`/`FnOnce` over pre-formatted segments, so each
//! dialect keeps its own wire shape (Anthropic/Responses use named
//! `event:`/`data:` frames; OpenAI chat uses bare `data:` lines). The
//! closures capture their framer state directly (plain `FnMut` needs no
//! `Send` sharing across tasks); where a dialect's frame and finalizer
//! closures must share one state machine (anthropic, openai_chat,
//! openai_responses), the caller wraps it in `Arc<Mutex<..>>` — the driver
//! itself owns no state and never awaits inside a closure, so the locks are
//! uncontended.

#[cfg(feature = "axum")]
use crate::canonical::CanonChunk;
#[cfg(feature = "axum")]
use crate::error::ProxyError;
#[cfg(feature = "axum")]
use crate::error::error_response;
#[cfg(feature = "axum")]
use axum::body::Body;
#[cfg(feature = "axum")]
use axum::http::StatusCode;
#[cfg(feature = "axum")]
use axum::response::Response;
#[cfg(feature = "axum")]
use futures::{Stream, StreamExt};

/// Build the SSE `Response` for a canonical-chunk stream.
///
/// `frame` renders one pipeline item into zero or more pre-formatted SSE
/// segments (Ok chunks → dialect frames; Err items → the dialect's error
/// frame, which may carry its own terminator). `finish` runs once after the
/// stream ends and emits the dialect's finalizer segments (empty when the
/// error frame already terminated). The response is always a 200 — stream
/// errors are in-band dialect frames.
#[cfg(feature = "axum")]
pub fn sse_response<S, F, G>(inner: S, frame: F, finish: G) -> Response
where
    S: Stream<Item = Result<CanonChunk, ProxyError>> + Unpin + Send + 'static,
    F: FnMut(&mut Vec<String>, Result<CanonChunk, ProxyError>) + Send + 'static,
    G: FnOnce(&mut Vec<String>) + Send + 'static,
{
    let mut frame = frame;
    let stream = inner.map(move |item| {
        let mut out: Vec<String> = Vec::new();
        frame(&mut out, item);
        if out.is_empty() {
            None
        } else {
            Some(
                out.into_iter()
                    .map(Ok::<_, std::convert::Infallible>)
                    .collect::<Vec<_>>(),
            )
        }
    });
    let stream = stream.chain(futures::stream::once(async move {
        let mut out: Vec<String> = Vec::new();
        finish(&mut out);
        if out.is_empty() {
            None
        } else {
            Some(
                out.into_iter()
                    .map(Ok::<_, std::convert::Infallible>)
                    .collect::<Vec<_>>(),
            )
        }
    }));
    let body = Body::from_stream(
        stream
            .filter_map(|v| async move { v })
            .flat_map(futures::stream::iter),
    );
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("x-accel-buffering", "no")
        .body(body)
        .unwrap_or_else(|e| error_response(&ProxyError::Internal(e.into())))
}
