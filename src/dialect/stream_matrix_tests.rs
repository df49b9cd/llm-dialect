//! Cross-surface stream invariant matrix.
//!
//! One table of canonical chunk scripts (plain text, tool calls, extended
//! thinking, usage trailer, truncation) driven through all three downstream
//! framer entry points (openai chat, anthropic messages, openai responses),
//! asserting the invariants that must hold under every shape:
//! - exactly one terminal frame (`[DONE]` / `message_stop` / `completed|incomplete|failed`)
//! - deltas only on open content blocks (anthropic reopen rules)
//! - monotonic sequence numbers (responses)
//! - usage arithmetic: cache-inclusive canonical survives to the wire
//! - no content frames after the terminal
//!
//! Runs with no server, no ports, no mock: the framers take in-memory
//! streams directly.

use super::anthropic::stream::{
    StreamState, anthropic_stream_response, chunk_to_sse_events, finalize_stream,
};
use super::openai_chat::stream::openai_stream_response;
use super::openai_responses::stream::responses_stream_response;
use crate::canonical::{CanonChunk, ThinkingDelta, Usage};
use crate::error::ProxyError;

fn text(s: &str) -> CanonChunk {
    CanonChunk {
        delta_text: s.into(),
        ..Default::default()
    }
}

fn usage(p: u64, c: u64) -> Usage {
    Usage {
        prompt_tokens: p,
        completion_tokens: c,
        cached_read_tokens: 0,
        cache_write_tokens: 0,
        reasoning_tokens: None,
    }
}

fn ok_stream(
    chunks: Vec<CanonChunk>,
) -> futures::stream::BoxStream<'static, Result<CanonChunk, ProxyError>> {
    Box::pin(futures::stream::iter(chunks.into_iter().map(Ok)))
}

async fn body(resp: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// The shared script set, per scenario. Every scenario ends terminated or
/// truncated; truncation exercises the framers' guaranteed finalizers.
fn scripts() -> Vec<(&'static str, Vec<CanonChunk>)> {
    vec![
        ("plain_text", vec![text("hello "), text("world")]),
        (
            "tool_calls",
            vec![
                text("checking"),
                CanonChunk {
                    tool_calls: Some(serde_json::json!([
                        {"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":""}}
                    ])),
                    ..text("")
                },
                CanonChunk {
                    tool_calls: Some(serde_json::json!([
                        {"index":0,"function":{"arguments":"{\"cmd\":\"ls\"}"}}
                    ])),
                    ..text("")
                },
                CanonChunk {
                    finish_reason: Some("tool_calls".into()),
                    ..text("")
                },
            ],
        ),
        (
            "thinking",
            vec![
                CanonChunk {
                    thinking: Some(ThinkingDelta {
                        block_index: 0,
                        kind: "thinking",
                        text: "let me think".into(),
                    }),
                    ..text("")
                },
                text("answer"),
                CanonChunk {
                    finish_reason: Some("stop".into()),
                    ..text("")
                },
            ],
        ),
        (
            "finish_then_usage_trailer",
            vec![
                text("hi"),
                CanonChunk {
                    finish_reason: Some("stop".into()),
                    ..text("")
                },
                CanonChunk {
                    usage: Some(usage(9, 4)),
                    ..text("")
                },
            ],
        ),
        ("truncated", vec![text("partial")]),
    ]
}

async fn assert_chat_surface(script: &[CanonChunk]) {
    let s = body(openai_stream_response(
        ok_stream(script.to_vec()),
        "chatcmpl-1".into(),
        "m".into(),
        0,
        true,
    ))
    .await;
    // exactly one [DONE], last frame, and the terminal finish rides it
    assert_eq!(s.matches("data: [DONE]").count(), 1, "chat {s}");
    assert!(
        s.trim_end().ends_with("data: [DONE]"),
        "chat [DONE] last: {s}"
    );
    // role preamble exactly once, before any data frame
    assert_eq!(
        s.matches("\"delta\":{\"role\":\"assistant\"}").count(),
        1,
        "chat role preamble: {s}"
    );
    // usage arithmetic survives (include_usage=true, trailer present)
    if script.iter().any(|c| c.usage.is_some()) {
        assert!(s.contains("\"prompt_tokens\":"), "chat usage missing: {s}");
    }
    // tool-call scripts must carry tool_calls deltas to the client
    if script.iter().any(|c| c.tool_calls.is_some()) {
        assert!(s.contains("\"tool_calls\""), "chat tool deltas: {s}");
        assert!(
            s.contains("\"finish_reason\":\"tool_calls\""),
            "chat tool finish: {s}"
        );
    }
}

async fn assert_anthropic_surface(script: &[CanonChunk]) {
    let resp = anthropic_stream_response(
        ok_stream(script.to_vec()),
        "m".into(),
        "msg_1".into(),
        vec![],
    );
    let s = body(resp).await;
    // exactly one message_stop, and nothing content-bearing after it
    assert_eq!(s.matches("event: message_stop").count(), 1, "anthropic {s}");
    let stop_at = s.find("event: message_stop").unwrap();
    let tail = &s[stop_at..];
    assert!(
        !tail.contains("content_block_delta") && !tail.contains("content_block_start"),
        "anthropic content after message_stop: {tail}"
    );
    // message_start preamble exactly once
    assert_eq!(
        s.matches("event: message_start").count(),
        1,
        "anthropic {s}"
    );
    // deltas only on open blocks: track start/stop indices across the stream
    let mut open = std::collections::HashSet::new();
    for frame in s.split("\n\n") {
        let Some(data) = frame.strip_prefix("data: ") else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
            continue;
        };
        match frame.strip_prefix("event: ") {
            Some("content_block_start") => {
                let i = v["index"].as_i64().unwrap();
                assert!(open.insert(i), "block {i} started twice: {s}");
            }
            Some("content_block_stop") => {
                open.remove(&v["index"].as_i64().unwrap());
            }
            Some("content_block_delta") => {
                let i = v["index"].as_i64().unwrap();
                assert!(
                    open.contains(&i),
                    "anthropic delta on non-open block {i}: {s}"
                );
            }
            _ => {}
        }
    }
    // usage arithmetic: terminal message_delta input_tokens must be
    // cache-correct (fresh = prompt - cached - write); these scripts carry no
    // cache classes, so prompt must survive as-is when a usage chunk exists
    if let Some(u) = script.iter().find_map(|c| c.usage.as_ref()) {
        let md = s
            .split("\n\n")
            .find(|f| f.starts_with("event: message_delta"))
            .unwrap();
        let data = md.split("data: ").nth(1).unwrap();
        let v: serde_json::Value = serde_json::from_str(data).unwrap();
        assert_eq!(
            v["usage"]["input_tokens"].as_u64().unwrap(),
            u.prompt_tokens,
            "anthropic terminal usage: {v}"
        );
    }
}

async fn assert_responses_surface(script: &[CanonChunk]) {
    let resp = responses_stream_response(ok_stream(script.to_vec()), "resp_1".into(), "m".into());
    let s = body(resp).await;
    // exactly one terminal lifecycle frame (completed | incomplete), never
    // both, and no item frames after it
    let terminals = s.matches("event: response.completed").count()
        + s.matches("event: response.incomplete").count();
    assert_eq!(terminals, 1, "responses terminal count: {s}");
    let term_ev = if s.contains("event: response.completed") {
        "event: response.completed"
    } else {
        "event: response.incomplete"
    };
    let tail = &s[s.find(term_ev).unwrap()..];
    assert!(
        !tail.contains("output_text.delta") && !tail.contains("function_call_arguments.delta"),
        "responses deltas after terminal: {tail}"
    );
    // response.created preamble exactly once, sequence numbers strictly increase
    assert_eq!(
        s.matches("event: response.created").count(),
        1,
        "responses {s}"
    );
    let mut last = -1i64;
    for frame in s.split("\n\n") {
        let Some(data) = frame.strip_prefix("data: ") else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
            continue;
        };
        if let Some(n) = v["sequence_number"].as_i64() {
            assert!(n > last, "responses seq not increasing: {n} after {last}");
            last = n;
        }
    }
    // tool identity survives: name-bearing calls reach the completed output
    if script.iter().any(|c| c.tool_calls.is_some()) {
        assert!(s.contains("bash"), "responses tool identity: {s}");
    }
    // finish/usage semantics: usage rides the terminal frame
    if let Some(u) = script.iter().find_map(|c| c.usage.as_ref()) {
        let term = s.split("\n\n").find(|f| f.starts_with(term_ev)).unwrap();
        assert!(
            term.contains(&format!("\"input_tokens\":{}", u.prompt_tokens)),
            "responses terminal usage: {term}"
        );
    }
}

#[tokio::test]
async fn matrix_invariants_hold_on_every_surface() {
    for (name, script) in scripts() {
        let _ = name;
        assert_chat_surface(&script).await;
        assert_anthropic_surface(&script).await;
        assert_responses_surface(&script).await;
    }
    // silence the unused imports when the pure-framer helpers go unused here
    let _ = (
        chunk_to_sse_events,
        finalize_stream,
        StreamState::new,
        ProxyError::upstream,
    );
}

#[tokio::test]
async fn matrix_midstream_error_terminates_every_surface() {
    let err =
        || Err::<CanonChunk, ProxyError>(ProxyError::upstream(502, "upstream blew up".into()));
    // openai chat: error frame + its own [DONE]
    let items: Vec<Result<CanonChunk, ProxyError>> = vec![Ok(text("partial")), err()];
    let s = body(openai_stream_response(
        Box::pin(futures::stream::iter(items)),
        "chatcmpl-1".into(),
        "m".into(),
        0,
        false,
    ))
    .await;
    assert!(s.contains("\"type\":\"server_error\""), "chat error: {s}");
    assert_eq!(
        s.matches("data: [DONE]").count(),
        1,
        "chat error [DONE]: {s}"
    );
    // anthropic: event: error terminator, no message_stop after
    let items: Vec<Result<CanonChunk, ProxyError>> = vec![Ok(text("partial")), err()];
    let resp = anthropic_stream_response(
        Box::pin(futures::stream::iter(items)),
        "m".into(),
        "msg_1".into(),
        vec![],
    );
    let s = body(resp).await;
    assert_eq!(s.matches("event: error").count(), 1, "anthropic error: {s}");
    assert_eq!(
        s.matches("event: message_stop").count(),
        0,
        "anthropic error: {s}"
    );
    // responses: response.failed, no completed after
    let items: Vec<Result<CanonChunk, ProxyError>> = vec![Ok(text("partial")), err()];
    let resp = responses_stream_response(
        Box::pin(futures::stream::iter(items)),
        "resp_1".into(),
        "m".into(),
    );
    let s = body(resp).await;
    assert!(
        s.contains("event: response.failed"),
        "responses failed: {s}"
    );
    assert_eq!(
        s.matches("event: response.completed").count(),
        0,
        "responses failed: {s}"
    );
}
