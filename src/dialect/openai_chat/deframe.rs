//! OpenAI chat-completions SSE frames → canonical [`CanonChunk`]s (client
//! direction).
//!
//! Inverse of [`super::stream::OpenAiFramer`]: one state machine fed with
//! already-split `data:` payloads (SSE line splitting is a transport concern
//! — M4's client drives this from `eventsource-stream`). In framer order:
//! role preamble, content/tool/thinking deltas, the finish-reason frame, the
//! opt-in choice-less usage trailer, then exactly one terminal
//! `data: [DONE]`. In-band `{"error": …}` frames surface as an `Err` chunk
//! and terminate the stream (their `[DONE]` rides along in the same write
//! but is consumed silently).

use crate::canonical::{CanonChunk, ThinkingDelta, Usage};
use crate::error::ProxyError;

/// One `data:` payload → zero or one canonical chunks. `stream_end` marks
/// the terminal frame: nothing valid follows it.
#[derive(Debug, Default)]
pub struct DeframeOut {
    pub chunk: Option<CanonChunk>,
    pub stream_end: bool,
}

pub struct OpenAiDeframer {
    done_seen: bool,
}

impl OpenAiDeframer {
    pub fn new() -> Self {
        Self { done_seen: false }
    }

    /// Feed one `data:` payload (the text after `data: `, without the
    /// trailing blank-line separator). The chunk payload arrives via `Ok`;
    /// errors are split by kind:
    ///
    /// * `Err(ProxyError::Upstream)` — an in-band provider error frame.
    ///   Terminal: the frame carries its own `[DONE]` per the protocol.
    /// * `Err(ProxyError::Transport)` — malformed payload (bad JSON), or a
    ///   frame arriving after the terminal. The stream is broken; the caller
    ///   should abort.
    pub fn push_data(&mut self, data: &str) -> Result<DeframeOut, ProxyError> {
        if self.done_seen {
            return Err(ProxyError::Transport(
                "chunk received after terminal [DONE]".into(),
            ));
        }
        if data.trim() == "[DONE]" {
            self.done_seen = true;
            return Ok(DeframeOut {
                chunk: None,
                stream_end: true,
            });
        }
        let v: serde_json::Value = serde_json::from_str(data)
            .map_err(|e| ProxyError::Transport(format!("malformed stream chunk JSON: {e}")))?;
        if v["error"].is_object() {
            self.done_seen = true;
            let msg = v["error"]["message"]
                .as_str()
                .unwrap_or("upstream error")
                .to_string();
            return Err(ProxyError::Upstream {
                status: v["error"]["status"].as_u64().unwrap_or(500) as u16,
                body: msg,
                retry_after_secs: None,
            });
        }
        Ok(self.chunk_from(v))
    }

    fn chunk_from(&self, v: serde_json::Value) -> DeframeOut {
        // choice-less usage trailer: choices is empty, usage carries the data.
        if v["choices"].as_array().is_none_or(|c| c.is_empty()) {
            let usage = v["usage"].is_object().then(|| usage_from(&v["usage"]));
            return DeframeOut {
                chunk: usage.map(|u| CanonChunk {
                    usage: Some(u),
                    ..Default::default()
                }),
                stream_end: false,
            };
        }
        let choice = &v["choices"][0];
        let delta = &choice["delta"];
        let mut chunk = CanonChunk::default();
        // The role preamble is bookkeeping, not payload — skip (the canonical
        // side doesn't model roles, the framer just guarantees it arrives
        // first).
        if delta["role"].is_string() {
            return DeframeOut::default();
        }
        if let Some(t) = delta["content"].as_str() {
            chunk.delta_text = t.to_string();
        }
        if delta["tool_calls"].is_array() {
            chunk.tool_calls = Some(delta["tool_calls"].clone());
        }
        if let Some(th) = delta.get("thinking").filter(|t| t.is_object()) {
            chunk.thinking = Some(ThinkingDelta {
                block_index: th["block_index"].as_u64().unwrap_or(0),
                kind: match th["kind"].as_str() {
                    Some("signature") => "signature",
                    _ => "thinking",
                },
                text: th["text"].as_str().unwrap_or_default().to_string(),
            });
        }
        if let Some(fr) = choice["finish_reason"].as_str() {
            chunk.finish_reason = Some(fr.to_string());
        }
        // policy: usage rides the finish-reason chunk when both are present;
        // a choice-less trailer carries usage on its own (handled above).
        if v["usage"].is_object() {
            chunk.usage = Some(usage_from(&v["usage"]));
        }
        // empty chunks (e.g. a usage-less trailer on a payload frame) don't
        // surface — the canonical model doesn't carry no-op markers
        let has_payload = !chunk.delta_text.is_empty()
            || chunk.tool_calls.is_some()
            || chunk.finish_reason.is_some()
            || chunk.thinking.is_some()
            || chunk.usage.is_some();
        DeframeOut {
            chunk: has_payload.then_some(chunk),
            stream_end: false,
        }
    }

    /// End-of-body invoice: the framer guarantees exactly one `[DONE]`; a
    /// body that closes without one is a truncation the embedder must hear
    /// about, not a clean EOF.
    pub fn finalize(&self) -> Result<(), ProxyError> {
        if self.done_seen {
            Ok(())
        } else {
            Err(ProxyError::Transport(
                "stream ended without terminal [DONE]".into(),
            ))
        }
    }
}

impl Default for OpenAiDeframer {
    fn default() -> Self {
        Self::new()
    }
}

/// OpenAI-style usage block → canonical cache-inclusive Usage. `prompt_tokens`
/// is already cache-inclusive by convention; the detail objects lift cached /
/// reasoning counters when the provider reports them.
fn usage_from(u: &serde_json::Value) -> Usage {
    Usage {
        prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0),
        completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
        cached_read_tokens: u["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0),
        cache_write_tokens: u["cache_creation_input_tokens"].as_u64().unwrap_or(0),
        reasoning_tokens: u["completion_tokens_details"]["reasoning_tokens"].as_u64(),
    }
}

/// Re-split raw SSE bytes into `data:` payloads and feed them through —
/// convenience for in-memory tests (the M4 transport uses
/// `eventsource-stream` instead, which yields the same payloads).
#[cfg(all(test, feature = "axum"))]
pub(crate) fn deframe_sse_body(body: &str, d: &mut OpenAiDeframer) -> Vec<DeframeOut> {
    let mut outs = Vec::new();
    for frame in body.split("\n\n") {
        for line in frame.lines() {
            if let Some(payload) = line.strip_prefix("data: ")
                && !payload.is_empty()
            {
                match d.push_data(payload) {
                    Ok(o) => outs.push(o),
                    Err(_) => return outs,
                }
            }
        }
    }
    outs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::Usage;

    fn chunk(text: &str) -> CanonChunk {
        CanonChunk {
            delta_text: text.into(),
            ..Default::default()
        }
    }

    fn end_chunk() -> CanonChunk {
        CanonChunk {
            finish_reason: Some("stop".into()),
            usage: Some(Usage {
                prompt_tokens: 5,
                completion_tokens: 2,
                cached_read_tokens: 0,
                cache_write_tokens: 0,
                reasoning_tokens: None,
            }),
            ..Default::default()
        }
    }

    fn drive(chunks: Vec<CanonChunk>) -> Vec<DeframeOut> {
        let mut framer = super::super::stream::OpenAiFramer::new("id".into(), "m".into(), 0, true);
        let mut deframer = OpenAiDeframer::new();
        let mut outs = Vec::new();
        for c in &chunks {
            for seg in framer.frame_chunk(c) {
                for payload in seg_split(&seg) {
                    outs.push(deframer.push_data(&payload).unwrap());
                }
            }
        }
        for seg in framer.finish() {
            for payload in seg_split(&seg) {
                outs.push(deframer.push_data(&payload).unwrap());
            }
        }
        outs
    }

    /// Split a concatenated framer segment into its `data:` payloads.
    fn seg_split(seg: &str) -> Vec<String> {
        seg.split("\n\n")
            .filter_map(|f| f.strip_prefix("data: ").filter(|s| !s.is_empty()))
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn round_trip_plain_text_reassembles() {
        let outs = drive(vec![chunk("hello "), chunk("world"), end_chunk()]);
        let text: String = outs
            .iter()
            .filter_map(|o| o.chunk.as_ref())
            .map(|c| c.delta_text.clone())
            .collect();
        assert_eq!(text, "hello world");
        let finish = outs
            .iter()
            .filter_map(|o| o.chunk.as_ref())
            .find(|c| c.finish_reason.is_some())
            .unwrap();
        assert_eq!(finish.finish_reason.as_deref(), Some("stop"));
        // the framer splits usage onto its own chunk; either the finish or
        // the trailer may carry it — find whoever does
        let usage = outs
            .iter()
            .filter_map(|o| o.chunk.as_ref())
            .find_map(|c| c.usage.as_ref())
            .expect("a usage chunk");
        assert_eq!(usage.prompt_tokens, 5);
        assert!(
            outs.last().unwrap().stream_end,
            "last frame must be the terminal"
        );
    }

    #[test]
    fn role_preamble_is_skipped_not_forwarded() {
        // the assistant-role preamble chunk carries no useful payload on the
        // canonical side — verify the deframer doesn't surface it as an
        // empty chunk
        let outs = drive(vec![chunk("x")]);
        let empties = outs
            .iter()
            .filter_map(|o| o.chunk.as_ref())
            .filter(|c| {
                c.delta_text.is_empty()
                    && c.tool_calls.is_none()
                    && c.finish_reason.is_none()
                    && c.usage.is_none()
                    && c.thinking.is_none()
            })
            .count();
        assert_eq!(empties, 0, "role preamble must not surface: {outs:?}");
    }

    #[test]
    fn tool_calls_round_trip() {
        let outs = drive(vec![
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":""}}
                ])),
                ..chunk("")
            },
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"function":{"arguments":"{\"cmd\":\"ls\"}"}}
                ])),
                ..chunk("")
            },
            CanonChunk {
                finish_reason: Some("tool_calls".into()),
                ..chunk("")
            },
        ]);
        let tcs: Vec<&serde_json::Value> = outs
            .iter()
            .filter_map(|o| o.chunk.as_ref())
            .filter_map(|c| c.tool_calls.as_ref())
            .collect();
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0][0]["id"], "call_1");
        assert_eq!(tcs[1][0]["function"]["arguments"], "{\"cmd\":\"ls\"}");
    }

    #[test]
    fn double_terminal_is_an_error() {
        let mut d = OpenAiDeframer::new();
        d.push_data("[DONE]").unwrap();
        assert!(d.push_data("[DONE]").is_err());
        assert!(d.finalize().is_ok());
    }

    #[test]
    fn missing_terminal_is_a_truncation() {
        let mut d = OpenAiDeframer::new();
        d.push_data(
            r#"{"id":"x","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#,
        )
        .unwrap();
        assert!(d.finalize().is_err(), "EOF without [DONE] must not be OK");
    }

    #[test]
    fn error_frame_terminates_as_upstream_error() {
        let mut d = OpenAiDeframer::new();
        let err = d
            .push_data(r#"{"error":{"message":"boom","type":"server_error"}}"#)
            .unwrap_err();
        assert!(matches!(err, ProxyError::Upstream { status: 500, .. }));
    }

    #[test]
    fn malformed_json_is_an_error() {
        let mut d = OpenAiDeframer::new();
        assert!(d.push_data("{not json").is_err());
    }

    #[cfg(feature = "axum")]
    #[tokio::test]
    async fn end_to_end_through_the_server_pump() {
        use futures::StreamExt;
        let chunks = vec![
            Ok::<_, ProxyError>(chunk("a")),
            Ok(chunk("b")),
            Ok(end_chunk()),
        ];
        let stream = futures::stream::iter(chunks);
        let resp = super::super::stream::openai_stream_response(
            stream.boxed(),
            "id".into(),
            "m".into(),
            0,
            true,
        );
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        let mut d = OpenAiDeframer::new();
        let outs = deframe_sse_body(&body, &mut d);
        assert!(d.finalize().is_ok());
        let text: String = outs
            .iter()
            .filter_map(|o| o.chunk.as_ref())
            .map(|c| c.delta_text.clone())
            .collect();
        assert_eq!(text, "ab");
    }
}
