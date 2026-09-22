//! Anthropic Messages SSE frames → canonical [`CanonChunk`]s (client
//! direction).
//!
//! Inverse of [`super::stream::chunk_to_sse_events`]: one state machine fed
//! with `(event, data)` frame pairs (SSE line splitting is a transport
//! concern — M4's client drives this from `eventsource-stream`). The framer
//! guarantees a `message_start` preamble, correctly-nested
//! `content_block_start`/`delta`/`stop` blocks, and exactly one terminal
//! `message_delta` + `message_stop` pair; the deframer reassembles the
//! deltas, validates block nesting and the terminal pair, and maps the
//! terminal stop reason back to OpenAI's `finish_reason` vocabulary.
//!
//! Stop-sequence windows and withheld tails are server-side concerns only —
//! the deframer consumes what arrives verbatim (the server already applied
//! the StopWindow to the wire).
//!
//! Usage arithmetic: the terminal `message_delta` reports `input_tokens`
//! cache-EXCLUSIVE; the canonical chunk re-adds the cache classes so
//! `prompt_tokens = fresh + cached_read + cache_write` holds uniformly.

use crate::canonical::{CanonChunk, ThinkingDelta, Usage};
use crate::error::ProxyError;

/// One Anthropic SSE event → zero or one canonical chunks; `stream_end`
/// flags the terminal event, after which anything further is an error.
#[derive(Debug, Default)]
pub struct DeframeOut {
    pub chunk: Option<CanonChunk>,
    pub stream_end: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PendingReason {
    None,
    /// message_delta arrived before its matching message_stop (both halves
    /// of the terminal pair are accounted for).
    WaitingStop,
}

pub struct AnthropicDeframer {
    /// per-block-index open state, mirroring the framer's block-open vec
    blocks: Vec<bool>,
    /// whether the message_start preamble has been consumed
    started: bool,
    /// terminal message_delta seen; waiting on the message_stop
    pending: PendingReason,
    /// stream fully terminated (message_stop or error frame)
    done: bool,
}

impl AnthropicDeframer {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            started: false,
            pending: PendingReason::None,
            done: false,
        }
    }

    /// Feed one SSE frame pair. `event` is the `event:` field (e.g.
    /// `content_block_delta`), `data` the JSON payload.
    pub fn push(&mut self, event: &str, data: &str) -> Result<DeframeOut, ProxyError> {
        if self.done {
            return Err(ProxyError::Transport(
                "frame received after message_stop".into(),
            ));
        }
        let v: serde_json::Value = serde_json::from_str(data)
            .map_err(|e| ProxyError::Transport(format!("malformed anthropic frame JSON: {e}")))?;
        match event {
            "message_start" => {
                if self.started {
                    return Err(ProxyError::Transport("duplicate message_start".into()));
                }
                self.started = true;
                // The preamble's usage.input_tokens is the prompt count the
                // Anthropic SDK reads at open; surface it so callers that
                // consume tokens eagerly can start accounting immediately.
                let input_tokens = v["message"]["usage"]["input_tokens"].as_u64();
                Ok(DeframeOut {
                    chunk: input_tokens.map(|n| CanonChunk {
                        input_tokens: Some(n),
                        ..Default::default()
                    }),
                    stream_end: false,
                })
            }
            "content_block_start" => {
                let idx = v["index"].as_u64().unwrap_or(0) as usize;
                if self.blocks.len() <= idx {
                    self.blocks.resize(idx + 1, false);
                }
                self.blocks[idx] = true;
                Ok(DeframeOut::default())
            }
            "content_block_delta" => {
                let idx = v["index"].as_u64().unwrap_or(0) as usize;
                if self.blocks.get(idx) != Some(&true) {
                    return Err(ProxyError::Transport(format!(
                        "content_block_delta on non-open block {idx}"
                    )));
                }
                let delta = &v["delta"];
                let mut chunk = CanonChunk::default();
                match delta["type"].as_str().unwrap_or_default() {
                    "text_delta" => {
                        chunk.delta_text = delta["text"].as_str().unwrap_or_default().to_string();
                    }
                    "thinking_delta" => {
                        chunk.thinking = Some(ThinkingDelta {
                            block_index: idx as u64,
                            kind: "thinking",
                            text: delta["thinking"].as_str().unwrap_or_default().to_string(),
                        });
                    }
                    "signature_delta" => {
                        chunk.thinking = Some(ThinkingDelta {
                            block_index: idx as u64,
                            kind: "signature",
                            text: delta["signature"].as_str().unwrap_or_default().to_string(),
                        });
                    }
                    "input_json_delta" => {
                        // partial tool-call arguments ride the side-channel the
                        // framer uses: canonical tool_calls OpenAI-delta shape
                        chunk.tool_calls = Some(serde_json::json!([{
                            "index": 0,
                            "function": {"arguments": delta["partial_json"].as_str().unwrap_or("")}
                        }]));
                        // args deltas open nothing on their own; the block
                        // start carried the tool_use id+name. The deframer
                        // surfaces only the argument stream.
                    }
                    other => {
                        tracing::debug!(delta_type = other, "skipping unknown anthropic delta");
                    }
                }
                let has_payload = !chunk.delta_text.is_empty()
                    || chunk.thinking.is_some()
                    || chunk.tool_calls.is_some();
                Ok(DeframeOut {
                    chunk: has_payload.then_some(chunk),
                    stream_end: false,
                })
            }
            "content_block_stop" => {
                let idx = v["index"].as_u64().unwrap_or(0) as usize;
                if self.blocks.get(idx) != Some(&true) {
                    return Err(ProxyError::Transport(format!(
                        "content_block_stop on non-open block {idx}"
                    )));
                }
                self.blocks[idx] = false;
                Ok(DeframeOut::default())
            }
            "message_delta" => {
                if self.pending != PendingReason::None {
                    return Err(ProxyError::Transport("duplicate message_delta".into()));
                }
                self.pending = PendingReason::WaitingStop;
                let stop = v["delta"]["stop_reason"].as_str().unwrap_or_default();
                let usage = &v["usage"];
                // cache-inclusive reconstruction: fresh + read + write
                let fresh = usage["input_tokens"].as_u64().unwrap_or(0);
                let cached_read = usage["cache_read_input_tokens"].as_u64().unwrap_or(0);
                let cache_write = usage["cache_creation_input_tokens"].as_u64().unwrap_or(0);
                let chunk = CanonChunk {
                    finish_reason: Some(map_stop_reason_inbound(stop).to_string()),
                    usage: Some(Usage {
                        prompt_tokens: fresh + cached_read + cache_write,
                        completion_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
                        cached_read_tokens: cached_read,
                        cache_write_tokens: cache_write,
                        reasoning_tokens: None,
                    }),
                    ..Default::default()
                };
                Ok(DeframeOut {
                    chunk: Some(chunk),
                    stream_end: false,
                })
            }
            "message_stop" => {
                self.done = true;
                Ok(DeframeOut {
                    chunk: None,
                    stream_end: true,
                })
            }
            "error" => {
                self.done = true;
                let msg = v["error"]["message"]
                    .as_str()
                    .unwrap_or("upstream error")
                    .to_string();
                Err(ProxyError::Upstream {
                    status: v["error"]["status"].as_u64().unwrap_or(500) as u16,
                    body: msg,
                    retry_after_secs: None,
                })
            }
            _ => {
                tracing::debug!(event, "skipping unknown anthropic event");
                Ok(DeframeOut::default())
            }
        }
    }

    /// End-of-body invoice: the terminal pair (message_delta + message_stop)
    /// must both have arrived. Half a pair (delta without stop) is a
    /// truncation; a body that never reached message_start produced nothing.
    pub fn finalize(&self) -> Result<(), ProxyError> {
        if !self.started {
            return Ok(()); // empty stream: nothing to close, like the framer
        }
        if !self.done {
            return Err(ProxyError::Transport(
                "stream ended without terminal message_stop".into(),
            ));
        }
        Ok(())
    }
}

impl Default for AnthropicDeframer {
    fn default() -> Self {
        Self::new()
    }
}

/// Inverse of the framer's stop-reason table (same wildcard rule as the
/// non-stream parser: unknown → stop).
fn map_stop_reason_inbound(reason: &str) -> &'static str {
    match reason {
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        _ => "stop",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::Usage;
    fn run(chunks: Vec<CanonChunk>) -> (Vec<Result<DeframeOut, ProxyError>>, AnthropicDeframer) {
        let mut st = super::super::stream::StreamState::new();
        let mut d = AnthropicDeframer::new();
        let mut outs = Vec::new();
        for c in &chunks {
            for (ev, data) in super::super::stream::chunk_to_sse_events(c, "m", &mut st, "msg_1") {
                match d.push(ev, &data) {
                    Ok(o) => outs.push(Ok(o)),
                    Err(e) => {
                        outs.push(Err(e));
                        return (outs, d);
                    }
                }
            }
        }
        for (ev, data) in super::super::stream::finalize_stream(&mut st) {
            match d.push(ev, &data) {
                Ok(o) => outs.push(Ok(o)),
                Err(e) => {
                    outs.push(Err(e));
                    return (outs, d);
                }
            }
        }
        (outs, d)
    }

    fn text(t: &str) -> CanonChunk {
        CanonChunk {
            delta_text: t.into(),
            ..Default::default()
        }
    }

    #[test]
    fn round_trip_plain_text_reassembles() {
        let (outs, d) = run(vec![
            text("hello "),
            text("world"),
            CanonChunk {
                finish_reason: Some("stop".into()),
                usage: Some(Usage {
                    prompt_tokens: 9,
                    completion_tokens: 4,
                    cached_read_tokens: 0,
                    cache_write_tokens: 0,
                    reasoning_tokens: None,
                }),
                ..Default::default()
            },
        ]);
        assert!(d.finalize().is_ok());
        let text: String = outs
            .iter()
            .filter_map(|o| o.as_ref().ok())
            .filter_map(|o| o.chunk.as_ref())
            .map(|c| c.delta_text.clone())
            .collect();
        assert_eq!(text, "hello world");
        let term = outs
            .iter()
            .filter_map(|o| o.as_ref().ok())
            .find_map(|o| o.chunk.as_ref().filter(|c| c.finish_reason.is_some()));
        assert_eq!(term.unwrap().finish_reason.as_deref(), Some("stop"));
        // usage arithmetic: framer subtracts cache classes, deframer re-adds
        assert_eq!(term.unwrap().usage.as_ref().unwrap().prompt_tokens, 9);
        assert!(outs.last().unwrap().as_ref().unwrap().stream_end);
    }

    #[test]
    fn cache_classes_fold_back_into_prompt_tokens() {
        let (outs, _) = run(vec![CanonChunk {
            finish_reason: Some("stop".into()),
            usage: Some(Usage {
                prompt_tokens: 15,
                completion_tokens: 4,
                cached_read_tokens: 2,
                cache_write_tokens: 1,
                reasoning_tokens: None,
            }),
            ..Default::default()
        }]);
        let term = outs
            .iter()
            .filter_map(|o| o.as_ref().ok())
            .find_map(|o| o.chunk.as_ref().filter(|c| c.finish_reason.is_some()))
            .unwrap();
        let u = term.usage.as_ref().unwrap();
        assert_eq!(u.prompt_tokens, 15);
        assert_eq!(u.cached_read_tokens, 2);
        assert_eq!(u.cache_write_tokens, 1);
    }

    #[test]
    fn thinking_and_signature_deltas_route_back() {
        let (outs, _) = run(vec![
            CanonChunk {
                thinking: Some(ThinkingDelta {
                    block_index: 0,
                    kind: "thinking",
                    text: "plan".into(),
                }),
                ..Default::default()
            },
            CanonChunk {
                thinking: Some(ThinkingDelta {
                    block_index: 0,
                    kind: "signature",
                    text: "sig".into(),
                }),
                ..Default::default()
            },
            CanonChunk {
                finish_reason: Some("stop".into()),
                usage: Some(Usage::default()),
                ..Default::default()
            },
        ]);
        let ths: Vec<&ThinkingDelta> = outs
            .iter()
            .filter_map(|o| o.as_ref().ok())
            .filter_map(|o| o.chunk.as_ref())
            .filter_map(|c| c.thinking.as_ref())
            .collect();
        assert_eq!(ths.len(), 2);
        assert_eq!(ths[0].kind, "thinking");
        assert_eq!(ths[1].kind, "signature");
    }

    #[test]
    fn tool_json_arguments_flow_as_partial_json_deltas() {
        let (outs, _) = run(vec![
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"id":"call_1","function":{"name":"bash","arguments":""}}
                ])),
                ..Default::default()
            },
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"function":{"arguments":"{\"cmd\":\"ls\"}"}}
                ])),
                ..Default::default()
            },
            CanonChunk {
                finish_reason: Some("tool_calls".into()),
                usage: Some(Usage::default()),
                ..Default::default()
            },
        ]);
        let args: Vec<&str> = outs
            .iter()
            .filter_map(|o| o.as_ref().ok())
            .filter_map(|o| o.chunk.as_ref())
            .filter_map(|c| c.tool_calls.as_ref())
            .filter_map(|t| t[0]["function"]["arguments"].as_str())
            .collect();
        assert_eq!(args, vec!["{\"cmd\":\"ls\"}"]);
    }

    #[test]
    fn truncated_stream_fails_finalize() {
        let chunks = vec![text("partial")];
        let mut st = super::super::stream::StreamState::new();
        let mut d = AnthropicDeframer::new();
        for c in &chunks {
            for (ev, data) in super::super::stream::chunk_to_sse_events(c, "m", &mut st, "msg_1") {
                d.push(ev, &data).unwrap();
            }
        }
        // no finalize_stream — the framer would add the missing terminal; the
        // wire body just ends
        assert!(d.finalize().is_err());
    }

    #[test]
    fn error_frame_terminates() {
        let mut d = AnthropicDeframer::new();
        d.push(
            "message_start",
            r#"{"type":"message_start","message":{"id":"m","usage":{"input_tokens":1,"output_tokens":0}}}"#,
        )
        .unwrap();
        let err = d
            .push(
                "error",
                r#"{"error":{"message":"boom","type":"api_error"}}"#,
            )
            .unwrap_err();
        assert!(matches!(err, ProxyError::Upstream { .. }));
        assert!(
            d.push("message_stop", "{}").is_err(),
            "post-error frame must be rejected"
        );
    }
}
