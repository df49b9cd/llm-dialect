//! OpenAI Responses SSE frames → canonical [`CanonChunk`]s (client
//! direction).
//!
//! Inverse of [`super::stream::ResponsesFramer`]: one state machine fed with
//! `(event, data)` frame pairs (SSE line splitting is a transport concern —
//! M4's client drives this from `eventsource-stream`). The framer emits the
//! full item lifecycle (`response.output_item.added` → channel deltas →
//! per-item close-out → a single terminal `response.completed`/`incomplete`/
//! `failed`), stamping monotonic `sequence_number`s on everything; the
//! deframer validates sequencing, folds deltas into canonical chunks
//! (published at close time, since the canonical model carries *complete*
//! chunks per turn, not partial-JSON streams), and enforces the
//! exactly-one-terminal-frame invariant.

use crate::canonical::{CanonChunk, ThinkingDelta, Usage};
use crate::error::ProxyError;

/// One `(event, data)` pair → zero or one canonical chunks; `stream_end`
/// flags the terminal frame.
#[derive(Debug, Default)]
pub struct DeframeOut {
    pub chunk: Option<CanonChunk>,
    pub stream_end: bool,
}

/// Per-output-item accumulation, keyed by output_index.
#[derive(Debug, Clone)]
enum ItemAcc {
    Text {
        #[allow(dead_code)]
        id: String,
        text: String,
    },
    Reasoning {
        #[allow(dead_code)]
        id: String,
        text: String,
    },
    ToolCall {
        call_id: String,
        name: String,
        args: String,
    },
}

pub struct ResponsesDeframer {
    items: std::collections::BTreeMap<usize, ItemAcc>,
    /// published chunks, in stream order — one per output item, emitted at
    /// that item's `.done` frame (or at the terminal for unfinished items)
    last_seq: Option<i64>,
    terminal_seen: bool,
}

impl ResponsesDeframer {
    pub fn new() -> Self {
        Self {
            items: Default::default(),
            last_seq: None,
            terminal_seen: false,
        }
    }

    pub fn push(&mut self, event: &str, data: &str) -> Result<DeframeOut, ProxyError> {
        if self.terminal_seen {
            return Err(ProxyError::Transport(
                "frame received after response terminal".into(),
            ));
        }
        let v: serde_json::Value = serde_json::from_str(data)
            .map_err(|e| ProxyError::Transport(format!("malformed responses frame JSON: {e}")))?;
        // Sequence numbers arrive on every frame and must increase strictly.
        if let Some(n) = v["sequence_number"].as_i64() {
            if let Some(last) = self.last_seq
                && n <= last
            {
                return Err(ProxyError::Transport(format!(
                    "sequence_number not increasing: {n} after {last}"
                )));
            }
            self.last_seq = Some(n);
        }
        match event {
            "response.created" => Ok(DeframeOut::default()),
            "response.output_item.added" => {
                let item = &v["item"];
                let idx = v["output_index"].as_u64().unwrap_or(0) as usize;
                let acc = match item["type"].as_str().unwrap_or_default() {
                    "message" => ItemAcc::Text {
                        id: item["id"].as_str().unwrap_or_default().to_string(),
                        text: String::new(),
                    },
                    "reasoning" => ItemAcc::Reasoning {
                        id: item["id"].as_str().unwrap_or_default().to_string(),
                        text: String::new(),
                    },
                    "function_call" => ItemAcc::ToolCall {
                        call_id: item["call_id"].as_str().unwrap_or_default().to_string(),
                        name: item["name"].as_str().unwrap_or_default().to_string(),
                        args: String::new(),
                    },
                    other => {
                        tracing::debug!(item_type = other, "skipping unknown output item");
                        return Ok(DeframeOut::default());
                    }
                };
                self.items.insert(idx, acc);
                Ok(DeframeOut::default())
            }
            "response.output_text.delta" => {
                let idx = v["output_index"].as_u64().unwrap_or(0) as usize;
                let delta = v["delta"].as_str().unwrap_or_default().to_string();
                if let Some(ItemAcc::Text { text, .. }) = self.items.get_mut(&idx) {
                    text.push_str(&delta);
                }
                // canonical carries per-delta text, so re-emit immediately —
                // accumulation is for the terminal snapshot, the chunk here
                // feeds any consumer that doesn't want to wait for it.
                Ok(DeframeOut {
                    chunk: (!delta.is_empty()).then(|| CanonChunk {
                        delta_text: delta,
                        ..Default::default()
                    }),
                    stream_end: false,
                })
            }
            "response.reasoning_summary_text.delta" => {
                let idx = v["output_index"].as_u64().unwrap_or(0) as usize;
                if let Some(ItemAcc::Reasoning { text, .. }) = self.items.get_mut(&idx) {
                    text.push_str(v["delta"].as_str().unwrap_or_default());
                }
                Ok(DeframeOut::default())
            }
            "response.function_call_arguments.delta" => {
                let idx = v["output_index"].as_u64().unwrap_or(0) as usize;
                if let Some(ItemAcc::ToolCall { args, .. }) = self.items.get_mut(&idx) {
                    args.push_str(v["delta"].as_str().unwrap_or_default());
                }
                Ok(DeframeOut::default())
            }
            // The *_done and content_part frames are structure-only — the
            // deltas above carry the payloads; close-out frames only matter
            // for leak validation. Nothing to publish yet.
            "response.output_text.done"
            | "response.function_call_arguments.done"
            | "response.reasoning_summary_part.done"
            | "response.reasoning_summary_part.added"
            | "response.content_part.added"
            | "response.content_part.done"
            | "response.output_item.done" => Ok(DeframeOut::default()),
            "response.completed" | "response.incomplete" => {
                self.terminal_seen = true;
                let resp = &v["response"];
                let finish_reason = match resp["status"].as_str() {
                    Some("incomplete") | Some("failed") => {
                        match resp["incomplete_details"]["reason"].as_str() {
                            Some("max_output_tokens") | Some("max_tokens") => "length",
                            Some("content_filter") => "content_filter",
                            _ => "stop",
                        }
                    }
                    _ => "stop",
                };
                let u = &resp["usage"];
                let usage = Usage {
                    prompt_tokens: u["input_tokens"].as_u64().unwrap_or(0),
                    completion_tokens: u["output_tokens"].as_u64().unwrap_or(0),
                    cached_read_tokens: u["input_tokens_details"]["cached_tokens"]
                        .as_u64()
                        .unwrap_or(0),
                    cache_write_tokens: 0,
                    reasoning_tokens: u["output_tokens_details"]["reasoning_tokens"].as_u64(),
                };
                // Collapse accumulated items into one terminal chunk carrying
                // everything the caller needs.
                let mut chunk = CanonChunk {
                    finish_reason: Some(finish_reason.into()),
                    usage: Some(usage),
                    ..Default::default()
                };
                let mut thinking = String::new();
                let mut tool_calls: Vec<serde_json::Value> = Vec::new();
                for acc in self.items.values() {
                    match acc {
                        // text was already published per-delta above —
                        // accumulating it into the terminal would double
                        // deliver
                        ItemAcc::Text { .. } => {}
                        ItemAcc::Reasoning { text, .. } => thinking.push_str(text),
                        ItemAcc::ToolCall {
                            call_id,
                            name,
                            args,
                        } => {
                            tool_calls.push(serde_json::json!({
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": args},
                            }));
                        }
                    }
                }
                if !thinking.is_empty() {
                    chunk.thinking = Some(ThinkingDelta {
                        block_index: 0,
                        kind: "thinking",
                        text: thinking,
                    });
                }
                if !tool_calls.is_empty() {
                    chunk.tool_calls = Some(serde_json::Value::Array(tool_calls));
                    // only override the default "stop" — a "length" /
                    // "content_filter" from the terminal status must win
                    // (payload insight, not a real stop reason).
                    if matches!(chunk.finish_reason.as_deref(), Some("stop")) {
                        chunk.finish_reason = Some("tool_calls".into());
                    }
                }
                Ok(DeframeOut {
                    chunk: Some(chunk),
                    stream_end: true,
                })
            }
            "response.failed" => {
                self.terminal_seen = true;
                let msg = v["response"]["error"]["message"]
                    .as_str()
                    .unwrap_or("upstream error")
                    .to_string();
                Err(ProxyError::Upstream {
                    status: 500,
                    body: msg,
                    retry_after_secs: None,
                })
            }
            _ => {
                tracing::debug!(event, "skipping unknown responses event");
                Ok(DeframeOut::default())
            }
        }
    }

    /// End-of-body invoice: the terminal frame is mandatory. Called after the
    /// SSE body closes; a missing terminal is a truncation.
    pub fn finalize(&self) -> Result<(), ProxyError> {
        if !self.terminal_seen {
            return Err(ProxyError::Transport(
                "stream ended without a response terminal frame".into(),
            ));
        }
        Ok(())
    }
}

impl Default for ResponsesDeframer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::Usage;

    fn drive(chunks: Vec<CanonChunk>) -> (Vec<Result<DeframeOut, ProxyError>>, ResponsesDeframer) {
        let mut framer = super::super::stream::ResponsesFramer::new("resp_1".into(), "m".into());
        let mut d = ResponsesDeframer::new();
        let mut outs = Vec::new();
        for c in chunks {
            for seg in framer.frame_item(Ok(c)) {
                for (ev, data) in split_segment(&seg) {
                    match d.push(&ev, &data) {
                        Ok(o) => outs.push(Ok(o)),
                        Err(e) => {
                            outs.push(Err(e));
                            return (outs, d);
                        }
                    }
                }
            }
        }
        for seg in framer.finish() {
            for (ev, data) in split_segment(&seg) {
                match d.push(&ev, &data) {
                    Ok(o) => outs.push(Ok(o)),
                    Err(e) => {
                        outs.push(Err(e));
                        return (outs, d);
                    }
                }
            }
        }
        (outs, d)
    }

    /// Split a concatenated framer segment into (event, data) pairs. The
    /// framer emits `event: E\ndata: {json}\n\n` per frame.
    fn split_segment(seg: &str) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for frame in seg.split("\n\n") {
            let lines: Vec<&str> = frame.lines().collect();
            let ev = lines
                .iter()
                .find_map(|l| l.strip_prefix("event: "))
                .unwrap_or_default();
            let data = lines
                .iter()
                .find_map(|l| l.strip_prefix("data: "))
                .unwrap_or_default();
            if !ev.is_empty() && !data.is_empty() {
                out.push((ev.to_string(), data.to_string()));
            }
        }
        out
    }

    fn text(t: &str) -> CanonChunk {
        CanonChunk {
            delta_text: t.into(),
            ..Default::default()
        }
    }

    #[test]
    fn deltas_floated_and_terminal_carries_the_rest() {
        let (outs, d) = drive(vec![
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
        // per-delta text floats immediately
        let text_delta: String = outs
            .iter()
            .filter_map(|o| o.as_ref().ok())
            .filter(|o| !o.stream_end)
            .filter_map(|o| o.chunk.as_ref())
            .map(|c| c.delta_text.clone())
            .collect();
        assert_eq!(text_delta, "hello world");
        // terminal carries finish+usage, no duplicated text
        let term = outs
            .iter()
            .filter_map(|o| o.as_ref().ok())
            .find(|o| o.stream_end)
            .unwrap();
        let c = term.chunk.as_ref().unwrap();
        assert!(
            c.delta_text.is_empty(),
            "text must not be re-published at terminal"
        );
        assert_eq!(c.finish_reason.as_deref(), Some("stop"));
        assert_eq!(c.usage.as_ref().unwrap().prompt_tokens, 9);
    }

    #[test]
    fn tool_call_recovers_complete_json_arguments() {
        let (outs, _) = drive(vec![
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":""}}
                ])),
                ..text("")
            },
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"function":{"arguments":"{\"cmd\":"}}
                ])),
                ..text("")
            },
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"function":{"arguments":"\"ls\"}"}}
                ])),
                ..text("")
            },
            CanonChunk {
                finish_reason: Some("tool_calls".into()),
                usage: Some(Usage::default()),
                ..text("")
            },
        ]);
        let term = outs
            .iter()
            .filter_map(|o| o.as_ref().ok())
            .find(|o| o.stream_end)
            .unwrap();
        let c = term.chunk.as_ref().unwrap();
        let tcs = c.tool_calls.as_ref().unwrap().as_array().unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["id"], "call_1");
        assert_eq!(tcs[0]["function"]["name"], "bash");
        assert_eq!(tcs[0]["function"]["arguments"], "{\"cmd\":\"ls\"}");
        assert_eq!(c.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn reasoning_deltas_reassemble_to_thinking() {
        let (outs, _) = drive(vec![
            CanonChunk {
                thinking: Some(ThinkingDelta {
                    block_index: 0,
                    kind: "thinking",
                    text: "let me ".into(),
                }),
                ..text("")
            },
            CanonChunk {
                thinking: Some(ThinkingDelta {
                    block_index: 0,
                    kind: "thinking",
                    text: "think".into(),
                }),
                ..text("")
            },
            CanonChunk {
                finish_reason: Some("stop".into()),
                usage: Some(Usage::default()),
                ..text("")
            },
        ]);
        let term = outs
            .iter()
            .filter_map(|o| o.as_ref().ok())
            .find(|o| o.stream_end)
            .unwrap();
        let c = term.chunk.as_ref().unwrap();
        assert_eq!(c.thinking.as_ref().unwrap().text, "let me think");
    }

    #[test]
    fn non_monotonic_sequence_number_is_an_error() {
        let mut d = ResponsesDeframer::new();
        d.push(
            "response.output_item.added",
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m1"},"sequence_number":5}"#,
        )
        .unwrap();
        assert!(
            d.push(
                "response.output_text.done",
                r#"{"type":"response.output_text.done","item_id":"m1","output_index":0,"content_index":0,"text":"","sequence_number":4}"#,
            )
            .is_err()
        );
    }

    #[test]
    fn missing_terminal_is_a_truncation() {
        let (outs, mut d) = {
            // drive manually so the framer's finalizer is not involved
            let mut framer = super::super::stream::ResponsesFramer::new("resp".into(), "m".into());
            let mut d = ResponsesDeframer::new();
            let mut outs = Vec::new();
            for seg in framer.frame_item(Ok(text("partial"))) {
                for (ev, data) in split_segment(&seg) {
                    outs.push(d.push(&ev, &data).unwrap());
                }
            }
            (outs, d)
        };
        let _ = outs;
        assert!(d.finalize().is_err());
        d.push(
            "response.completed",
            r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}},"sequence_number":99}"#,
        )
        .unwrap();
        assert!(d.finalize().is_ok());
    }

    #[test]
    fn error_frame_emits_upstream_error() {
        let (outs, d) = drive(vec![]);
        let _ = outs;
        assert!(d.finalize().is_err());
        // an Err chunk drives frame_item(Err(..)) -> response.failed
        let mut framer = super::super::stream::ResponsesFramer::new("r".into(), "m".into());
        let segs = framer.frame_item(Err(ProxyError::upstream(502, "boom".into())));
        assert!(
            segs.iter().any(|s| s.contains("response.failed")),
            "{segs:?}"
        );
        let mut d2 = ResponsesDeframer::new();
        let mut saw_err = false;
        for seg in &segs {
            for (ev, data) in split_segment(seg) {
                if d2.push(&ev, &data).is_err() {
                    saw_err = true;
                }
            }
        }
        assert!(saw_err);
        assert!(d2.finalize().is_ok(), "failed counts as a terminal");
    }
}
