//! Canonical chunk stream → OpenAI Responses SSE framing.
//!
//! Emits the full item lifecycle (`response.output_item.added` → channel
//! deltas → per-item close-out → a single terminal `response.completed` /
//! `response.incomplete`), with per-frame sequence numbers and a guaranteed
//! terminal frame even when the upstream truncates. The item lifecycle state
//! machine below is pure; the pump is the shared SSE driver (`dialect::sse`),
//! and this file's `responses_stream_response` is the thin axum shell.

use crate::canonical::CanonChunk;
use crate::error::ProxyError;
#[cfg(feature = "axum")]
use futures::Stream;

/// SSE framing state for the Responses surface: canonical chunks become the
/// full item lifecycle — `response.output_item.added` → channel deltas →
/// `response.output_text.done` / `function_call_arguments.done` /
/// `reasoning_summary_part.done` + `content_part.done` →
/// `response.output_item.done` — and a single terminal
/// `response.completed` / `response.incomplete`.
/// The terminal frame is deferred until the usage trailer (or stream end) so
/// it can carry real token counts, and emitted exactly once — the finalizer
/// guarantees one even if the provider truncated.
#[derive(Clone, Copy, PartialEq)]
enum CurItemKind {
    Text,
    Reasoning,
}

struct ResponsesStreamState {
    first: bool,
    /// output items in stream order; deltas mutate them in place and the
    /// close-out marks them completed
    out: Vec<serde_json::Value>,
    /// the currently-open text/reasoning item (out[] index + kind); deltas
    /// append here and a different item starting closes it first. Tool calls
    /// are tracked separately — parallel calls interleave argument deltas, so
    /// several can be open at once.
    current: Option<(usize, CurItemKind)>,
    /// canonical tool-call index → out[] index (continuation deltas carry no id)
    tool_items: std::collections::HashMap<u64, usize>,
    /// wire id → out[] index for id-bearing calls (index is only reliable
    /// within a chunk for some upstreams; ids stay unique)
    tool_ids: std::collections::HashMap<String, usize>,
    /// out[] positions of opened function_call items not yet closed
    /// (closed together at the terminal frame, in open order)
    open_tools: Vec<usize>,
    usage: Option<crate::canonical::Usage>,
    saw_finish: Option<String>,
    completed_sent: bool,
}

impl ResponsesStreamState {
    /// Emit the close-out frames for the open item, if any, and mark it
    /// completed. Called when another item opens and at the terminal frame.
    fn close_current(&mut self, frames: &mut Vec<(String, serde_json::Value)>) {
        let Some((pos, kind)) = self.current.take() else {
            return;
        };
        let id = self.out[pos]["id"]
            .as_str()
            .or_else(|| self.out[pos]["call_id"].as_str())
            .unwrap_or_default()
            .to_string();
        match kind {
            CurItemKind::Text => {
                let text = self.out[pos]["content"][0]["text"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                self.out[pos]["status"] = serde_json::json!("completed");
                frames.push((
                    "response.output_text.done".into(),
                    serde_json::json!({
                        "type": "response.output_text.done",
                        "item_id": id, "output_index": pos, "content_index": 0,
                        "text": text,
                    }),
                ));
                frames.push((
                    "response.content_part.done".into(),
                    serde_json::json!({
                        "type": "response.content_part.done",
                        "item_id": id, "output_index": pos, "content_index": 0,
                        "part": {"type": "output_text", "text": text, "annotations": []},
                    }),
                ));
            }
            CurItemKind::Reasoning => {
                let text = self.out[pos]["summary"][0]["text"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                frames.push((
                    "response.reasoning_summary_part.done".into(),
                    serde_json::json!({
                        "type": "response.reasoning_summary_part.done",
                        "item_id": id, "output_index": pos, "summary_index": 0,
                        "part": {"type": "summary_text", "text": text},
                    }),
                ));
            }
        }
        frames.push((
            "response.output_item.done".into(),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": pos,
                "item": self.out[pos].clone(),
            }),
        ));
    }

    /// Close every still-open function_call item, in open order. Parallel
    /// calls interleave their argument deltas, so they can only be closed
    /// once the stream finishes producing arguments — i.e. at the terminal
    /// frame.
    fn close_tools(&mut self, frames: &mut Vec<(String, serde_json::Value)>) {
        for pos in std::mem::take(&mut self.open_tools) {
            let id = self.out[pos]["call_id"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let args = self.out[pos]["arguments"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            self.out[pos]["status"] = serde_json::json!("completed");
            frames.push((
                "response.function_call_arguments.done".into(),
                serde_json::json!({
                    "type": "response.function_call_arguments.done",
                    "item_id": id, "output_index": pos, "call_id": id,
                    "arguments": args,
                }),
            ));
            frames.push((
                "response.output_item.done".into(),
                serde_json::json!({
                    "type": "response.output_item.done",
                    "output_index": pos,
                    "item": self.out[pos].clone(),
                }),
            ));
        }
    }

    /// Terminal close-out: text/reasoning item first, then all open tool
    /// calls — matching how the channels appeared on the wire.
    fn close_all(&mut self, frames: &mut Vec<(String, serde_json::Value)>) {
        self.close_current(frames);
        self.close_tools(frames);
    }

    fn open_text_item(&mut self, frames: &mut Vec<(String, serde_json::Value)>) -> usize {
        if let Some((pos, CurItemKind::Text)) = self.current {
            return pos;
        }
        self.close_current(frames);
        let id = format!("msg_{}", uuid::Uuid::new_v4().simple());
        let pos = self.out.len();
        self.out.push(serde_json::json!({
            "type": "message",
            "id": id,
            "role": "assistant",
            "status": "in_progress",
            "content": [{"type": "output_text", "text": ""}],
        }));
        self.current = Some((pos, CurItemKind::Text));
        frames.push((
            "response.output_item.added".into(),
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": pos,
                "item": self.out[pos].clone(),
            }),
        ));
        frames.push((
            "response.content_part.added".into(),
            serde_json::json!({
                "type": "response.content_part.added",
                "item_id": id, "output_index": pos, "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }),
        ));
        pos
    }

    fn open_reasoning_item(&mut self, frames: &mut Vec<(String, serde_json::Value)>) -> usize {
        if let Some((pos, CurItemKind::Reasoning)) = self.current {
            return pos;
        }
        self.close_current(frames);
        let id = format!("rs_{}", uuid::Uuid::new_v4().simple());
        let pos = self.out.len();
        self.out.push(serde_json::json!({
            "type": "reasoning",
            "id": id,
            "summary": [{"type": "summary_text", "text": ""}],
        }));
        self.current = Some((pos, CurItemKind::Reasoning));
        frames.push((
            "response.output_item.added".into(),
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": pos,
                "item": self.out[pos].clone(),
            }),
        ));
        frames.push((
            "response.reasoning_summary_part.added".into(),
            serde_json::json!({
                "type": "response.reasoning_summary_part.added",
                "item_id": id, "output_index": pos, "summary_index": 0,
                "part": {"type": "summary_text", "text": ""},
            }),
        ));
        pos
    }
}

/// The single terminal frame: `response.completed`, or `response.incomplete`
/// when the model hit the output cap / a content filter (mirrors the
/// non-stream status mapping — a truncated stream must never claim
/// "completed").
fn rx_terminal_frame(
    st: &ResponsesStreamState,
    resp_id: &str,
    model: &str,
) -> (String, serde_json::Value) {
    let incomplete_reason = match st.saw_finish.as_deref() {
        Some("length") => Some("max_output_tokens"),
        Some("content_filter") => Some("content_filter"),
        _ => None,
    };
    let mut output = st.out.clone();
    for it in &mut output {
        // reasoning items carry no status field; message/function_call do
        if it.get("status").is_some() {
            it["status"] = serde_json::Value::String(
                if incomplete_reason.is_some() {
                    "incomplete"
                } else {
                    "completed"
                }
                .into(),
            );
        }
    }
    let (event, status) = if incomplete_reason.is_some() {
        ("response.incomplete", "incomplete")
    } else {
        ("response.completed", "completed")
    };
    let mut resp = serde_json::json!({
        "id": resp_id,
        "object": "response",
        "status": status,
        "model": model,
        "output": output,
    });
    if let Some(reason) = incomplete_reason {
        resp["incomplete_details"] = serde_json::json!({"reason": reason});
    }
    // usage always present — a truncated upstream stream must not produce a
    // response.completed whose usage key is missing (clients read it blindly)
    let (pin, pout) = st
        .usage
        .as_ref()
        .map(|u| (u.prompt_tokens, u.completion_tokens))
        .unwrap_or((0, 0));
    let mut usage = serde_json::json!({
        "input_tokens": pin,
        "output_tokens": pout,
        "total_tokens": pin + pout,
    });
    let reasoning = st.usage.as_ref().and_then(|u| u.reasoning_tokens);
    if let Some(r) = reasoning {
        usage["output_tokens_details"] = serde_json::json!({"reasoning_tokens": r});
    }
    resp["usage"] = usage;
    (
        event.into(),
        serde_json::json!({"type": event, "response": resp}),
    )
}

// ---------- pure framer (no runtime types) ----------

type Frame = (String, serde_json::Value);

/// Responses-API SSE framer over a canonical chunk stream: emits
/// `response.created` on first use, per-item frames from the state machine
/// above, a single terminal `response.completed`/`response.incomplete`, and
/// stamps per-frame `sequence_number`. Pure — a driver pumps it.
pub struct ResponsesFramer {
    state: ResponsesStreamState,
    seq: u64,
    response_id: String,
    model: String,
}

impl ResponsesFramer {
    pub fn new(response_id: String, model: String) -> Self {
        Self {
            state: ResponsesStreamState {
                first: true,
                out: Vec::new(),
                current: None,
                tool_items: Default::default(),
                tool_ids: Default::default(),
                open_tools: Vec::new(),
                usage: None,
                saw_finish: None,
                completed_sent: false,
            },
            seq: 0,
            response_id,
            model,
        }
    }

    /// All `event: …\ndata: …\n\n` segments for one canonical item, in order.
    pub fn frame_item(&mut self, item: Result<CanonChunk, ProxyError>) -> Vec<String> {
        let st = &mut self.state;
        let mut frames: Vec<Frame> = Vec::new();
        if st.first {
            st.first = false;
            frames.push((
                "response.created".into(),
                serde_json::json!({
                    "type": "response.created",
                    "response": {
                        "id": self.response_id,
                        "object": "response",
                        "status": "in_progress",
                        "model": self.model,
                        "output": [],
                    }
                }),
            ));
        }
        match item {
            Ok(chunk) => {
                if !chunk.delta_text.is_empty() {
                    let idx = st.open_text_item(&mut frames);
                    let id = st.out[idx]["id"].as_str().unwrap_or_default().to_string();
                    let cur = st.out[idx]["content"][0]["text"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string();
                    st.out[idx]["content"][0]["text"] =
                        serde_json::Value::String(format!("{cur}{}", chunk.delta_text));
                    frames.push((
                        "response.output_text.delta".into(),
                        serde_json::json!({
                            "type": "response.output_text.delta",
                            "item_id": id,
                            "output_index": idx,
                            "content_index": 0,
                            "delta": chunk.delta_text,
                        }),
                    ));
                }
                if let Some(th) = &chunk.thinking {
                    // signature deltas are an Anthropic artifact; Responses
                    // has no channel for them, and leaking the blob as
                    // summary text is worse than dropping it
                    if th.kind != "signature" {
                        let idx = st.open_reasoning_item(&mut frames);
                        let id = st.out[idx]["id"].as_str().unwrap_or_default().to_string();
                        let cur = st.out[idx]["summary"][0]["text"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string();
                        st.out[idx]["summary"][0]["text"] =
                            serde_json::Value::String(format!("{cur}{}", th.text));
                        frames.push((
                            "response.reasoning_summary_text.delta".into(),
                            serde_json::json!({
                                "type": "response.reasoning_summary_text.delta",
                                "item_id": id,
                                "output_index": idx,
                                "summary_index": 0,
                                "delta": th.text,
                            }),
                        ));
                    }
                }
                if let Some(tcs) = &chunk.tool_calls {
                    for tc in tcs.as_array().cloned().unwrap_or_default() {
                        let tc_idx = tc["index"].as_u64().unwrap_or(0);
                        let wire_id = tc["id"]
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .map(str::to_string);
                        // A name-bearing delta opens (or re-homes) a call.
                        // Identity is id-first: index alone collides across
                        // chunks on upstreams with chunk-local indexing.
                        if let Some(name) = tc["function"]["name"].as_str() {
                            // id is authoritative (unknown id = new call);
                            // only id-less chunks resolve by index — an
                            // index-only lookup would merge two id-bearing
                            // calls whose upstream uses chunk-local indices
                            let known = match &wire_id {
                                Some(id) => st.tool_ids.get(id).copied(),
                                None => st.tool_items.get(&tc_idx).copied(),
                            };
                            if known.is_none() {
                                let id = wire_id.clone().unwrap_or_else(|| {
                                    format!("fc_{}_{}", st.out.len(), tc_idx) // fc_ = wire prefix; indexes make it unique
                                });
                                let pos = st.out.len();
                                st.out.push(serde_json::json!({
                                    "type": "function_call",
                                    "id": id,
                                    "call_id": id,
                                    "name": name,
                                    "arguments": "",
                                    "status": "in_progress",
                                }));
                                st.tool_items.insert(tc_idx, pos);
                                if let Some(wid) = &wire_id {
                                    st.tool_ids.insert(wid.clone(), pos);
                                }
                                st.open_tools.push(pos);
                                // a function call closes any open text/reasoning
                                // item — text arriving after it starts a NEW
                                // message item
                                st.close_current(&mut frames);
                                frames.push((
                                    "response.output_item.added".into(),
                                    serde_json::json!({
                                        "type": "response.output_item.added",
                                        "output_index": pos,
                                        "item": st.out[pos].clone(),
                                    }),
                                ));
                            }
                        }
                        if let Some(args) = tc["function"]["arguments"].as_str()
                            && !args.is_empty()
                        {
                            let resolved = wire_id
                                .as_ref()
                                .and_then(|id| st.tool_ids.get(id).copied())
                                .or_else(|| st.tool_items.get(&tc_idx).copied());
                            let Some(pos) = resolved else {
                                continue;
                            };
                            let cur = st.out[pos]["arguments"]
                                .as_str()
                                .unwrap_or_default()
                                .to_string();
                            st.out[pos]["arguments"] =
                                serde_json::Value::String(format!("{cur}{args}"));
                            let id = st.out[pos]["call_id"]
                                .as_str()
                                .unwrap_or_default()
                                .to_string();
                            frames.push((
                                "response.function_call_arguments.delta".into(),
                                serde_json::json!({
                                    "type": "response.function_call_arguments.delta",
                                    "item_id": id,
                                    "output_index": pos,
                                    "call_id": id,
                                    "delta": args,
                                }),
                            ));
                        }
                    }
                }
                if let Some(u) = &chunk.usage {
                    st.usage = Some(u.clone());
                }
                if let Some(fr) = &chunk.finish_reason {
                    st.saw_finish = Some(fr.clone());
                }
                // terminal only once the usage trailer has landed (it
                // follows the finish chunk on OpenAI-shape streams); the
                // finalizer below handles the never-arrived case
                if st.saw_finish.is_some() && st.usage.is_some() && !st.completed_sent {
                    st.completed_sent = true;
                    st.close_all(&mut frames);
                    frames.push(rx_terminal_frame(st, &self.response_id, &self.model));
                }
            }
            Err(e) => {
                st.completed_sent = true; // suppress the finalizer's completed
                // Client-facing messages for upstream failures stay
                // generic: provider error bodies can echo request
                // material and are not client-safe (same rule as
                // state::error_response).
                let msg = match &e {
                    ProxyError::Upstream { status, .. } => {
                        format!("upstream returned status {status}")
                    }
                    other => other.to_string(),
                };
                frames.push((
                    "response.failed".into(),
                    serde_json::json!({
                        "type": "response.failed",
                        "response": {
                            "id": self.response_id,
                            "object": "response",
                            "status": "failed",
                            "error": {"code": "server_error", "message": msg},
                        }
                    }),
                ));
            }
        }
        self.stamp(frames)
    }

    /// Segments ending the stream (empty if completed or nothing was emitted).
    pub fn finish(&mut self) -> Vec<String> {
        let st = &mut self.state;
        if st.completed_sent || st.first {
            // completed already sent, or the stream never produced anything
            return Vec::new();
        }
        st.completed_sent = true;
        let mut frames: Vec<Frame> = Vec::new();
        st.close_all(&mut frames);
        frames.push(rx_terminal_frame(st, &self.response_id, &self.model));
        self.stamp(frames)
    }

    fn stamp(&mut self, frames: Vec<Frame>) -> Vec<String> {
        let mut out = Vec::new();
        for (ev, mut p) in frames {
            let n = self.seq;
            self.seq += 1;
            p["sequence_number"] = serde_json::json!(n);
            out.push(format!("event: {ev}\ndata: {p}\n\n"));
        }
        out
    }
}

/// SSE framing for the Responses surface over the accounted canonical stream.
/// The framer above is pure; this shell only pumps.
#[cfg(feature = "axum")]
pub fn responses_stream_response<S>(
    inner: S,
    response_id: String,
    model: String,
) -> axum::response::Response
where
    S: Stream<Item = Result<crate::canonical::CanonChunk, crate::error::ProxyError>>
        + Unpin
        + Send
        + 'static,
{
    let fr = std::sync::Arc::new(std::sync::Mutex::new(ResponsesFramer::new(
        response_id,
        model,
    )));
    let fr_done = fr.clone();
    crate::dialect::sse::sse_response(
        inner,
        move |out, item| {
            let mut f = fr.lock().unwrap();
            for seg in f.frame_item(item) {
                out.push(seg);
            }
        },
        move |out| {
            let mut f = fr_done.lock().unwrap();
            for seg in f.finish() {
                out.push(seg);
            }
        },
    )
}

#[cfg(all(test, feature = "axum"))]
mod stream_tests {
    use super::*;
    use crate::canonical::Usage;

    fn logged(
        chunks: Vec<Result<CanonChunk, ProxyError>>,
    ) -> futures::stream::BoxStream<'static, Result<CanonChunk, ProxyError>> {
        Box::pin(futures::stream::iter(chunks))
    }

    async fn collect(resp: axum::response::Response) -> String {
        let body = resp.into_body();
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    fn chunk(text: &str) -> CanonChunk {
        CanonChunk {
            delta_text: text.into(),
            tool_calls: None,
            finish_reason: None,
            usage: None,
            thinking: None,
            input_tokens: None,
        }
    }

    #[tokio::test]
    async fn completed_carries_usage_and_output() {
        let chunks = vec![
            Ok(chunk("hello ")),
            Ok(chunk("world")),
            Ok(CanonChunk {
                finish_reason: Some("stop".into()),
                ..chunk("")
            }),
            Ok(CanonChunk {
                usage: Some(Usage {
                    prompt_tokens: 7,
                    completion_tokens: 3,
                    cached_read_tokens: 0,
                    cache_write_tokens: 0,
                    reasoning_tokens: None,
                }),
                ..chunk("")
            }),
        ];
        let r = responses_stream_response(logged(chunks), "resp_1".into(), "m".into());
        let s = collect(r).await;
        assert!(s.contains("response.created"));
        let completed = s
            .split("\n\n")
            .find(|f| f.starts_with("event: response.completed"))
            .unwrap();
        assert!(
            completed.contains("\"input_tokens\":7"),
            "completed missing usage: {completed}"
        );
        assert!(
            completed.contains("hello world"),
            "completed missing output text"
        );
        // exactly one completed event
        assert_eq!(s.matches("event: response.completed").count(), 1);
    }

    #[tokio::test]
    async fn tool_call_continuation_uses_recorded_id() {
        let chunks = vec![
            Ok(CanonChunk {
                tool_calls: Some(
                    serde_json::json!([{"index":0,"id":"call_1","function":{"name":"bash","arguments":""}}]),
                ),
                ..chunk("")
            }),
            Ok(CanonChunk {
                tool_calls: Some(
                    serde_json::json!([{"index":0,"function":{"arguments":"{\"cmd\":"}}]),
                ),
                ..chunk("")
            }),
            Ok(CanonChunk {
                tool_calls: Some(
                    serde_json::json!([{"index":0,"function":{"arguments":"\"ls\"}"}}]),
                ),
                ..chunk("")
            }),
            Ok(CanonChunk {
                finish_reason: Some("tool_calls".into()),
                ..chunk("")
            }),
        ];
        let r = responses_stream_response(logged(chunks), "resp_1".into(), "m".into());
        let s = collect(r).await;
        for frame in s
            .split("\n\n")
            .filter(|f| f.contains("function_call_arguments.delta"))
        {
            assert!(
                frame.contains("\"call_id\":\"call_1\""),
                "arg delta missing id: {frame}"
            );
        }
        // completed is emitted by the finalizer even without a usage trailer
        assert!(s.contains("response.completed"), "no completed: {s}");
        assert!(
            s.contains("\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\""),
            "completed args: {s}"
        );
    }

    #[tokio::test]
    async fn same_index_distinct_ids_do_not_collapse() {
        // gemini-style upstream: chunk-local indices (both calls claim index 0)
        // with unique ids per chunk. The framer must key identity by id.
        let chunks = vec![
            Ok(CanonChunk {
                tool_calls: Some(
                    serde_json::json!([{"index":0,"id":"gemini-A","function":{"name":"a","arguments":"{\"x\":1}"}}]),
                ),
                ..chunk("")
            }),
            Ok(CanonChunk {
                tool_calls: Some(
                    serde_json::json!([{"index":0,"id":"gemini-B","function":{"name":"b","arguments":"{\"y\":2}"}}]),
                ),
                ..chunk("")
            }),
            Ok(CanonChunk {
                finish_reason: Some("tool_calls".into()),
                ..chunk("")
            }),
        ];
        let r = responses_stream_response(logged(chunks), "resp_1".into(), "m".into());
        let s = collect(r).await;
        let completed = s
            .split("\n\n")
            .find(|f| f.starts_with("event: response.completed"))
            .unwrap();
        let data = completed.split("data: ").nth(1).unwrap();
        let v: serde_json::Value = serde_json::from_str(data).unwrap();
        let out = v["response"]["output"].as_array().unwrap();
        let names: Vec<&str> = out
            .iter()
            .filter(|i| i["type"] == "function_call")
            .filter_map(|i| i["name"].as_str())
            .collect();
        assert_eq!(names, vec!["a", "b"], "both calls must survive: {s}");
        let args: Vec<&str> = out
            .iter()
            .filter(|i| i["type"] == "function_call")
            .filter_map(|i| i["arguments"].as_str())
            .collect();
        assert_eq!(
            args,
            vec!["{\"x\":1}", "{\"y\":2}"],
            "args must not mix: {s}"
        );
    }

    #[tokio::test]
    async fn truncated_stream_completed_carries_usage_object() {
        // no usage trailer ever arrives — the terminal frame must still carry
        // a usage object (zeros) because clients read response.usage blindly
        let r = responses_stream_response(
            logged(vec![Ok(chunk("partial"))]),
            "resp_1".into(),
            "m".into(),
        );
        let s = collect(r).await;
        let completed = s
            .split("\n\n")
            .find(|f| f.starts_with("event: response.completed"))
            .unwrap();
        assert!(
            completed.contains("\"input_tokens\":0"),
            "completed without usage object: {completed}"
        );
    }

    #[tokio::test]
    async fn sequence_numbers_strictly_increase_across_frames() {
        let chunks = vec![
            Ok(chunk("a")),
            Ok(CanonChunk {
                finish_reason: Some("stop".into()),
                usage: Some(Usage {
                    prompt_tokens: 1,
                    completion_tokens: 2,
                    cached_read_tokens: 0,
                    cache_write_tokens: 0,
                    reasoning_tokens: None,
                }),
                ..chunk("")
            }),
        ];
        let r = responses_stream_response(logged(chunks), "resp_1".into(), "m".into());
        let s = collect(r).await;
        let mut last = -1i64;
        for line in s
            .split("\n\n")
            .filter(|f| f.contains("\"sequence_number\""))
        {
            let data = line.split("data: ").nth(1).unwrap();
            let v: serde_json::Value = serde_json::from_str(data).unwrap();
            let n = v["sequence_number"].as_i64().unwrap();
            assert!(n > last, "seq not increasing: {n} after {last}");
            last = n;
        }
        assert!(last >= 1, "expected at least 2 frames, got {s}");
    }

    #[tokio::test]
    async fn truncated_stream_still_completes() {
        // provider ends the stream without finish/usage: the finalizer must
        // still emit exactly one response.completed
        let r = responses_stream_response(
            logged(vec![Ok(chunk("partial"))]),
            "resp_1".into(),
            "m".into(),
        );
        let s = collect(r).await;
        assert_eq!(s.matches("event: response.completed").count(), 1, "{s}");
        assert!(s.contains("partial"));
    }

    #[tokio::test]
    async fn text_item_full_lifecycle_frames() {
        // the Responses spec's item lifecycle: output_item.added →
        // content_part.added → deltas → output_text.done → content_part.done →
        // output_item.done → response.completed
        let chunks = vec![
            Ok(chunk("hello")),
            Ok(CanonChunk {
                finish_reason: Some("stop".into()),
                usage: Some(Usage {
                    prompt_tokens: 3,
                    completion_tokens: 1,
                    cached_read_tokens: 0,
                    cache_write_tokens: 0,
                    reasoning_tokens: None,
                }),
                ..chunk("")
            }),
        ];
        let r = responses_stream_response(logged(chunks), "resp_1".into(), "m".into());
        let s = collect(r).await;
        for needle in [
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ] {
            assert!(s.contains(needle), "missing {needle}\n{s}");
        }
        let idx = |needle: &str| s.find(needle).unwrap();
        assert!(idx("response.output_text.done") < idx("event: response.completed"));
        assert_eq!(s.matches("event: response.output_item.done").count(), 1);
        assert!(s.contains("\"text\":\"hello\""));
    }

    #[tokio::test]
    async fn length_finish_streams_response_incomplete() {
        let chunks = vec![
            Ok(chunk("partial")),
            Ok(CanonChunk {
                finish_reason: Some("length".into()),
                usage: Some(Usage {
                    prompt_tokens: 5,
                    completion_tokens: 100,
                    cached_read_tokens: 0,
                    cache_write_tokens: 0,
                    reasoning_tokens: None,
                }),
                ..chunk("")
            }),
        ];
        let r = responses_stream_response(logged(chunks), "resp_1".into(), "m".into());
        let s = collect(r).await;
        assert!(s.contains("event: response.incomplete"), "{s}");
        assert!(s.contains("max_output_tokens"), "{s}");
        assert!(!s.contains("event: response.completed"), "{s}");
    }

    #[tokio::test]
    async fn parallel_tool_calls_get_arguments_done_each() {
        let mk_open = |idx: u64, id: &str, name: &str| {
            Ok(CanonChunk {
                tool_calls: Some(serde_json::json!([{
                    "index": idx, "id": id, "type": "function",
                    "function": {"name": name, "arguments": ""}
                }])),
                ..chunk("")
            })
        };
        let mk_args = |idx: u64, a: &str| {
            Ok(CanonChunk {
                tool_calls: Some(serde_json::json!([{
                    "index": idx, "function": {"arguments": a}
                }])),
                ..chunk("")
            })
        };
        let chunks = vec![
            mk_open(0, "call_1", "a"),
            mk_open(1, "call_2", "b"),
            mk_args(0, "{\"x\":1}"),
            mk_args(1, "{\"y\":2}"),
            Ok(CanonChunk {
                finish_reason: Some("tool_calls".into()),
                usage: Some(Usage {
                    prompt_tokens: 1,
                    completion_tokens: 2,
                    cached_read_tokens: 0,
                    cache_write_tokens: 0,
                    reasoning_tokens: None,
                }),
                ..chunk("")
            }),
        ];
        let r = responses_stream_response(logged(chunks), "resp_1".into(), "m".into());
        let s = collect(r).await;
        assert_eq!(
            s.matches("event: response.function_call_arguments.done")
                .count(),
            2,
            "{s}"
        );
        assert!(s.contains("\"arguments\":\"{\\\"x\\\":1}\""), "{s}");
        assert!(s.contains("\"arguments\":\"{\\\"y\\\":2}\""), "{s}");
        assert!(s.contains("event: response.completed"));
    }

    #[tokio::test]
    async fn midstream_error_emits_failed_and_no_completed() {
        // an Err item after content: the surface must emit response.failed,
        // never a response.completed after it, and never claim the turn
        // succeeded (clients key retry logic off this frame)
        let chunks = vec![
            Ok(chunk("partial text")),
            Err(ProxyError::upstream(
                502,
                "provider blew up mid-stream".into(),
            )),
        ];
        let r = responses_stream_response(logged(chunks), "resp_1".into(), "m".into());
        let s = collect(r).await;
        let failed = s
            .split("\n\n")
            .find(|f| f.starts_with("event: response.failed"))
            .expect("response.failed missing");
        assert!(
            failed.contains("\"status\":\"failed\""),
            "failed frame shape: {failed}"
        );
        // upstream bodies must never leak into client frames
        assert!(
            !s.contains("provider blew up mid-stream"),
            "upstream body leaked: {s}"
        );
        assert_eq!(
            s.matches("event: response.completed").count(),
            0,
            "completed after a failed frame: {s}"
        );
        // items opened before the failure are closed out first
        assert!(
            s.find("response.output_text.delta").expect("no deltas")
                < s.find("event: response.failed").unwrap(),
            "deltas must precede the failed frame: {s}"
        );
    }
}
