//! Canonical chunk stream → Anthropic Messages SSE framing.
//!
//! Moved from the legacy `anthropic_in.rs`; owns everything between the
//! pipeline's `CanonChunk` stream and the client-visible `event:`/`data:`
//! frames: block-index assignment (thinking/text/tool blocks reopen at fresh
//! indices per Anthropic's stream protocol), usage repair, and the guaranteed
//! terminal `message_delta`/`message_stop`. The framer core is pure —
//! `CanonChunk` in, frame pairs out, no I/O; the pump is the shared SSE
//! driver (`dialect::sse`), and this file's `anthropic_stream_response` is
//! the thin axum shell wiring state machine to pump.

use crate::canonical::{json_str, CanonChunk, Usage};
#[cfg(feature = "axum")]
use crate::error::ProxyError;
#[cfg(feature = "axum")]
use futures::Stream;

/// Sliding window that recognises a client stop sequence inside the streamed
/// text and keeps it from reaching the client.
///
/// A stop sequence can straddle delta boundaries, so the last `max_len - 1`
/// characters of the open block are withheld until either more text arrives
/// or the block closes. Only ever constructed when the request carried stop
/// sequences — the common path never allocates or copies.
#[derive(Debug, Default)]
pub struct StopWindow {
    seqs: Vec<String>,
    max_len: usize,
    /// characters withheld from `block`, still possibly a partial match
    tail: String,
    block: Option<usize>,
    kind: &'static str,
    /// the sequence that fired, once one has
    pub matched: Option<String>,
}

impl StopWindow {
    fn new(seqs: Vec<String>) -> Option<Self> {
        let seqs: Vec<String> = seqs.into_iter().filter(|s| !s.is_empty()).collect();
        let max_len = seqs.iter().map(|s| s.chars().count()).max()?;
        Some(StopWindow {
            seqs,
            max_len,
            tail: String::new(),
            block: None,
            kind: "",
            matched: None,
        })
    }

    /// Feed one delta; returns the prefix that is safe to emit now. Once a
    /// stop has fired everything after it is swallowed — the upstream keeps
    /// streaming its own trailing frames, but Anthropic's contract is that the
    /// turn ended at the sequence.
    fn feed(&mut self, idx: usize, kind: &'static str, incoming: &str) -> String {
        if self.matched.is_some() {
            return String::new();
        }
        if self.block != Some(idx) {
            // caller flushes the previous block before it closes
            self.block = Some(idx);
            self.kind = kind;
            self.tail.clear();
        }
        let mut buf = std::mem::take(&mut self.tail);
        buf.push_str(incoming);
        for s in &self.seqs {
            if let Some(pos) = buf.find(s.as_str()) {
                self.matched = Some(s.clone());
                buf.truncate(pos);
                return buf;
            }
        }
        let keep = self.max_len.saturating_sub(1);
        let split = buf
            .char_indices()
            .rev()
            .take(keep)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(buf.len());
        self.tail = buf.split_off(split);
        buf
    }

    /// Withheld text for the still-open block, surrendered because that block
    /// is about to close without a stop ever firing.
    fn take_tail(&mut self) -> Option<(usize, &'static str, String)> {
        let idx = self.block?;
        if self.tail.is_empty() {
            return None;
        }
        Some((idx, self.kind, std::mem::take(&mut self.tail)))
    }
}

pub struct StreamState {
    /// gates the `message_start` preamble
    pub first: bool,
    /// per-content-block open/closed, indexed by block index
    pub blocks: Vec<bool>,
    /// fixed index for the upstream thinking block, once opened.
    /// Thinking uses indices 0..max_thinking_index; everything else opens after.
    pub thinking_index: Option<usize>,
    /// index the upstream `content` text started at; parallel tool calls come
    /// after this. None until first text delta arrives.
    pub text_index: Option<usize>,
    /// where parallel tool calls start; runs after any thinking/text blocks
    /// that have already been opened.
    pub tool_base_index: Option<usize>,
    /// known-at-preamble input tokens carried through from the upstream
    /// `message_start` chunk (Anthropic) — 0 until the first payload arrives
    pub input_tokens: u64,
    pub stop_reason: Option<String>,
    /// usage from a choice-less trailer chunk that arrived BEFORE any finish
    /// chunk: buffered here and flushed with the terminal message_delta,
    /// instead of terminating the stream early.
    pub pending_usage: Option<TerminalUsage>,
    pub message_stopped: bool,
    /// Present only when the request carried stop sequences.
    pub stop_window: Option<StopWindow>,
}

impl StreamState {
    pub fn new() -> Self {
        Self {
            first: true,
            blocks: Vec::new(),
            thinking_index: None,
            text_index: None,
            tool_base_index: None,
            input_tokens: 0,
            stop_reason: None,
            pending_usage: None,
            message_stopped: false,
            stop_window: None,
        }
    }

    /// State for a request carrying Anthropic `stop_sequences`.
    pub fn with_stop_sequences(seqs: Vec<String>) -> Self {
        Self {
            stop_window: StopWindow::new(seqs),
            ..Self::new()
        }
    }
}

impl Default for StreamState {
    fn default() -> Self {
        Self::new()
    }
}

/// The four usage counters as they appear on a canonical chunk's usage block
/// (input/completion/cached-read/cache-write), named because they recur at
/// several terminal-frame sites. `pending_usage` uses this; compute paths may
/// also build it ad hoc from a fresh usage payload.
#[derive(Debug, Clone, Copy, Default)]
pub struct TerminalUsage {
    pub input: u64,
    pub output: u64,
    pub cached_read: u64,
    pub cache_write: u64,
}

impl TerminalUsage {
    /// From the typed canonical Usage. Ignores reasoning_tokens — Anthropic's
    /// wire has no slot for it (it's folded into output_tokens upstream).
    pub fn from_usage(u: &Usage) -> Self {
        Self {
            input: u.prompt_tokens,
            output: u.completion_tokens,
            cached_read: u.cached_read_tokens,
            cache_write: u.cache_write_tokens,
        }
    }
    /// Anthropic's wire reports fresh-only input; canonical is cache-inclusive.
    fn wire_input(&self, fallback: u64) -> u64 {
        (if self.input > 0 { self.input } else { fallback })
            .saturating_sub(self.cached_read + self.cache_write)
    }
}

/// `content_block_delta` frame pair — the hot per-token shape. Hand-assembled
/// with keys in serde's BTreeMap (alphabetical) order so the framer skips a
/// `Value`-tree allocation per streamed token; dynamic leaves still escape
/// through `json_str`, keeping the wire bytes identical to the `json!`
/// original this replaced.
fn block_delta(idx: usize, delta_json: String) -> (String, String) {
    (
        "content_block_delta".into(),
        format!("{{\"delta\":{delta_json},\"index\":{idx},\"type\":\"content_block_delta\"}}"),
    )
}

/// `content_block_stop` frame pair (same hand-assembly rationale).
fn block_stop(idx: usize) -> (String, String) {
    (
        "content_block_stop".into(),
        format!("{{\"index\":{idx},\"type\":\"content_block_stop\"}}"),
    )
}

fn open_block(out: &mut Vec<(String, String)>, state: &mut StreamState, idx: usize, block: &str) {
    flush_stop_tail(out, state);
    // close everything below idx that is still open — SSE blocks are sequential
    for i in 0..idx {
        if i < state.blocks.len() && state.blocks[i] {
            state.blocks[i] = false;
            out.push(block_stop(i));
        }
    }
    if state.blocks.len() <= idx {
        state.blocks.resize(idx + 1, false);
    }
    state.blocks[idx] = true;
    out.push((
        "content_block_start".into(),
        format!("{{\"content_block\":{block},\"index\":{idx},\"type\":\"content_block_start\"}}"),
    ));
}

/// Emit any text withheld by the stop window into its own block, before that
/// block is closed. No-op on the common (no stop sequences) path.
fn flush_stop_tail(out: &mut Vec<(String, String)>, state: &mut StreamState) {
    let Some((idx, kind, text)) = state.stop_window.as_mut().and_then(StopWindow::take_tail) else {
        return;
    };
    if state.blocks.get(idx) != Some(&true) {
        return;
    }
    let delta = if kind == "thinking" {
        format!(
            "{{\"thinking\":{},\"type\":\"thinking_delta\"}}",
            json_str(&text)
        )
    } else {
        format!("{{\"text\":{},\"type\":\"text_delta\"}}", json_str(&text))
    };
    out.push(block_delta(idx, delta));
}

fn close_block(out: &mut Vec<(String, String)>, state: &mut StreamState, upto: usize) {
    flush_stop_tail(out, state);
    for (i, open) in state.blocks.iter_mut().enumerate().take(upto) {
        if *open {
            *open = false;
            out.push(block_stop(i));
        }
    }
}

/// Returns the next free block index. Slots are dedicated: thinking→0..N,
/// text after that, tools after that. Never reuse an index in one stream.
fn next_index(state: &StreamState) -> usize {
    state.blocks.len()
}

/// Terminal `message_delta` usage. Output tokens always; input tokens
/// whenever known — for non-Anthropic upstreams the prompt count only ever
/// arrives in the trailer, and this frame is the sole place it can surface.
/// Cache counters ride along when present (clients bill on them).
fn terminal_usage_json(input: u64, output: u64, cached_read: u64, cache_write: u64) -> String {
    // keys in serde's alphabetical order: cache_creation < cache_read < input < output
    let mut u = String::from("{");
    if cache_write > 0 {
        u.push_str(&format!("\"cache_creation_input_tokens\":{cache_write},"));
    }
    if cached_read > 0 {
        u.push_str(&format!("\"cache_read_input_tokens\":{cached_read},"));
    }
    u.push_str(&format!("\"input_tokens\":{input},\"output_tokens\":{output}}}"));
    u
}

/// Terminal `message_delta` + `message_stop` carrying a mapped stop reason
/// and usage. Shared by the trailer, finish, and finalizer paths so the
/// exactly-one-terminal-frame invariant lives in one place.
fn emit_terminal(
    out: &mut Vec<(String, String)>,
    state: &mut StreamState,
    stop_reason: String,
    usage: TerminalUsage,
) {
    close_block(out, state, state.blocks.len());
    let prompt = usage.wire_input(state.input_tokens);
    // A stop sequence that actually fired outranks the upstream's verdict:
    // OpenAI-dialect backends report it as a plain "stop", indistinguishable
    // from running out of things to say.
    let matched = state.stop_window.as_ref().and_then(|w| w.matched.clone());
    let (stop_reason, stop_sequence) = match matched {
        Some(s) => ("stop_sequence".to_string(), json_str(&s)),
        None => (stop_reason, "null".to_string()),
    };
    out.push((
        "message_delta".into(),
        format!(
            "{{\"delta\":{{\"stop_reason\":{},\"stop_sequence\":{stop_sequence}}},\"type\":\"message_delta\",\"usage\":{}}}",
            json_str(&stop_reason),
            terminal_usage_json(prompt, usage.output, usage.cached_read, usage.cache_write),
        ),
    ));
    out.push((
        "message_stop".into(),
        "{\"type\":\"message_stop\"}".into(),
    ));
    state.message_stopped = true;
}

/// One canonical chunk → zero or more `event:`/`data:` frame pairs.
///
/// The framer consumes the typed `CanonChunk` directly (no serialize →
/// parse → mutate → re-serialize round-trip). Usage and finish_reason ride
/// the same chunk, so Anthropic's terminal `message_delta` can carry both
/// (the only place usage lands on that wire).
pub fn chunk_to_sse_events(
    chunk: &CanonChunk,
    model: &str,
    state: &mut StreamState,
    msg_id: &str,
) -> Vec<(String, String)> {
    if state.message_stopped {
        return Vec::new();
    }
    let mut out = Vec::new();

    // Anthropic upstream reports the prompt size in its message_start chunk;
    // surface it in our own preamble. Must be read BEFORE the first-chunk
    // message_start emission below — the preamble carries the count, and the
    // Anthropic SDK reads message_start.usage.input_tokens.
    if let Some(n) = chunk.input_tokens.filter(|n| *n > 0) {
        state.input_tokens = n;
    }

    if state.first {
        state.first = false;
        out.push((
            "message_start".into(),
            format!(
                "{{\"message\":{{\"content\":[],\"id\":{},\"model\":{},\"role\":\"assistant\",\"stop_reason\":null,\"stop_sequence\":null,\"type\":\"message\",\"usage\":{{\"input_tokens\":{},\"output_tokens\":0}}}},\"type\":\"message_start\"}}",
                json_str(msg_id),
                json_str(model),
                state.input_tokens,
            ),
        ));
    }

    // Usage-only trailer chunk (no content of any kind — the gemini shape
    // attaches usage to text-bearing chunks, which must still flow through the
    // normal handlers). A chunk carrying BOTH finish_reason and usage falls
    // through to the finish handler below and terminates in the terminal
    // block there.
    let is_trailer = chunk.finish_reason.is_none()
        && chunk.usage.is_some()
        && chunk.delta_text.is_empty()
        && chunk.thinking.is_none()
        && chunk.tool_calls.is_none();
    let merged_usage = chunk.usage.as_ref().map(TerminalUsage::from_usage);
    if is_trailer {
        let Some(sr) = state.stop_reason.take() else {
            // No finish chunk seen yet — this is a mid-stream usage ping (some
            // vLLM builds emit these), not the stream terminator. Buffer the
            // usage for the eventual terminal frame and keep going.
            if let Some(u) = merged_usage {
                state.pending_usage = Some(u);
            }
            return out;
        };
        // trailer AFTER a finish chunk: it carries the terminal usage.
        let usage = merged_usage
            .or(state.pending_usage.take())
            .unwrap_or_default();
        emit_terminal(&mut out, state, sr, usage);
        return out;
    }

    // Extended-thinking deltas: thinking takes a fresh block the first time it
    // appears. If a later block (text/tool) was opened since and closed the
    // thinking block, a renewed thinking delta must open a NEW block — reusing
    // the closed index violates Anthropic's stream protocol.
    if let Some(th) = &chunk.thinking {
        let idx = match state.thinking_index {
            Some(i) if state.blocks.get(i) == Some(&true) => i,
            _ => {
                let i = next_index(state);
                state.thinking_index = Some(i);
                open_block(&mut out, state, i, "{\"thinking\":\"\",\"type\":\"thinking\"}");
                i
            }
        };
        // Reasoning models apply stop sequences to the thinking channel too,
        // so it is filtered exactly like visible text. Signatures are opaque
        // and never scanned.
        let delta_json = match th.kind {
            "signature" => Some(format!(
                "{{\"signature\":{},\"type\":\"signature_delta\"}}",
                json_str(&th.text)
            )),
            _ => match state.stop_window.as_mut() {
                Some(w) => {
                    let emit = w.feed(idx, "thinking", &th.text);
                    (!emit.is_empty()).then(|| {
                        format!(
                            "{{\"thinking\":{},\"type\":\"thinking_delta\"}}",
                            json_str(&emit)
                        )
                    })
                }
                None => Some(format!(
                    "{{\"thinking\":{},\"type\":\"thinking_delta\"}}",
                    json_str(&th.text)
                )),
            },
        };
        if let Some(delta_json) = delta_json {
            out.push(block_delta(idx, delta_json));
        }
    }
    if !chunk.delta_text.is_empty() {
        // Same reopen rule as thinking: if a tool block opened after text and
        // closed it, resumed text gets a fresh block index.
        let idx = match state.text_index {
            Some(i) if state.blocks.get(i) == Some(&true) => i,
            _ => {
                let i = next_index(state);
                state.text_index = Some(i);
                open_block(&mut out, state, i, "{\"text\":\"\",\"type\":\"text\"}");
                i
            }
        };
        let emit = match state.stop_window.as_mut() {
            Some(w) => w.feed(idx, "text", &chunk.delta_text),
            None => chunk.delta_text.clone(),
        };
        if !emit.is_empty() {
            out.push(block_delta(
                idx,
                format!("{{\"text\":{},\"type\":\"text_delta\"}}", json_str(&emit)),
            ));
        }
    }
    // Tool-call deltas: canonical streams them OpenAI-style. Each upstream
    // `index` gets a stable anthropic block index, running right after any
    // thinking + text blocks that already opened.
    if let Some(tcs) = chunk.tool_calls.as_ref().and_then(|t| t.as_array()) {
        if state.tool_base_index.is_none() {
            state.tool_base_index = Some(next_index(state));
        }
        let base = state.tool_base_index.unwrap_or(0);
        for tc in tcs {
            let idx = tc["index"].as_u64().unwrap_or(0) as usize + base;
            let tc_id = tc["id"].as_str().filter(|s| !s.is_empty());
            if let Some(name) = tc["function"]["name"].as_str() {
                // Anthropic requires a stable tool_use id that is unique
                // across the conversation; see anthropic_tool_use_id.
                let id = anthropic_tool_use_id(
                    tc_id,
                    msg_id,
                    tc["index"].as_u64().unwrap_or(0) as usize,
                );
                open_block(
                    &mut out,
                    state,
                    idx,
                    &format!(
                        "{{\"id\":{},\"input\":{{}},\"name\":{},\"type\":\"tool_use\"}}",
                        json_str(&id),
                        json_str(name),
                    ),
                );
            }
            if let Some(args) = tc["function"]["arguments"].as_str()
                && !args.is_empty()
            {
                if state.blocks.get(idx) != Some(&true) {
                    // Also fires when a later text/thinking block reopened
                    // after the tool block closed it: late args can't be
                    // emitted validly (a reopened block would restart partial
                    // JSON), so we drop — loudly.
                    tracing::warn!(
                        index = idx,
                        "dropping tool argument delta for closed/unopened content block"
                    );
                    continue;
                }
                out.push(block_delta(
                    idx,
                    format!(
                        "{{\"partial_json\":{},\"type\":\"input_json_delta\"}}",
                        json_str(args)
                    ),
                ));
            }
        }
    }
    if let Some(fr) = &chunk.finish_reason {
        let sr = map_stop_reason_outbound(fr).to_string();
        if let Some(usage) = merged_usage {
            // usage rode along on the finish chunk: emit the single terminal
            // frame carrying both stop_reason and usage (spec: exactly one
            // message_delta per stream).
            emit_terminal(&mut out, state, sr, usage);
        } else {
            // No usage yet — the trailer (or finalize_stream) will emit the
            // terminal frame; stash the mapped reason for it.
            state.stop_reason = Some(sr);
        }
    }
    out
}

/// Client-facing Anthropic `tool_use` id.
///
/// Anthropic guarantees tool-use ids are unique *within a conversation* —
/// clients key pending tool calls by id, so a repeat is dropped as a duplicate
/// and its tool never runs. Upstreams are not bound by that: OpenAI-dialect
/// backends have been seen returning per-message counters (`Read:0`) that
/// collide the moment the same tool is called again on a later turn. Anything
/// that is not already an Anthropic id is therefore namespaced by the message
/// id, which is unique per response and so unique across the conversation.
pub(super) fn anthropic_tool_use_id(
    upstream_id: Option<&str>,
    msg_id: &str,
    index: usize,
) -> String {
    // The message id is `chatcmpl-<unique>` on OpenAI-dialect upstreams; only
    // the unique half earns its place in a client-facing id.
    let msg_id = msg_id.strip_prefix("chatcmpl-").unwrap_or(msg_id);
    match upstream_id.filter(|s| !s.is_empty()) {
        // Genuine Anthropic passthrough: already unique, kept verbatim so
        // round trips stay byte-identical.
        Some(id) if id.starts_with("toolu_") => id.to_string(),
        Some(id) => {
            let slug: String = id
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .collect();
            format!("toolu_x_{msg_id}_{slug}")
        }
        None => format!("toolu_x_{msg_id}_{index}"),
    }
}

pub(super) fn map_stop_reason_outbound(fr: &str) -> &str {
    match fr {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        // OpenAI/Gemini safety-stop reason has no direct Anthropic analogue;
        // refusal is the closest fit (model declined rather than completed).
        "content_filter" => "refusal",
        // spec-tolerated pass-through so newer Claude stop reasons keep working
        known @ ("end_turn" | "max_tokens" | "stop_sequence" | "tool_use" | "pause_turn"
        | "refusal") => known,
        // unknown → end_turn (strict SDKs reject unrecognised stop reasons)
        _ => "end_turn",
    }
}

/// Terminal frames for a stream that ended without a finish chunk (provider
/// truncated, client-visible end closer). Mirrors the OpenAI surface's
/// guaranteed `[DONE]`. No-op when the message already stopped.
pub fn finalize_stream(state: &mut StreamState) -> Vec<(String, String)> {
    if state.message_stopped {
        return Vec::new();
    }
    state.message_stopped = true;
    if state.first {
        // never emitted anything — nothing to finalize
        return Vec::new();
    }
    let mut out = Vec::new();
    let usage = state.pending_usage.take().unwrap_or_default();
    let sr = state
        .stop_reason
        .take()
        .unwrap_or_else(|| "end_turn".to_string());
    emit_terminal(&mut out, state, sr, usage);
    out
}

/// SSE framing for the Anthropic surface over the accounted canonical stream
/// (idle timeout, billing and circuit-breaker reporting all happen inside
/// `LoggedStream`). The framer core above is pure; this shell only pumps.
#[cfg(feature = "axum")]
pub fn anthropic_stream_response<S>(
    inner: S,
    model: String,
    msg_id: String,
    stop_sequences: Vec<String>,
) -> axum::response::Response
where
    S: Stream<Item = Result<CanonChunk, ProxyError>> + Unpin + Send + 'static,
{
    let state = std::sync::Arc::new(std::sync::Mutex::new(StreamState::with_stop_sequences(
        stop_sequences,
    )));
    let state_done = state.clone();
    crate::dialect::sse::sse_response(
        inner,
        move |out, item| {
            let mut st = state.lock().unwrap();
            match item {
                Ok(chunk) => {
                    for (ev, d) in chunk_to_sse_events(&chunk, &model, &mut st, &msg_id) {
                        out.push(format!("event: {ev}\ndata: {d}\n\n"));
                    }
                }
                Err(e) => {
                    // terminal: suppress the end-of-stream finalizer
                    st.message_stopped = true;
                    let (_, j) = super::out::error_json(&e);
                    out.push(format!("event: error\ndata: {j}\n\n"));
                }
            }
        },
        move |out| {
            let mut st = state_done.lock().unwrap();
            for (ev, d) in finalize_stream(&mut st) {
                out.push(format!("event: {ev}\ndata: {d}\n\n"));
            }
        },
    )
}

#[cfg(all(test, feature = "axum"))]
mod tests {
    use super::*;
    use crate::canonical::{ThinkingDelta, Usage};

    /// Drive a whole streamed turn through the framer and return the frames.
    fn run_stream(stops: Vec<String>, chunks: Vec<CanonChunk>) -> String {
        let mut st = StreamState::with_stop_sequences(stops);
        let mut out = String::new();
        for c in &chunks {
            for (ev, d) in chunk_to_sse_events(c, "m", &mut st, "msg_1") {
                out.push_str(&format!("event: {ev}\ndata: {d}\n\n"));
            }
        }
        out
    }

    fn streamed_text(frames: &str) -> String {
        frames
            .lines()
            .filter(|l| l.starts_with("data: "))
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(&l[6..]).ok())
            .filter(|v| v["type"] == "content_block_delta")
            .filter_map(|v| {
                v["delta"]["text"]
                    .as_str()
                    .or(v["delta"]["thinking"].as_str())
                    .map(str::to_string)
            })
            .collect()
    }

    fn finish_stop() -> CanonChunk {
        CanonChunk {
            finish_reason: Some("stop".into()),
            usage: Some(usage(5, 5)),
            ..Default::default()
        }
    }

    #[test]
    fn streamed_stop_sequence_is_reported_and_withheld() {
        let frames = run_stream(
            vec!["<END>".into()],
            vec![
                text("one two "),
                text("<END>"),
                text(" three"),
                finish_stop(),
            ],
        );
        assert_eq!(streamed_text(&frames), "one two ");
        assert!(
            frames.contains(r#""stop_reason":"stop_sequence""#),
            "{frames}"
        );
        assert!(frames.contains(r#""stop_sequence":"<END>""#), "{frames}");
    }

    #[test]
    fn stop_sequence_split_across_deltas_is_still_caught() {
        // the sequence never appears whole in any single delta
        let frames = run_stream(
            vec!["<END>".into()],
            vec![
                text("keep"),
                text("<E"),
                text("N"),
                text("D> drop"),
                finish_stop(),
            ],
        );
        assert_eq!(streamed_text(&frames), "keep");
        assert!(frames.contains(r#""stop_sequence":"<END>""#), "{frames}");
    }

    #[test]
    fn withheld_tail_is_flushed_when_no_stop_fires() {
        // "<EN" looked like a partial match but the turn ended naturally —
        // the held-back characters must still reach the client.
        let frames = run_stream(
            vec!["<END>".into()],
            vec![text("all of it <EN"), finish_stop()],
        );
        assert_eq!(streamed_text(&frames), "all of it <EN");
        assert!(frames.contains(r#""stop_reason":"end_turn""#), "{frames}");
    }

    #[test]
    fn stop_sequence_in_streamed_thinking_is_caught() {
        let th = |t: &str| CanonChunk {
            thinking: Some(crate::canonical::ThinkingDelta {
                kind: "thinking",
                text: t.into(),
                block_index: 0,
            }),
            ..Default::default()
        };
        let frames = run_stream(
            vec!["FIVE".into()],
            vec![th("ONE TWO "), th("FIVE SIX"), finish_stop()],
        );
        assert_eq!(streamed_text(&frames), "ONE TWO ");
        assert!(frames.contains(r#""stop_sequence":"FIVE""#), "{frames}");
    }

    #[test]
    fn no_stop_sequences_streams_byte_for_byte() {
        let frames = run_stream(vec![], vec![text("a"), text("b"), text("c"), finish_stop()]);
        assert_eq!(streamed_text(&frames), "abc");
        assert!(frames.contains(r#""stop_sequence":null"#), "{frames}");
    }

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

    fn usage_with(p: u64, c: u64, cr: u64, cw: u64) -> Usage {
        Usage {
            prompt_tokens: p,
            completion_tokens: c,
            cached_read_tokens: cr,
            cache_write_tokens: cw,
            reasoning_tokens: None,
        }
    }

    /// Run typed canonical chunks through the framer, return all frame pairs
    /// including the finalizer's.
    fn frames(chunks: Vec<CanonChunk>) -> Vec<(String, String)> {
        let mut st = StreamState::new();
        let mut all = Vec::new();
        for c in &chunks {
            all.extend(chunk_to_sse_events(c, "route-alias", &mut st, "msg_x"));
        }
        all.extend(finalize_stream(&mut st));
        all
    }

    fn types(all: &[(String, String)]) -> Vec<&str> {
        all.iter().map(|(e, _)| e.as_str()).collect()
    }

    fn data_of<'a>(all: &'a [(String, String)], ev: &str) -> Vec<&'a str> {
        all.iter()
            .filter(|(e, _)| e == ev)
            .map(|(_, d)| d.as_str())
            .collect()
    }

    fn starts_indices(all: &[(String, String)]) -> Vec<i64> {
        data_of(all, "content_block_start")
            .iter()
            .filter_map(|d| serde_json::from_str::<serde_json::Value>(d).unwrap()["index"].as_i64())
            .collect()
    }

    #[test]
    fn stream_chunks_emit_message_start_then_text() {
        let mut st = StreamState::new();
        let evs = chunk_to_sse_events(&text("Hi"), "translate-model", &mut st, "msg_1");
        assert_eq!(evs[0].0, "message_start");
        assert!(evs.iter().any(|(t, _)| t == "content_block_delta"));
        assert!(
            !evs.iter().any(|(t, _)| t == "ping"),
            "ping is one-shot preamble noise; real Anthropic streams ping periodically, not here"
        );

        let evs2 = chunk_to_sse_events(
            &CanonChunk {
                finish_reason: Some("stop".into()),
                usage: Some(usage(3, 1)),
                ..text("")
            },
            "translate-model",
            &mut st,
            "msg_1",
        );
        let t = types(&evs2);
        assert!(t.contains(&"message_delta"));
        assert!(t.contains(&"message_stop"));
        let delta = data_of(&evs2, "message_delta")[0];
        assert!(delta.contains("end_turn"));
    }

    #[test]
    fn message_start_carries_preamble_input_tokens() {
        // Anthropic upstream reports the prompt size in its message_start
        // chunk; the preamble must carry it (the SDK reads
        // message_start.usage.input_tokens). Regression guard: the capture
        // must happen BEFORE the preamble emission.
        let mut st = StreamState::new();
        let evs = chunk_to_sse_events(
            &CanonChunk {
                input_tokens: Some(42),
                ..text("")
            },
            "m",
            &mut st,
            "msg_1",
        );
        assert_eq!(evs[0].0, "message_start");
        let ms = serde_json::from_str::<serde_json::Value>(&evs[0].1).unwrap();
        assert_eq!(
            ms["message"]["usage"]["input_tokens"], 42,
            "preamble must carry the upstream prompt count: {ms}"
        );
    }

    #[test]
    fn terminal_usage_reports_fresh_only_input_tokens() {
        // Canonical usage is cache-inclusive; Anthropic clients must see
        // input_tokens the way Anthropic reports them (fresh tokens only).
        let all = frames(vec![
            text("hi"),
            CanonChunk {
                finish_reason: Some("stop".into()),
                usage: Some(usage_with(18204, 7, 18000, 200)),
                ..text("")
            },
        ]);
        let md =
            serde_json::from_str::<serde_json::Value>(data_of(&all, "message_delta")[0]).unwrap();
        assert_eq!(md["usage"]["input_tokens"], 4);
        assert_eq!(md["usage"]["output_tokens"], 7);
        assert_eq!(md["usage"]["cache_read_input_tokens"], 18000);
        assert_eq!(md["usage"]["cache_creation_input_tokens"], 200);
    }

    #[test]
    fn anthropic_upstream_tool_stream_is_well_formed() {
        // finish_reason + usage on one chunk (Anthropic upstream shape): exactly one
        // message_delta carries BOTH the mapped stop_reason and the usage, each
        // content block is stopped exactly once, and nothing is emitted after stop.
        let all = frames(vec![
            text("checking"),
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"id":"toolu_1","function":{"name":"bash","arguments":""}}
                ])),
                ..text("")
            },
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"function":{"arguments":"{}"}}
                ])),
                ..text("")
            },
            CanonChunk {
                finish_reason: Some("tool_calls".into()),
                usage: Some(usage(10, 4)),
                ..text("")
            },
        ]);
        let deltas = data_of(&all, "message_delta");
        assert_eq!(deltas.len(), 1, "exactly one message_delta: {deltas:?}");
        assert!(deltas[0].contains("tool_use"));
        assert!(deltas[0].contains("\"input_tokens\":10"));
        let stops = data_of(&all, "content_block_stop");
        assert_eq!(
            stops.len(),
            2,
            "text block 0 + tool block 1 each closed once"
        );
        assert_eq!(data_of(&all, "message_stop").len(), 1);
    }

    #[test]
    fn reopened_block_gets_fresh_index_when_closed() {
        // thinking → text → thinking interleave: block 0 (thinking) was closed
        // when text opened block 1; the resumed thinking must not reuse 0.
        let all = frames(vec![
            CanonChunk {
                thinking: Some(ThinkingDelta {
                    block_index: 0,
                    kind: "thinking",
                    text: "h1".into(),
                }),
                ..text("")
            },
            text("t"),
            CanonChunk {
                thinking: Some(ThinkingDelta {
                    block_index: 0,
                    kind: "thinking",
                    text: "h2".into(),
                }),
                ..text("")
            },
        ]);
        let indices = starts_indices(&all);
        assert_eq!(
            indices,
            vec![0, 1, 2],
            "blocks must never reuse an index: {indices:?}"
        );
        // every delta must target an index that was started and not yet stopped
        let mut open: std::collections::HashSet<i64> = Default::default();
        let mut seen_started: std::collections::HashSet<i64> = Default::default();
        for (ev, d) in &all {
            let v: serde_json::Value = serde_json::from_str(d).unwrap();
            let idx = v["index"].as_i64();
            match ev.as_str() {
                "content_block_start" => {
                    if let Some(i) = idx {
                        assert!(seen_started.insert(i), "block {i} started twice");
                        open.insert(i);
                    }
                }
                "content_block_stop" => {
                    if let Some(i) = idx {
                        open.remove(&i);
                    }
                }
                "content_block_delta" => {
                    if let Some(i) = idx {
                        assert!(open.contains(&i), "delta on non-open block {i}");
                    }
                }
                _ => {}
            }
        }
    }

    #[test]
    fn thinking_delta_streams_with_own_block_index() {
        let mut st = StreamState::new();
        let mut events: Vec<String> = Vec::new();
        for c in [
            CanonChunk {
                thinking: Some(ThinkingDelta {
                    block_index: 0,
                    kind: "thinking",
                    text: "let me".into(),
                }),
                ..text("")
            },
            CanonChunk {
                thinking: Some(ThinkingDelta {
                    block_index: 0,
                    kind: "thinking",
                    text: " think".into(),
                }),
                ..text("")
            },
            CanonChunk {
                thinking: Some(ThinkingDelta {
                    block_index: 0,
                    kind: "signature",
                    text: "sig123".into(),
                }),
                ..text("")
            },
            text("answer"),
            CanonChunk {
                finish_reason: Some("stop".into()),
                usage: Some(usage(5, 3)),
                ..text("")
            },
        ] {
            for (ev, d) in chunk_to_sse_events(&c, "m", &mut st, "msg_1") {
                events.push(format!("{ev}: {d}"));
            }
        }
        let joined = events.join("\n");
        assert!(
            joined.contains("thinking_delta"),
            "missing thinking_delta: {joined}"
        );
        assert!(
            joined.contains("signature_delta"),
            "missing signature_delta"
        );
        assert!(
            joined.contains("\"index\":0"),
            "thinking block should be index 0"
        );
        // thinking is block 0, so the text answer lands at index 1
        assert!(joined.contains("\"index\":1") && joined.contains("\"type\":\"text\""));
    }

    #[test]
    fn usage_every_chunk_does_not_double_stop() {
        // Gemini-style providers attach usage to every chunk — a usage-only
        // chunk with no finish is a trailer, not a terminator, unless a finish
        // chunk already stashed a stop reason.
        let mut st = StreamState::new();
        let mut all: Vec<(String, String)> = Vec::new();
        for _ in 0..2 {
            all.extend(chunk_to_sse_events(
                &CanonChunk {
                    usage: Some(usage(5, 2)),
                    ..text("")
                },
                "m",
                &mut st,
                "msg_1",
            ));
        }
        // no finish seen: nothing emitted, usage is buffered for the terminal frame
        assert!(!types(&all).contains(&"message_stop"));
        assert!(!st.message_stopped);
        // the real finish chunk terminates once, carrying the buffered usage
        all.extend(chunk_to_sse_events(
            &CanonChunk {
                finish_reason: Some("stop".into()),
                ..text("")
            },
            "m",
            &mut st,
            "msg_1",
        ));
        all.extend(finalize_stream(&mut st));
        assert_eq!(data_of(&all, "message_stop").len(), 1);
        assert_eq!(data_of(&all, "message_delta").len(), 1);
    }

    #[test]
    fn trailer_usage_carries_input_and_cache_tokens() {
        // non-Anthropic upstream: message_start goes out with input_tokens: 0;
        // the trailer must still surface prompt + cache counts in message_delta.
        // Canonical prompt_tokens is cache-inclusive: 151 = fresh 42 + 100 read
        // + 9 written; the client must see the fresh 42 plus cache counts.
        let mut st = StreamState::new();
        let mut evs = chunk_to_sse_events(&text("hi"), "m", &mut st, "msg_1");
        evs.extend(chunk_to_sse_events(
            &CanonChunk {
                finish_reason: Some("stop".into()),
                ..text("")
            },
            "m",
            &mut st,
            "msg_1",
        ));
        evs.extend(chunk_to_sse_events(
            &CanonChunk {
                usage: Some(usage_with(151, 7, 100, 9)),
                ..text("")
            },
            "m",
            &mut st,
            "msg_1",
        ));
        let md = serde_json::from_str::<serde_json::Value>(
            evs.iter()
                .find(|(e, _)| e == "message_delta")
                .map(|(_, d)| d.as_str())
                .unwrap(),
        )
        .unwrap();
        assert_eq!(md["usage"]["input_tokens"], 42);
        assert_eq!(md["usage"]["output_tokens"], 7);
        assert_eq!(md["usage"]["cache_read_input_tokens"], 100);
        assert_eq!(md["usage"]["cache_creation_input_tokens"], 9);
    }

    /// what providers::anthropic::translate_stream emits for a tool-call turn:
    /// exactly one terminal message_delta with the mapped stop reason + usage.
    #[test]
    fn repro_anthropic_upstream_tool_use() {
        let all = frames(vec![
            text("Let me check"),
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"id":"toolu_1","type":"function","function":{"name":"get_weather","arguments":""}}
                ])),
                ..text("")
            },
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"function":{"arguments":"{\"city\":\"Rome\"}"}}
                ])),
                ..text("")
            },
            // message_delta from the Anthropic upstream: finish_reason AND usage
            CanonChunk {
                finish_reason: Some("tool_calls".into()),
                usage: Some(usage(100, 20)),
                ..text("")
            },
        ]);
        let deltas = data_of(&all, "message_delta");
        assert_eq!(deltas.len(), 1);
        assert!(
            deltas[0].contains("\"stop_reason\":\"tool_use\""),
            "{deltas:?}"
        );
        assert!(deltas[0].contains("\"input_tokens\":100"));
        assert_eq!(data_of(&all, "message_stop").len(), 1);
        let stops = data_of(&all, "content_block_stop");
        assert_eq!(stops.len(), 2, "text + tool blocks closed once each");
    }

    #[test]
    fn repro_openai_upstream() {
        // OpenAI upstream splits finish and usage across two chunks: the finish
        // stashes end_turn, the usage trailer terminates with it.
        let all = frames(vec![
            text("Hi"),
            text(" there"),
            CanonChunk {
                finish_reason: Some("stop".into()),
                ..text("")
            },
            CanonChunk {
                usage: Some(usage(5, 2)),
                ..text("")
            },
        ]);
        let deltas = data_of(&all, "message_delta");
        assert_eq!(deltas.len(), 1, "{deltas:?}");
        assert!(
            deltas[0].contains("\"stop_reason\":\"end_turn\""),
            "{deltas:?}"
        );
        assert!(deltas[0].contains("\"input_tokens\":5"));
        assert!(deltas[0].contains("\"output_tokens\":2"));
        assert_eq!(data_of(&all, "message_stop").len(), 1);
    }

    #[test]
    fn repro_gemini_upstream_usage_every_chunk() {
        // Gemini attaches usage to every chunk; the finish-less usage chunks
        // must buffer, and the finish chunk must terminate exactly once.
        let all = frames(vec![
            CanonChunk {
                usage: Some(usage(5, 2)),
                ..text("Hello")
            },
            CanonChunk {
                usage: Some(usage(5, 2)),
                ..text(" world")
            },
            CanonChunk {
                usage: Some(usage(5, 2)),
                finish_reason: Some("stop".into()),
                ..text("!")
            },
        ]);
        assert_eq!(data_of(&all, "message_delta").len(), 1);
        assert_eq!(data_of(&all, "message_stop").len(), 1);
        let text_deltas = data_of(&all, "content_block_delta");
        assert_eq!(text_deltas.len(), 3, "{text_deltas:?}");
    }

    /// tool_use with args split across chunks, then finish (+usage merged).
    #[test]
    fn pure_tool_turn_indices() {
        let mut st = StreamState::new();
        let chunks = [
            CanonChunk {
                thinking: Some(ThinkingDelta {
                    block_index: 0,
                    kind: "thinking",
                    text: "let me check".into(),
                }),
                ..text("")
            },
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"id":"call_1","type":"function","function":{"name":"Bash","arguments":""}}
                ])),
                ..text("")
            },
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"function":{"arguments":"{\"command\":\"ls\"}"}}
                ])),
                ..text("")
            },
            CanonChunk {
                finish_reason: Some("tool_calls".into()),
                usage: Some(usage(100, 30)),
                ..text("")
            },
        ];
        let mut all: Vec<(String, String)> = Vec::new();
        for c in chunks {
            all.extend(chunk_to_sse_events(&c, "m", &mut st, "msg_1"));
        }
        all.extend(finalize_stream(&mut st));
        // block indices in order of opening: thinking then tool_use, contiguous
        assert_eq!(starts_indices(&all), vec![0, 1]);
        let stops: Vec<usize> = data_of(&all, "content_block_stop")
            .iter()
            .filter_map(|d| {
                serde_json::from_str::<serde_json::Value>(d).unwrap()["index"]
                    .as_u64()
                    .map(|x| x as usize)
            })
            .collect();
        assert_eq!(stops, vec![0, 1]);
        let joined = show(&all);
        assert!(joined.contains("\"stop_reason\":\"tool_use\""));
        assert_eq!(data_of(&all, "message_delta").len(), 1);
        assert_eq!(data_of(&all, "message_stop").len(), 1);
    }

    #[test]
    fn mixed_text_plus_tool_in_one_upstream_chunk() {
        let mut st = StreamState::new();
        // text delta and tool_use open in the same canonical chunk
        let c = CanonChunk {
            delta_text: "Checking the code".into(),
            tool_calls: Some(serde_json::json!([
                {"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}
            ])),
            ..Default::default()
        };
        let events = chunk_to_sse_events(&c, "m", &mut st, "msg_1");
        assert_eq!(
            starts_indices(&events),
            vec![0, 1],
            "text block 0 then tool block 1, exactly once"
        );
    }

    #[test]
    fn a_text_tool_text() {
        // text → tool → text: the trailing text after a closed text block must
        // open a fresh block, and tool args must not land on a closed block.
        let all = frames(vec![
            text("Let me check."),
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"id":"call_1","type":"function","function":{"name":"Bash","arguments":"{}"}}
                ])),
                ..text("")
            },
            text(" Done."),
            CanonChunk {
                finish_reason: Some("tool_calls".into()),
                usage: Some(usage(10, 4)),
                ..text("")
            },
        ]);
        assert_eq!(starts_indices(&all), vec![0, 1, 2], "{:?}", show(&all));
        assert_eq!(data_of(&all, "message_stop").len(), 1);
    }

    #[test]
    fn b_tool_then_text() {
        // tool opens first; trailing prose after it must start a new text block
        let all = frames(vec![
            CanonChunk {
                tool_calls: Some(serde_json::json!([
                    {"index":0,"id":"call_1","type":"function","function":{"name":"Bash","arguments":"{}"}}
                ])),
                ..text("")
            },
            text("trailing prose"),
            CanonChunk {
                finish_reason: Some("tool_calls".into()),
                usage: Some(usage(10, 4)),
                ..text("")
            },
        ]);
        assert_eq!(starts_indices(&all), vec![0, 1], "{:?}", show(&all));
        assert_eq!(data_of(&all, "message_stop").len(), 1);
    }

    #[test]
    fn c_text_then_reasoning() {
        // openai.rs hardcodes block_index 0 for reasoning_content; the framer
        // must still assign its own fresh blocks (text 0, thinking 1, text 2)
        let all = frames(vec![
            text("visible"),
            CanonChunk {
                thinking: Some(ThinkingDelta {
                    block_index: 0,
                    kind: "thinking",
                    text: "hidden".into(),
                }),
                ..text("")
            },
            text("more"),
            CanonChunk {
                finish_reason: Some("stop".into()),
                usage: Some(usage(1, 1)),
                ..text("")
            },
        ]);
        assert_eq!(starts_indices(&all), vec![0, 1, 2], "{:?}", show(&all));
        assert_eq!(data_of(&all, "message_stop").len(), 1);
    }

    fn show(all: &[(String, String)]) -> String {
        all.iter()
            .map(|(e, d)| format!("event: {e}\ndata: {d}"))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Mid-stream provider error, end to end through the axum shell: an Err
    /// item after content must emit exactly one `event: error` frame, no
    /// message_stop after it (the error frame is the terminator), and never
    /// leak the upstream error body.
    #[tokio::test]
    async fn midstream_error_terminates_with_error_frame() {
        use crate::error::ProxyError;
        let chunks: Vec<Result<CanonChunk, ProxyError>> = vec![
            Ok(text("partial")),
            Err(ProxyError::upstream(502, "secret upstream body".into())),
        ];
        let resp = anthropic_stream_response(
            Box::pin(futures::stream::iter(chunks)),
            "m".into(),
            "msg_1".into(),
            vec![],
        );
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let s = String::from_utf8(bytes.to_vec()).unwrap();
        assert_eq!(
            s.matches("event: error").count(),
            1,
            "exactly one error frame: {s}"
        );
        let err = s
            .split("\n\n")
            .find(|f| f.starts_with("event: error"))
            .unwrap();
        assert!(err.contains("\"type\":\""), "anthropic error shape: {err}");
        // upstream bodies must never leak into client frames
        assert!(!s.contains("secret upstream body"), "body leaked: {s}");
        assert_eq!(
            s.matches("event: message_stop").count(),
            0,
            "message_stop after an error frame: {s}"
        );
        // content deltas precede the error frame
        assert!(
            s.find("text_delta").expect("no deltas") < s.find("event: error").unwrap(),
            "deltas must precede the error frame: {s}"
        );
    }
}
