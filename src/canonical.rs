use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    // Some upstreams deserialize messages into a
    // struct where these are required, non-Option fields — an explicit `null`
    // 400s with "missing field". Omit them instead of serializing as null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Anthropic extended-thinking blocks belonging to this assistant turn,
    /// kept as their JSON form (`{"type":"thinking","thinking":...,"signature":...}`).
    /// Dropped when the outbound dialect can't carry them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_blocks: Option<Vec<serde_json::Value>>,
    /// vLLM/DeepSeek-style unsigned reasoning, carried on assistant turns.
    /// OpenAI-compatible upstreams read it; Anthropic outbound
    /// never sees it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    pub fn text(&self) -> String {
        match &self.content {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Array(parts)) => join_text_parts(parts),
            _ => String::new(),
        }
    }
}

/// Concatenate the "text" entries of an OpenAI content-parts array.
pub fn join_text_parts(parts: &[serde_json::Value]) -> String {
    parts
        .iter()
        .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
        .collect::<Vec<_>>()
        .join("")
}

/// The JSON encoding of a string leaf — quotes and escapes exactly as
/// serde_json emits them — so hand-assembled frames stay byte-identical to
/// the `Value`-tree output they replaced (keys ride in serde's BTreeMap
/// order; dynamic leaves still escape through serde itself).
pub(crate) fn json_str(s: &str) -> String {
    serde_json::to_string(s).expect("string JSON encoding is infallible")
}

/// Append the JSON encoding of `s` (quotes + escapes, serde-identical) onto
/// `buf` without an intermediate allocation — the writer variant of
/// [`json_str`], for hot paths that assemble frames into one buffer.
///
/// Fast path first: streamed text rarely contains a byte that needs JSON
/// escaping, so scan for one (`"` `\` and the C0 controls) and append the
/// string verbatim between quotes; only fall back to serde's escaper when
/// the scan finds a byte it must handle. The two paths emit identical bytes.
pub(crate) fn write_json_str(buf: &mut String, s: &str) {
    // C0 controls, quote, backslash — every byte JSON must escape.
    fn needs_escape(c: char) -> bool {
        matches!(c, '"' | '\\' | '\u{0000}'..='\u{001f}')
    }
    if !s.contains(needs_escape) {
        buf.push('"');
        buf.push_str(s);
        buf.push('"');
        return;
    }
    serde_json::to_writer(StrWrite(buf), s).expect("string JSON encoding is infallible");
}

/// `io::Write` adapter over a `String`. serde_json's escaper emits each
/// escape sequence and each passthrough slice as one write; both are valid
/// UTF-8 on their own, so the validation below always succeeds — it exists
/// to keep the borrow sound, not to police serde.
struct StrWrite<'a>(&'a mut String);

impl std::io::Write for StrWrite<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match std::str::from_utf8(buf) {
            Ok(s) => {
                self.0.push_str(s);
                Ok(buf.len())
            }
            Err(_) => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "non-UTF-8 byte in JSON writer output",
            )),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// `ChatRequest::extra` key meaning "the trailing assistant turn is a prefill
/// the model must continue, not a finished turn". Set by the items deflater,
/// consumed and removed by the provider senders — it names an intent the
/// OpenAI wire has no field for, and must never reach an upstream.
pub const PREFILL_MARKER: &str = "_prefill";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub stream_options: Option<serde_json::Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Cache-INCLUSIVE prompt count (OpenAI convention): fresh + cached_read +
    /// cache_write. Translators normalize to this at the boundary so
    /// `fresh = prompt_tokens - cached_read - cache_write` holds uniformly
    /// (billing and rate-limit math rely on it). Anthropic's wire reports
    /// input_tokens cache-exclusive — both Anthropic translators add the
    /// cache classes in, and the Anthropic client surface subtracts them back.
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Provider-side prompt-cache reads (tokens reused from KV cache)
    #[serde(default)]
    pub cached_read_tokens: u64,
    /// Tokens written to the prompt cache this turn (Anthropic cache_creation,
    /// which bills above the plain input rate)
    #[serde(default)]
    pub cache_write_tokens: u64,
    /// Reasoning/CoT tokens inside `completion_tokens`, when the provider
    /// reports them (OpenAI completion_tokens_details.reasoning_tokens,
    /// Responses output_tokens_details, Gemini thoughtsTokenCount).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: UsageJson,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageJson {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cached_read_tokens: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cache_write_tokens: u64,
    /// Reasoning tokens, carried for the Responses surface. Skipped on the
    /// wire: the chat surface must keep its stock OpenAI shape.
    #[serde(skip)]
    pub reasoning_tokens: Option<u64>,
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

impl ChatResponse {
    pub fn new(model: &str, content: String, finish_reason: Option<String>, usage: Usage) -> Self {
        Self::full(model, content, None, finish_reason, usage)
    }

    pub fn full(
        model: &str,
        content: String,
        tool_calls: Option<serde_json::Value>,
        finish_reason: Option<String>,
        usage: Usage,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        ChatResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
            object: "chat.completion".into(),
            created: now,
            model: model.to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: Some(serde_json::Value::String(content)),
                    name: None,
                    tool_calls,
                    tool_call_id: None,
                    thinking_blocks: None,
                    reasoning_content: None,
                },
                finish_reason,
            }],
            usage: UsageJson {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.prompt_tokens + usage.completion_tokens,
                cached_read_tokens: usage.cached_read_tokens,
                cache_write_tokens: usage.cache_write_tokens,
                reasoning_tokens: usage.reasoning_tokens,
            },
        }
    }
}

/// One streaming chunk in canonical (OpenAI delta) form.
#[derive(Debug, Clone, Default)]
pub struct CanonChunk {
    pub delta_text: String,
    /// OpenAI-shaped streamed tool_call deltas ({index, id, type, function}).
    pub tool_calls: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
    /// Anthropic extended-thinking blocks are streamed as their own content
    /// blocks; carry them through so the Anthropic inbound surface can re-emit
    /// `thinking_delta` / `signature_delta`. `block_index` is the upstream
    /// content-block index; `kind` is "thinking" | "signature".
    pub thinking: Option<ThinkingDelta>,
    /// Known upstream input-token count, when the provider reports it at
    /// stream start (Anthropic's own `message_start`). Used to fill the
    /// passthrough stream's `message_start` with something more useful than 0.
    pub input_tokens: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct ThinkingDelta {
    pub block_index: u64,
    pub kind: &'static str,
    pub text: String,
}

impl CanonChunk {
    pub fn to_sse_json(
        &self,
        id: &str,
        model: &str,
        created: u64,
        include_usage: bool,
    ) -> Option<String> {
        // The early-out gate stays in this thin prologue (one `is_empty` and
        // three `Option::is_none` checks) so the canary-guarded empty-chunk
        // skip keeps its inline fast path; the assembly body below is a
        // separate cold function the prologue tail-calls only when there is
        // work — codegen then sizes each path for its own job.
        if include_usage {
            return self
                .usage
                .as_ref()
                .map(|u| self.usage_frame(id, model, created, u));
        }
        if self.delta_text.is_empty()
            && self.tool_calls.is_none()
            && self.finish_reason.is_none()
            && self.thinking.is_none()
        {
            return None;
        }
        Some(self.delta_frame(id, model, created))
    }

    /// The usage-trailer chunk frame (one-buffer assembly; alphabetical key
    /// order as serde emits it).
    fn usage_frame(&self, id: &str, model: &str, created: u64, u: &Usage) -> String {
        let mut out = String::with_capacity(160 + id.len() + model.len());
        out.push_str("{\"choices\":[],\"created\":");
        out.push_str(&created.to_string());
        out.push_str(",\"id\":");
        write_json_str(&mut out, id);
        out.push_str(",\"model\":");
        write_json_str(&mut out, model);
        out.push_str(",\"object\":\"chat.completion.chunk\",\"usage\":{\"cache_write_tokens\":");
        out.push_str(&u.cache_write_tokens.to_string());
        out.push_str(",\"cached_read_tokens\":");
        out.push_str(&u.cached_read_tokens.to_string());
        out.push_str(",\"completion_tokens\":");
        out.push_str(&u.completion_tokens.to_string());
        out.push_str(",\"prompt_tokens\":");
        out.push_str(&u.prompt_tokens.to_string());
        out.push_str(",\"total_tokens\":");
        out.push_str(&(u.prompt_tokens + u.completion_tokens).to_string());
        out.push_str("}}");
        out
    }

    // omit `content` entirely on tool-call-only deltas — a literal "" confuses
    // strict merge-by-index clients
    //
    /// The delta-chunk frame. The whole frame assembles into ONE buffer — no
    /// intermediate delta/finish_reason/id/model strings; dynamic leaves
    /// escape straight into `out` via the writer, keeping the wire bytes
    /// identical.
    fn delta_frame(&self, id: &str, model: &str, created: u64) -> String {
        let mut out = String::with_capacity(112 + id.len() + model.len() + self.delta_text.len());
        out.push_str("{\"choices\":[{\"delta\":{");
        let mut wrote_key = false;
        if !self.delta_text.is_empty() {
            out.push_str("\"content\":");
            write_json_str(&mut out, &self.delta_text);
            wrote_key = true;
        }
        if let Some(th) = &self.thinking {
            if wrote_key {
                out.push(',');
            }
            wrote_key = true;
            out.push_str("\"thinking\":{\"block_index\":");
            out.push_str(&th.block_index.to_string());
            out.push_str(",\"kind\":");
            write_json_str(&mut out, th.kind);
            out.push_str(",\"text\":");
            write_json_str(&mut out, &th.text);
            out.push('}');
        }
        if let Some(tcs) = &self.tool_calls {
            if wrote_key {
                out.push(',');
            }
            out.push_str("\"tool_calls\":");
            // a Value serializes to the identical bytes it contributed inside
            // the parent tree — embed it without the deep clone
            serde_json::to_writer(StrWrite(&mut out), tcs)
                .expect("Value serialization is infallible");
        }
        out.push_str("},\"finish_reason\":");
        match &self.finish_reason {
            Some(fr) => write_json_str(&mut out, fr),
            None => out.push_str("null"),
        }
        out.push_str(",\"index\":0}],\"created\":");
        out.push_str(&created.to_string());
        out.push_str(",\"id\":");
        write_json_str(&mut out, id);
        out.push_str(",\"model\":");
        write_json_str(&mut out, model);
        out.push_str(",\"object\":\"chat.completion.chunk\"}");
        out
    }
}
