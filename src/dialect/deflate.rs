//! Deflate: the items canonical → legacy OpenAI-shaped ChatRequest.
//!
//! Pure data transformation — no I/O, no runtime (enforced by the purity
//! lint): an accumulator folds each inbound message's content items into an
//! OpenAI wire message, tool results split off as role:"tool" messages, and
//! typed cfg fields / extras map onto the ChatRequest the pipeline consumes.
//! Moved verbatim from proxy_api.rs (its former home); lives here because it
//! operates on the items model that owns this tree.

use crate::canonical::ChatRequest;
use crate::error::ProxyError;
use crate::items::{ContentItem, ItemRequest, ResponseFormat, Role};

/// Accumulator for one inbound message as it folds into an OpenAI-dialect
/// wire message. `saw_payload` marks whether anything was folded at all (so
/// empty messages never emit a bare `{"role": ...}`).
#[derive(Default)]
struct MsgAcc {
    saw_payload: bool,
    content: String,
    reasoning: String,
    tool_calls: Vec<serde_json::Value>,
    thinking_blocks: Vec<serde_json::Value>,
}

impl MsgAcc {
    /// Fold a single content item; refuses non-text tool results with 400.
    fn fold(
        &mut self,
        item: &ContentItem,
        out_tool_msgs: &mut Vec<serde_json::Value>,
    ) -> Result<(), ProxyError> {
        match item {
            ContentItem::Text { text } => {
                self.saw_payload = true;
                self.content.push_str(text);
            }
            ContentItem::Thinking {
                text,
                signature,
                redacted_data,
                ..
            } => {
                self.saw_payload = true;
                if let Some(data) = redacted_data {
                    self.thinking_blocks.push(serde_json::json!({
                        "type": "redacted_thinking",
                        "data": data,
                    }));
                } else if signature.is_some() {
                    self.thinking_blocks.push(serde_json::json!({
                        "type": "thinking",
                        "thinking": text,
                        "signature": signature,
                    }));
                } else {
                    self.reasoning.push_str(text);
                }
            }
            ContentItem::ToolCall {
                id,
                name,
                arguments,
            } => {
                self.saw_payload = true;
                self.tool_calls.push(serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments.to_string(),
                    },
                }));
            }
            ContentItem::ToolResult {
                tool_call_id: tid,
                content: c,
                is_error,
            } => {
                // Tool messages must precede the coalesced user text (emitted
                // at message end) — OpenAI upstreams require tool messages to
                // follow the assistant tool_calls turn immediately.
                let mut out = match c {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Array(parts) => parts
                        .iter()
                        .filter_map(|p| p["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("\n"),
                    other => {
                        return Err(ProxyError::BadRequest(format!(
                            "tool result content must be text or text parts, got {}",
                            other
                                .as_object()
                                .map(|_| "object")
                                .unwrap_or("non-text value")
                        )));
                    }
                };
                if *is_error {
                    out = format!("[tool error] {out}");
                }
                out_tool_msgs.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": out,
                }));
            }
            ContentItem::Refusal { text } => {
                self.saw_payload = true;
                self.content.push_str(text);
            }
        }
        Ok(())
    }

    /// Emit the accumulated wire message if anything was folded into it.
    fn finish(self, role: &str, name: Option<&str>) -> Option<serde_json::Value> {
        if !self.saw_payload {
            return None;
        }
        let MsgAcc {
            saw_payload: _,
            mut content,
            mut reasoning,
            mut tool_calls,
            mut thinking_blocks,
        } = self;
        let mut msg = serde_json::json!({"role": role});
        if let Some(n) = name {
            msg["name"] = serde_json::json!(n);
        }
        match role {
            "assistant" => {
                // legacy parity: content present unless the turn is
                // tool-call-only (bare null content upstreams reject the
                // explicit null less often than a missing key)
                if !content.is_empty() || tool_calls.is_empty() {
                    msg["content"] = serde_json::Value::String(std::mem::take(&mut content));
                }
                if !reasoning.is_empty() {
                    msg["reasoning_content"] =
                        serde_json::Value::String(std::mem::take(&mut reasoning));
                }
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = serde_json::Value::Array(std::mem::take(&mut tool_calls));
                }
                if !thinking_blocks.is_empty() {
                    msg["thinking_blocks"] =
                        serde_json::Value::Array(std::mem::take(&mut thinking_blocks));
                }
            }
            _ => {
                msg["content"] = serde_json::Value::String(std::mem::take(&mut content));
            }
        }
        Some(msg)
    }
}

/// Canonical tool set → OpenAI `tools` wire array.
fn tools_to_wire(items: &ItemRequest) -> Option<serde_json::Value> {
    if items.tools.is_empty() {
        return None;
    }
    Some(serde_json::Value::Array(
        items
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                })
            })
            .collect(),
    ))
}

fn tool_choice_to_wire(tc: &crate::items::ToolChoice) -> serde_json::Value {
    match tc {
        crate::items::ToolChoice::Auto => serde_json::json!("auto"),
        crate::items::ToolChoice::None => serde_json::json!("none"),
        crate::items::ToolChoice::Required => serde_json::json!("required"),
        crate::items::ToolChoice::Tool { name } => serde_json::json!({
            "type": "function",
            "function": {"name": name},
        }),
    }
}

fn response_format_to_wire(fmt: &ResponseFormat) -> serde_json::Value {
    match fmt {
        ResponseFormat::Text => serde_json::json!({"type":"text"}),
        ResponseFormat::JsonObject => serde_json::json!({"type":"json_object"}),
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => serde_json::json!({
            "type": "json_schema",
            "json_schema": { "name": name, "schema": schema, "strict": strict },
        }),
    }
}

/// Deflate the items canonical back into the legacy OpenAI-shaped ChatRequest
/// the pipeline consumes. Item order is semantic; tool results split off as
/// role:"tool" messages (each carrying its tool_call_id — a bare role:"tool"
/// message 400s on every real upstream).
pub fn items_to_chat_request(items: &ItemRequest) -> Result<ChatRequest, ProxyError> {
    let mut messages: Vec<serde_json::Value> = Vec::new();
    for m in &items.messages {
        let role = match m.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            // a canonical tool message never becomes a wire role:"tool"
            // framing message (no tool_call_id on it); its results split off
            // below, so any accompanying text surfaces as a user turn
            Role::Tool => "user",
        };
        let mut acc = MsgAcc::default();
        for item in &m.items {
            acc.fold(item, &mut messages)?;
        }
        if let Some(msg) = acc.finish(role, m.metadata.name.as_deref()) {
            messages.push(msg);
        }
    }
    let mut v = serde_json::json!({
        "model": items.model,
        "messages": messages,
        "stream": items.stream,
    });
    if let Some(mt) = items.max_tokens {
        v["max_tokens"] = serde_json::json!(mt);
    }
    if let Some(t) = items.temperature {
        v["temperature"] = serde_json::json!(t);
    }
    if let Some(p) = items.top_p {
        v["top_p"] = serde_json::json!(p);
    }
    if let Some(s) = &items.stop_sequences {
        v["stop"] = serde_json::json!(s);
    }
    // A trailing assistant turn carrying text is Anthropic's prefill: the
    // model must continue that text rather than open a new turn. Marked here
    // because only the canonical knows the shape; the provider decides what
    // the upstream can be told (see providers::openai::request_body).
    if items.messages.last().is_some_and(|m| {
        m.role == Role::Assistant
            && m.items.iter().any(
                |i| matches!(i, ContentItem::Text { text } if !text.trim().is_empty()),
            )
            // a turn still holding tool calls is a mid-loop replay awaiting
            // results, not a prefix the model is meant to continue
            && !m
                .items
                .iter()
                .any(|i| matches!(i, ContentItem::ToolCall { .. }))
    }) {
        v[crate::canonical::PREFILL_MARKER] = serde_json::json!(true);
    }
    if let Some(tools) = tools_to_wire(items) {
        v["tools"] = tools;
    }
    if let Some(tc) = &items.tool_choice {
        v["tool_choice"] = tool_choice_to_wire(tc);
    }
    if let Some(fmt) = &items.response_format {
        v["response_format"] = response_format_to_wire(fmt);
    }
    // an Anthropic-origin `thinking` object (possibly {"type":"disabled"})
    // rides in extra and must win over the typed cfg
    if let Some(raw) = items.extra.get("thinking").filter(|x| x.is_object()) {
        v["thinking"] = raw.clone();
    } else if let Some(th) = &items.thinking {
        // only Anthropic-shaped thinking with a real budget is emittable;
        // effort-only maps to reasoning_effort for the dialects that take it
        if let Some(b) = th.budget_tokens {
            v["thinking"] = serde_json::json!({
                "type": "enabled",
                "budget_tokens": b,
            });
        }
        if let Some(e) = &th.effort {
            v["reasoning_effort"] = serde_json::json!(e);
        }
    }
    // extras fill gaps; never clobber the keys this translator just computed.
    // Responses-only fields (captured from that surface's wildcard extras)
    // are dropped here: they 400 on chat-completions upstreams, and the
    // Responses provider never reads them out of the deflated request.
    const RESPONSES_ONLY_EXTRA: &[&str] = &[
        "background",
        "conversation",
        "include",
        "previous_response_id",
        "truncation",
    ];
    for (k, v2) in &items.extra {
        if v.get(k).is_none() && !RESPONSES_ONLY_EXTRA.contains(&k.as_str()) {
            v[k] = v2.clone();
        }
    }
    // Client-derived extras are flattened into ChatRequest fields here; a
    // typed-field collision must surface as a 500 body, not a request-task panic.
    serde_json::from_value(v).map_err(|e| {
        ProxyError::Internal(anyhow::anyhow!(
            "items→chat deflate produced invalid request: {e}"
        ))
    })
}

#[cfg(test)]
mod items_deflate_tests {
    use super::*;

    /// Anthropic prefill: the trailing assistant turn is a prefix to continue,
    /// and plain OpenAI chat cannot say so on its own.
    #[test]
    fn trailing_assistant_text_is_marked_as_prefill() {
        let items = crate::dialect::anthropic::req::from_anthropic(&serde_json::json!({
            "model":"m","max_tokens":100,
            "messages":[
                {"role":"user","content":"Name a colour."},
                {"role":"assistant","content":"The colour is"}
            ]
        }))
        .unwrap();
        let chat = items_to_chat_request(&items).unwrap();
        assert_eq!(
            chat.extra.get(crate::canonical::PREFILL_MARKER),
            Some(&serde_json::json!(true))
        );
    }

    #[test]
    fn trailing_user_turn_is_not_a_prefill() {
        let items = crate::dialect::anthropic::req::from_anthropic(&serde_json::json!({
            "model":"m","max_tokens":100,
            "messages":[{"role":"user","content":"hi"}]
        }))
        .unwrap();
        let chat = items_to_chat_request(&items).unwrap();
        assert!(chat.extra.get(crate::canonical::PREFILL_MARKER).is_none());
    }

    #[test]
    fn trailing_tool_call_turn_is_not_a_prefill() {
        // mid-loop replay awaiting tool results, not a prefix to continue
        let items = crate::dialect::anthropic::req::from_anthropic(&serde_json::json!({
            "model":"m","max_tokens":100,
            "messages":[
                {"role":"user","content":"go"},
                {"role":"assistant","content":[
                    {"type":"text","text":"let me check"},
                    {"type":"tool_use","id":"t1","name":"bash","input":{}}
                ]}
            ]
        }))
        .unwrap();
        let chat = items_to_chat_request(&items).unwrap();
        assert!(chat.extra.get(crate::canonical::PREFILL_MARKER).is_none());
    }

    /// C1 regression: a Responses tool round-trip must never emit a bare
    /// role:"tool" message (no tool_call_id) — upstreams 400 on those.
    #[test]
    fn responses_tool_loop_never_emits_bare_tool_message() {
        let raw = serde_json::json!({
            "model": "m",
            "input": [
                {"type":"message","role":"user","content":"run ls"},
                {"type":"function_call","call_id":"fc_1","name":"bash","arguments":"{}"},
                {"type":"function_call_output","call_id":"fc_1","output":"file.txt"}
            ]
        });
        let items = crate::dialect::openai_responses::req::from_openai_responses(&raw).unwrap();
        let chat = items_to_chat_request(&items).unwrap();
        let v = serde_json::to_value(&chat).unwrap();
        for msg in v["messages"].as_array().unwrap() {
            if msg["role"] == "tool" {
                assert!(
                    msg["tool_call_id"].as_str().is_some_and(|s| !s.is_empty()),
                    "bare tool message without tool_call_id: {msg}"
                );
            }
        }
        // exactly one tool message, paired with fc_1
        let tools: Vec<_> = v["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m["role"] == "tool")
            .collect();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["tool_call_id"], "fc_1");
        // the assistant turn carries the call
        let assistant = v["messages"][1].clone();
        assert_eq!(assistant["role"], "assistant");
        assert_eq!(assistant["tool_calls"][0]["id"], "fc_1");
    }

    #[test]
    fn tool_result_error_prefix_survives_deflation() {
        let items = crate::dialect::anthropic::req::from_anthropic(&serde_json::json!({
            "model":"m","max_tokens":10,
            "messages":[
                {"role":"user","content":"go"},
                {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"bash","input":{}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"boom","is_error":true}]}
            ]
        }))
        .unwrap();
        let chat = items_to_chat_request(&items).unwrap();
        let tool = chat
            .messages
            .iter()
            .find(|m| m.role == "tool")
            .expect("tool message");
        assert_eq!(tool.tool_call_id.as_deref(), Some("t1"));
        assert!(
            tool.text().starts_with("[tool error] "),
            "is_error must annotate the text: {:?}",
            tool.text()
        );
    }

    #[test]
    fn signed_thinking_becomes_thinking_blocks_unsigned_becomes_reasoning() {
        let items = crate::dialect::anthropic::req::from_anthropic(&serde_json::json!({
            "model":"m","max_tokens":100,
            "messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"plan","signature":"sig1"},
                    {"type":"text","text":"answer"}
                ]}
            ]
        }))
        .unwrap();
        let chat = items_to_chat_request(&items).unwrap();
        let assistant = &chat.messages[1];
        let blocks = assistant.thinking_blocks.as_ref().expect("thinking_blocks");
        assert_eq!(blocks[0]["signature"], "sig1");
        assert_eq!(assistant.text(), "answer");
        assert!(assistant.tool_calls.is_none());
        // and no reasoning_content alias for signed blocks
        assert!(assistant.reasoning_content.is_none());
    }

    #[test]
    fn thinking_disabled_stays_disabled() {
        let items = crate::dialect::anthropic::req::from_anthropic(&serde_json::json!({
            "model":"m","max_tokens":100,
            "thinking":{"type":"disabled"},
            "messages":[{"role":"user","content":"hi"}]
        }))
        .unwrap();
        let chat = items_to_chat_request(&items).unwrap();
        assert_eq!(
            chat.extra.get("thinking"),
            Some(&serde_json::json!({"type":"disabled"})),
            "disabled must not flip to enabled"
        );
    }

    #[test]
    fn no_cap_means_no_max_tokens_field() {
        let raw = serde_json::json!({"model":"m","input":"hi"});
        let items = crate::dialect::openai_responses::req::from_openai_responses(&raw).unwrap();
        let chat = items_to_chat_request(&items).unwrap();
        assert!(chat.max_tokens.is_none());
    }

    #[test]
    fn per_message_name_survives_deflation() {
        let raw = serde_json::json!({
            "model": "m",
            "messages": [
                {"role":"user","name":"alice","content":"hi"},
                {"role":"assistant","name":"bot","content":"yo"},
            ]
        });
        let items = crate::dialect::openai_chat::req::from_openai_chat(&raw).unwrap();
        let chat = items_to_chat_request(&items).unwrap();
        let v = serde_json::to_value(&chat).unwrap();
        assert_eq!(v["messages"][0]["name"], "alice");
        assert_eq!(v["messages"][1]["name"], "bot");
    }

    #[test]
    fn responses_only_extras_do_not_reach_chat_upstreams() {
        let raw = serde_json::json!({
            "model": "m",
            "input": "hi",
            "background": true,
            "previous_response_id": "resp_1",
            "conversation": "conv_1",
            "include": ["output_logprobs"],
            "truncation": "auto",
            "store": false,
            "metadata": {"k":"v"},
            "stream_options": {"include_usage": true},
        });
        let items = crate::dialect::openai_responses::req::from_openai_responses(&raw).unwrap();
        let chat = items_to_chat_request(&items).unwrap();
        for k in [
            "background",
            "previous_response_id",
            "conversation",
            "include",
            "truncation",
        ] {
            assert!(!chat.extra.contains_key(k), "{k} must be dropped");
        }
        // chat-legal fields survive; stream_options lands in the typed field
        assert!(chat.extra.contains_key("store"));
        assert!(chat.extra.contains_key("metadata"));
        assert!(chat.stream_options.is_some());
    }

    #[test]
    fn non_text_tool_result_content_is_a_hard_error() {
        // hand-built: every dialect req adapter coerces to strings, so this
        // guards against future ingress paths shipping structured content
        // that would silently stringify
        let mut items =
            crate::dialect::openai_responses::req::from_openai_responses(&serde_json::json!({
                "model": "m",
                "input": [
                    {"type":"function_call","call_id":"fc_1","name":"bash","arguments":"{}"},
                    {"type":"function_call_output","call_id":"fc_1","output":"ok"}
                ]
            }))
            .unwrap();
        for m in &mut items.messages {
            for it in &mut m.items {
                if let ContentItem::ToolResult { content, .. } = it {
                    *content = serde_json::json!({"structured": true});
                }
            }
        }
        assert!(items_to_chat_request(&items).is_err());
    }

    #[test]
    fn unsigned_thinking_deflates_to_reasoning_not_null_signed_block() {
        let items = crate::dialect::anthropic::req::from_anthropic(&serde_json::json!({
            "model":"m","max_tokens":100,
            "messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"plan"},
                    {"type":"text","text":"answer"}
                ]}
            ]
        }))
        .unwrap();
        let chat = items_to_chat_request(&items).unwrap();
        let assistant = &chat.messages[1];
        let v = serde_json::to_value(assistant).unwrap();
        assert!(
            v.get("thinking_blocks").is_none(),
            "unsigned thinking must not emit a thinking block: {v}"
        );
        assert_eq!(assistant.reasoning_content.as_deref(), Some("plan"));
    }

    #[test]
    fn tool_message_precedes_coalesced_user_text() {
        // OpenAI hard-400s on a user message inserted between the assistant
        // tool_calls turn and its tool results.
        let items = crate::dialect::anthropic::req::from_anthropic(&serde_json::json!({
            "model":"m","max_tokens":100,
            "messages":[
                {"role":"user","content":"go"},
                {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"bash","input":{}}]},
                {"role":"user","content":[
                    {"type":"text","text":"note to self"},
                    {"type":"tool_result","tool_use_id":"t1","content":"out"}
                ]}
            ]
        }))
        .unwrap();
        let chat = items_to_chat_request(&items).unwrap();
        let roles: Vec<&str> = chat.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool", "user"]);
        assert_eq!(chat.messages[2].tool_call_id.as_deref(), Some("t1"));
        assert_eq!(chat.messages[3].text(), "note to self");
    }
}
