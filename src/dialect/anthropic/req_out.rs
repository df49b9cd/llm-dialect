//! `ItemRequest` → Anthropic Messages request JSON (client direction).
//!
//! Inverse of [`crate::dialect::anthropic::req::from_anthropic`]: renders the
//! canonical items model back onto the `/v1/messages` wire so the same model
//! can drive an Anthropic upstream directly. System-role messages hoist to
//! the top-level `system` field (Anthropic has no system message slot), and a
//! trailing assistant turn is sent as-is — Anthropic natively continues it,
//! which is prefill; the `_prefill` marker is a server-side concern and never
//! appears here.
//!
//! Fidelity rules mirror the parser's:
//! * `extra["_sys_blocks"]` (a system array carrying `cache_control`
//!   breakpoints) is re-emitted verbatim as `system` and the derived System
//!   messages are consumed by it.
//! * `extra["thinking"]` wins over the typed `ThinkingCfg` — the gateway
//!   parse path stores the verbatim thinking config there so exotic shapes
//!   (`{"type":"disabled"}`) reach Anthropic as sent.
//! * Anthropic-native extras (`container`, `mcp_servers`, `service_tier`,
//!   `top_k`, `cache_control`) ride through; everything else drops.
//! * Only Anthropic-signed thinking is rendered: `signature: Some` becomes a
//!   `thinking` block and `redacted_data` becomes `redacted_thinking`;
//!   unsigned thinking has no Anthropic request channel and is dropped
//!   (same rule as the Response-side renderers).

use crate::error::ProxyError;
use crate::items::{ContentItem, ItemRequest, Role, ToolChoice};

/// Anthropic's Messages API requires `max_tokens`; when the caller left it
/// unset we substitute the same default the ingress surface documents.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Extra keys that are Anthropic-native and pass through to the wire.
const PASSTHROUGH_EXTRA: [&str; 5] = [
    "container",
    "mcp_servers",
    "service_tier",
    "top_k",
    "cache_control",
];

pub fn to_anthropic(req: &ItemRequest) -> Result<serde_json::Value, ProxyError> {
    req.validate().map_err(ProxyError::BadRequest)?;

    let mut out = serde_json::json!({
        "model": req.model,
        "max_tokens": req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
    });
    if req.stream {
        out["stream"] = serde_json::json!(true);
    }
    if let Some(t) = req.temperature {
        out["temperature"] = serde_json::json!(t);
    }
    if let Some(p) = req.top_p {
        out["top_p"] = serde_json::json!(p);
    }
    if let Some(stops) = &req.stop_sequences
        && !stops.is_empty()
    {
        out["stop_sequences"] = serde_json::json!(stops);
    }

    render_system(req, &mut out);

    let messages: Vec<serde_json::Value> = req
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(render_message)
        .collect::<Result<_, _>>()?;
    out["messages"] = serde_json::Value::Array(messages);

    if !req.tools.is_empty() {
        out["tools"] = req
            .tools
            .iter()
            .map(|t| {
                let mut tool = serde_json::json!({
                    "name": t.name,
                    "input_schema": t.input_schema,
                });
                if let Some(d) = &t.description {
                    tool["description"] = serde_json::json!(d);
                }
                tool
            })
            .collect();
    }
    match &req.tool_choice {
        // "auto" is the upstream default — omit, same as the parse path.
        None | Some(ToolChoice::Auto) => {}
        Some(ToolChoice::None) => out["tool_choice"] = serde_json::json!({"type": "none"}),
        Some(ToolChoice::Required) => out["tool_choice"] = serde_json::json!({"type": "any"}),
        Some(ToolChoice::Tool { name }) => {
            out["tool_choice"] = serde_json::json!({"type": "tool", "name": name})
        }
    }
    match &req.response_format {
        None => {}
        Some(crate::items::ResponseFormat::Text) => {
            out["response_format"] = serde_json::json!({"type": "text"})
        }
        Some(crate::items::ResponseFormat::JsonObject) => {
            out["response_format"] = serde_json::json!({"type": "json_object"})
        }
        Some(crate::items::ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        }) => {
            out["response_format"] = serde_json::json!({
                "type": "json_schema", "name": name, "schema": schema, "strict": strict,
            })
        }
    }

    // thinking: a verbatim config captured at parse time wins over the typed
    // one — exotic shapes must reach Anthropic as sent.
    if let Some(verbatim) = req.extra.get("thinking") {
        out["thinking"] = verbatim.clone();
    } else if let Some(th) = &req.thinking {
        let mut thinking = serde_json::json!({"type": "enabled"});
        if let Some(b) = th.budget_tokens {
            thinking["budget_tokens"] = serde_json::json!(b);
        }
        if let Some(e) = &th.effort {
            thinking["effort"] = serde_json::json!(e);
        }
        out["thinking"] = thinking;
    }

    // usage-classification metadata rides back out as Anthropic's
    // metadata.user_id (the parser flattened it to OpenAI's `user`).
    if let Some(u) = req.extra.get("user").and_then(|u| u.as_str()) {
        out["metadata"] = serde_json::json!({"user_id": u});
    }
    for key in PASSTHROUGH_EXTRA {
        if let Some(v) = req.extra.get(key) {
            out[key] = v.clone();
        }
    }
    Ok(out)
}

/// Hoist system-role messages into the top-level `system` field. A verbatim
/// `_sys_blocks` array (cache_control breakpoints) wins and consumes the
/// derived system messages; otherwise the system texts join with '\n'.
fn render_system(req: &ItemRequest, out: &mut serde_json::Value) {
    if let Some(blocks) = req.extra.get("_sys_blocks") {
        out["system"] = blocks.clone();
        return;
    }
    let text = req
        .messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| {
            m.items
                .iter()
                .filter_map(|i| match i {
                    ContentItem::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("")
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if !text.is_empty() {
        out["system"] = serde_json::json!(text);
    }
}

fn render_message(m: &crate::items::ItemStreamMessage) -> Result<serde_json::Value, ProxyError> {
    let role = match m.role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => unreachable!("system messages are hoisted by render_system"),
        Role::Tool => {
            return Err(ProxyError::BadRequest(
                "tool-role messages are an OpenAI chat shape; Anthropic carries tool results in user messages".into(),
            ))
        }
    };

    // A lone plain-text item (and no cache markers) renders as the string
    // shorthand — the parse path accepts both forms equally.
    if let [ContentItem::Text { text }] = m.items.as_slice()
        && m.metadata.cache_control.is_empty()
        && m.metadata.name.is_none()
    {
        return Ok(serde_json::json!({"role": role, "content": text}));
    }

    let cc = &m.metadata.cache_control;
    let block_cc = |i: usize| cc.get(i).and_then(|c| c.as_ref());

    let mut blocks: Vec<serde_json::Value> = Vec::new();
    for (i, item) in m.items.iter().enumerate() {
        let mut block = match item {
            ContentItem::Text { text } => serde_json::json!({"type": "text", "text": text}),
            ContentItem::Thinking {
                text,
                signature,
                redacted_data,
                ..
            } => {
                if let Some(data) = redacted_data {
                    serde_json::json!({"type": "redacted_thinking", "data": data})
                } else if let Some(sig) = signature {
                    serde_json::json!({"type": "thinking", "thinking": text, "signature": sig})
                } else {
                    // Unsigned thinking has no Anthropic request channel —
                    // re-emitting it would 400 on a real Anthropic upstream.
                    tracing::debug!("dropping unsigned thinking block on anthropic render");
                    continue;
                }
            }
            ContentItem::ToolCall {
                id,
                name,
                arguments,
            } => serde_json::json!({
                "type": "tool_use", "id": id, "name": name, "input": arguments,
            }),
            ContentItem::ToolResult {
                tool_call_id,
                content,
                is_error,
            } => {
                let content = match content {
                    serde_json::Value::String(_) => content.clone(),
                    other => serde_json::Value::String(other.to_string()),
                };
                let mut b = serde_json::json!({
                    "type": "tool_result", "tool_use_id": tool_call_id, "content": content,
                });
                if *is_error {
                    b["is_error"] = serde_json::json!(true);
                }
                b
            }
            ContentItem::Refusal { .. } => {
                // Anthropic has no refusal block on the request side.
                tracing::debug!("dropping refusal block on anthropic render");
                continue;
            }
        };
        if let Some(c) = block_cc(i) {
            block["cache_control"] = c.clone();
        }
        blocks.push(block);
    }
    Ok(serde_json::json!({"role": role, "content": blocks}))
}

#[cfg(test)]
mod tests {
    use super::super::req::from_anthropic;
    use super::*;
    use crate::items::{ItemMeta, ItemStreamMessage, ThinkingCfg};

    fn req_with(messages: Vec<ItemStreamMessage>) -> ItemRequest {
        ItemRequest {
            model: "claude".into(),
            messages,
            max_tokens: Some(100),
            ..Default::default()
        }
    }

    fn user_text(t: &str) -> ItemStreamMessage {
        ItemStreamMessage {
            role: Role::User,
            items: vec![ContentItem::Text { text: t.into() }],
            metadata: ItemMeta::default(),
        }
    }

    #[test]
    fn plain_text_renders_string_shorthand() {
        let out = to_anthropic(&req_with(vec![user_text("hello")])).unwrap();
        assert_eq!(out["messages"][0]["content"], "hello");
        assert_eq!(out["max_tokens"], 100);
        assert!(out.get("stream").is_none());
    }

    #[test]
    fn missing_max_tokens_substitutes_default() {
        let mut req = req_with(vec![user_text("hi")]);
        req.max_tokens = None;
        let out = to_anthropic(&req).unwrap();
        assert_eq!(out["max_tokens"], DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn system_messages_hoist_to_system_field_joined() {
        let req = ItemRequest {
            messages: vec![
                ItemStreamMessage {
                    role: Role::System,
                    items: vec![ContentItem::Text {
                        text: "be brief".into(),
                    }],
                    metadata: ItemMeta::default(),
                },
                user_text("hi"),
            ],
            ..req_with(vec![])
        };
        let out = to_anthropic(&req).unwrap();
        assert_eq!(out["system"], "be brief");
        assert_eq!(out["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn sys_blocks_extra_wins_verbatim() {
        let blocks = serde_json::json!([
            {"type":"text","text":"long context","cache_control":{"type":"ephemeral"}}
        ]);
        let mut req = req_with(vec![
            ItemStreamMessage {
                role: Role::System,
                items: vec![ContentItem::Text {
                    text: "long context".into(),
                }],
                metadata: ItemMeta::default(),
            },
            user_text("hi"),
        ]);
        req.extra.insert("_sys_blocks".into(), blocks.clone());
        let out = to_anthropic(&req).unwrap();
        assert_eq!(out["system"], blocks);
    }

    #[test]
    fn signed_thinking_renders_unsigned_drops() {
        let req = ItemRequest {
            messages: vec![ItemStreamMessage {
                role: Role::Assistant,
                items: vec![
                    ContentItem::Thinking {
                        text: "plan".into(),
                        signature: Some("sig_ABC".into()),
                        encrypted: None,
                        redacted_data: None,
                    },
                    ContentItem::Thinking {
                        text: "unsigned vllm reasoning".into(),
                        signature: None,
                        encrypted: None,
                        redacted_data: None,
                    },
                    ContentItem::Thinking {
                        text: String::new(),
                        signature: None,
                        encrypted: None,
                        redacted_data: Some("blob".into()),
                    },
                    ContentItem::Text {
                        text: "answer".into(),
                    },
                ],
                metadata: ItemMeta::default(),
            }],
            ..req_with(vec![])
        };
        let out = to_anthropic(&req).unwrap();
        let blocks = out["messages"][0]["content"].as_array().unwrap();
        let types: Vec<&str> = blocks.iter().map(|b| b["type"].as_str().unwrap()).collect();
        assert_eq!(types, vec!["thinking", "redacted_thinking", "text"]);
        assert_eq!(blocks[0]["signature"], "sig_ABC");
        assert_eq!(blocks[1]["data"], "blob");
    }

    #[test]
    fn tool_round_trip_items_to_wire() {
        let mut req = req_with(vec![
            user_text("run it"),
            ItemStreamMessage {
                role: Role::Assistant,
                items: vec![ContentItem::ToolCall {
                    id: "toolu_1".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({"command": "ls"}),
                }],
                metadata: ItemMeta::default(),
            },
            ItemStreamMessage {
                role: Role::User,
                items: vec![ContentItem::ToolResult {
                    tool_call_id: "toolu_1".into(),
                    content: serde_json::json!("file.txt"),
                    is_error: false,
                }],
                metadata: ItemMeta::default(),
            },
        ]);
        req.tools = vec![crate::items::Tool {
            name: "bash".into(),
            description: Some("run a command".into()),
            input_schema: serde_json::json!({"type":"object"}),
        }];
        req.tool_choice = Some(ToolChoice::Required);
        let out = to_anthropic(&req).unwrap();
        assert_eq!(out["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(out["messages"][2]["content"][0]["type"], "tool_result");
        assert!(
            out["messages"][2]["content"][0].get("is_error").is_none(),
            "is_error defaults off and is omitted"
        );
        assert_eq!(out["tools"][0]["name"], "bash");
        assert!(out["tools"][0].get("type").is_none());
        assert_eq!(out["tool_choice"]["type"], "any");
        // and the wire form parses back to the same items
        let back = from_anthropic(&out).unwrap();
        assert_eq!(back.messages.len(), 3);
        assert!(matches!(
            back.messages[1].items[0],
            ContentItem::ToolCall { ref id, .. } if id == "toolu_1"
        ));
    }

    #[test]
    fn tool_choice_none_and_named_tool() {
        for (choice, ty) in [
            (ToolChoice::None, "none"),
            (
                ToolChoice::Tool {
                    name: "bash".into(),
                },
                "tool",
            ),
        ] {
            let mut req = req_with(vec![user_text("hi")]);
            req.tool_choice = Some(choice);
            let out = to_anthropic(&req).unwrap();
            assert_eq!(out["tool_choice"]["type"], ty);
        }
        let mut req = req_with(vec![user_text("hi")]);
        req.tool_choice = Some(ToolChoice::Auto);
        assert!(to_anthropic(&req).unwrap().get("tool_choice").is_none());
    }

    #[test]
    fn typed_thinking_renders_verbatim_extra_wins() {
        let mut req = req_with(vec![user_text("hi")]);
        req.max_tokens = Some(10000);
        req.thinking = Some(ThinkingCfg {
            budget_tokens: Some(4096),
            effort: Some("high".into()),
        });
        let out = to_anthropic(&req).unwrap();
        assert_eq!(out["thinking"]["type"], "enabled");
        assert_eq!(out["thinking"]["budget_tokens"], 4096);

        req.extra
            .insert("thinking".into(), serde_json::json!({"type": "disabled"}));
        let out = to_anthropic(&req).unwrap();
        assert_eq!(out["thinking"], serde_json::json!({"type": "disabled"}));
    }

    #[test]
    fn cache_control_marks_render_per_block() {
        let req = ItemRequest {
            messages: vec![ItemStreamMessage {
                role: Role::User,
                items: vec![
                    ContentItem::Text { text: "a".into() },
                    ContentItem::Text { text: "b".into() },
                ],
                metadata: ItemMeta {
                    cache_control: vec![None, Some(serde_json::json!({"type": "ephemeral"}))],
                    ..ItemMeta::default()
                },
            }],
            ..req_with(vec![])
        };
        let out = to_anthropic(&req).unwrap();
        let blocks = out["messages"][0]["content"].as_array().unwrap();
        assert!(blocks[0].get("cache_control").is_none());
        assert_eq!(blocks[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn user_extra_becomes_metadata_user_id() {
        let mut req = req_with(vec![user_text("hi")]);
        req.extra.insert("user".into(), serde_json::json!("u_123"));
        let out = to_anthropic(&req).unwrap();
        assert_eq!(out["metadata"]["user_id"], "u_123");
    }

    #[test]
    fn anthropic_native_extras_passthrough_others_drop() {
        let mut req = req_with(vec![user_text("hi")]);
        req.extra
            .insert("service_tier".into(), serde_json::json!("standard"));
        req.extra
            .insert("parallel_tool_calls".into(), serde_json::json!(false));
        let out = to_anthropic(&req).unwrap();
        assert_eq!(out["service_tier"], "standard");
        assert!(out.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn tool_role_message_is_an_error() {
        let req = req_with(vec![ItemStreamMessage {
            role: Role::Tool,
            items: vec![ContentItem::Text { text: "x".into() }],
            metadata: ItemMeta::default(),
        }]);
        assert!(to_anthropic(&req).is_err());
    }

    #[test]
    fn parse_render_parse_is_stable() {
        let wire = serde_json::json!({
            "model": "claude", "max_tokens": 100, "stream": true,
            "system": "be brief",
            "temperature": 0.5,
            "messages": [
                {"role":"user","content":"run it"},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"plan","signature":"sig"},
                    {"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"ls"}}
                ]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"toolu_1","content":"file.txt"}
                ]}
            ]
        });
        let once = from_anthropic(&wire).unwrap();
        let twice = from_anthropic(&to_anthropic(&once).unwrap()).unwrap();
        assert_eq!(once, twice);
    }
}
