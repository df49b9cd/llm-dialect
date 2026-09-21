//! `ItemRequest` → OpenAI Responses request JSON (client direction).
//!
//! Inverse of [`from_openai_responses`](crate::dialect::openai_responses::req::from_openai_responses):
//! renders the canonical items model onto the `/v1/responses` wire. Responses
//! is natively item-shaped, so every message's items flatten into the
//! top-level `input` array — with the same leading `./trailing input item
//! hoist the parser applies:
//!
//! * System-role messages hoist to top-level `instructions` (texts joined
//!   with '\n', the inverse of the parser's fold).
//! * `Reasoning` items render as `reasoning` input items — `summary[]`
//!   carries the text (split on '\n'), `encrypted_content` carries the
//!   encrypted payload, so multi-turn CoT continuity round-trips.
//!   Anthropic-signed thinking has no Responses request channel — the wire
//!   contract expects `rs_`-chain continuity, not foreign signatures — and
//!   is dropped.
//! * Tool calls render as top-level `function_call` input items with the
//!   canonical id as `call_id`, never inside a message (the inverse of the
//!   parser, which hoists them). Tool results render as
//!   `function_call_output` items.
//! * `Role::Tool` collapses into the flat input item the parser produces
//!   (its role is informational — the `tool_call_id` references the call).
//!
//! Response-format fields nest under `text.format` as the schema requires.

use crate::error::ProxyError;
use crate::items::{ContentItem, ItemRequest, ResponseFormat, Role, ToolChoice};

pub fn to_openai_responses(req: &ItemRequest) -> Result<serde_json::Value, ProxyError> {
    req.validate().map_err(ProxyError::BadRequest)?;

    let mut out = serde_json::json!({"model": req.model});
    if req.stream {
        out["stream"] = serde_json::json!(true);
    }
    if let Some(t) = req.temperature {
        out["temperature"] = serde_json::json!(t);
    }
    if let Some(p) = req.top_p {
        out["top_p"] = serde_json::json!(p);
    }
    if let Some(mt) = req.max_tokens {
        out["max_output_tokens"] = serde_json::json!(mt);
    }
    if let Some(stops) = &req.stop_sequences
        && !stops.is_empty()
    {
        out["stop"] = serde_json::json!(stops);
    }

    // System-role messages hoist to `instructions` — the same strings the
    // parser folded down, joined with '\n'.
    let instructions = req
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
    if !instructions.is_empty() {
        out["instructions"] = serde_json::json!(instructions);
    }

    let mut input: Vec<serde_json::Value> = Vec::new();
    for m in req.messages.iter().filter(|m| m.role != Role::System) {
        // Responses is item-shaped: every ContentItem becomes one input item,
        // in order. Message-role items wrap in {type:"message", role, content}.
        let mut content: Vec<serde_json::Value> = Vec::new();
        let role = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => {
                // The tool-role marker is informational only; its ToolResult
                // items render flat the same as tool results on any other
                // message.
                "user"
            }
            Role::System => unreachable!("system messages hoisted above"),
        };
        for item in &m.items {
            match item {
                ContentItem::Text { text } => {
                    let part_type = if m.role == Role::Assistant {
                        "output_text"
                    } else {
                        "input_text"
                    };
                    content.push(serde_json::json!({"type": part_type, "text": text}));
                }
                ContentItem::Thinking {
                    text,
                    encrypted,
                    signature,
                    redacted_data,
                } => {
                    if redacted_data.is_some() {
                        // Anthropic redacted thinking has no Responses channel.
                        tracing::debug!("dropping redacted thinking on responses render");
                        continue;
                    }
                    if signature.is_some() {
                        // Anthropic-signed thinking can't ride a Responses
                        // request — only encrypted content or plain summaries.
                        tracing::debug!("dropping anthropic-signed thinking on responses render");
                        continue;
                    }
                    // plaintext summary items; llama-server-style plain
                    // summaries split on '\n' to re-form the parts array.
                    let summary: Vec<serde_json::Value> = if text.is_empty() {
                        Vec::new()
                    } else {
                        text.split('\n')
                            .map(|t| serde_json::json!({"type": "summary_text", "text": t}))
                            .collect()
                    };
                    let mut item = serde_json::json!({"type": "reasoning"});
                    if !summary.is_empty() {
                        item["summary"] = serde_json::Value::Array(summary);
                    }
                    if let Some(enc) = encrypted {
                        item["encrypted_content"] = serde_json::json!(enc);
                    }
                    input.push(item);
                }
                ContentItem::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    // `arguments` on the wire is a JSON-encoded string; the
                    // canonical model holds the parsed value.
                    input.push(serde_json::json!({
                        "type": "function_call",
                        "call_id": id,
                        "name": name,
                        "arguments": arguments.to_string(),
                    }));
                }
                ContentItem::ToolResult {
                    tool_call_id,
                    content,
                    ..
                } => {
                    input.push(render_tool_result(tool_call_id, content)?);
                }
                ContentItem::Refusal { text } => {
                    content.push(serde_json::json!({"type": "refusal", "refusal": text}));
                }
            }
        }
        if !content.is_empty() {
            input.push(serde_json::json!({
                "type": "message",
                "role": role,
                "content": content,
            }));
        }
    }
    // A lone user-text request renders with the string shorthand the parser
    // accepts — otherwise the array form.
    let input_value = if let [serde_json::Value::Object(item)] = input.as_slice()
        && item.get("type").and_then(|t| t.as_str()) == Some("message")
        && item.get("role").and_then(|r| r.as_str()) == Some("user")
        && let Some(content) = item.get("content").and_then(|c| c.as_array())
        && content.len() == 1
        && content[0]["type"] == "input_text"
        && let Some(text) = content[0]["text"].as_str()
    {
        serde_json::json!(text)
    } else if input.is_empty() {
        // Everything got dropped (e.g. Anthropic-signed thinking with no user
        // text): degrade to a minimal valid input rather than error — a bare
        // empty message is parseable by the responses dialect as a no-op turn.
        serde_json::json!([{"type": "message", "role": "user", "content": []}])
    } else {
        serde_json::Value::Array(input)
    };
    out["input"] = input_value;

    if !req.tools.is_empty() {
        out["tools"] = req
            .tools
            .iter()
            .map(|t| {
                // Responses keeps tool fields flat (name/description/
                // parameters at the top level), not nested under `function`.
                let mut tool = serde_json::json!({
                    "type": "function",
                    "name": t.name,
                    "parameters": t.input_schema,
                });
                if let Some(d) = &t.description {
                    tool["description"] = serde_json::json!(d);
                }
                tool
            })
            .collect::<Vec<_>>()
            .into();
    }
    match &req.tool_choice {
        None => {}
        Some(ToolChoice::Auto) => out["tool_choice"] = serde_json::json!("auto"),
        Some(ToolChoice::None) => out["tool_choice"] = serde_json::json!("none"),
        Some(ToolChoice::Required) => out["tool_choice"] = serde_json::json!("required"),
        Some(ToolChoice::Tool { name }) => {
            out["tool_choice"] = serde_json::json!({"type": "function", "name": name})
        }
    }
    if let Some(rf) = &req.response_format {
        let format = match rf {
            ResponseFormat::Text => serde_json::json!({"type": "text"}),
            ResponseFormat::JsonObject => serde_json::json!({"type": "json_object"}),
            ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            } => serde_json::json!({
                "type": "json_schema", "name": name, "schema": schema, "strict": strict,
            }),
        };
        out["text"] = serde_json::json!({"format": format});
    }
    if let Some(th) = &req.thinking {
        let mut r = serde_json::json!({});
        if let Some(b) = th.budget_tokens {
            r["budget_tokens"] = serde_json::json!(b);
        }
        if let Some(e) = &th.effort {
            r["effort"] = serde_json::json!(e);
        }
        out["reasoning"] = r;
    }

    // Responses-native extras ride through; `user` is Anthropic's
    // metadata.user_id channel here, not an OpenAI field.
    for key in [
        "store",
        "metadata",
        "service_tier",
        "truncation",
        "parallel_tool_calls",
        "stream_options",
        "previous_response_id",
        "conversation",
        "include",
    ] {
        if let Some(v) = req.extra.get(key) {
            out[key] = v.clone();
        }
    }
    if let Some(u) = req.extra.get("user") {
        out["user"] = u.clone();
    }
    Ok(out)
}

/// One `ToolResult` → the structured `function_call_output` Responses item.
/// Non-string content is a hard error, mirroring the parser (which refuses
/// multimodal output parts instead of silently flattening them); strings go
/// through as-is.
fn render_tool_result(
    tool_call_id: &str,
    content: &serde_json::Value,
) -> Result<serde_json::Value, ProxyError> {
    let text = match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => {
            return Err(ProxyError::BadRequest(format!(
                "function_call_output content must be a string, got {}",
                other.to_string().chars().take(60).collect::<String>()
            )));
        }
    };
    Ok(serde_json::json!({
        "type": "function_call_output",
        "call_id": tool_call_id,
        "output": text,
    }))
}

#[cfg(test)]
mod tests {
    use super::super::req::from_openai_responses;
    use super::*;
    use crate::items::{ItemMeta, ItemStreamMessage, ThinkingCfg};

    fn req_with(messages: Vec<ItemStreamMessage>) -> ItemRequest {
        ItemRequest {
            model: "gpt".into(),
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
    fn lone_user_text_renders_string_shorthand() {
        let out = to_openai_responses(&req_with(vec![user_text("hello")])).unwrap();
        assert_eq!(out["input"], "hello");
        assert_eq!(out["max_output_tokens"], 100);
    }

    #[test]
    fn multi_turn_renders_array() {
        let req = req_with(vec![
            user_text("q"),
            ItemStreamMessage {
                role: Role::Assistant,
                items: vec![ContentItem::Text { text: "a".into() }],
                metadata: ItemMeta::default(),
            },
        ]);
        let out = to_openai_responses(&req).unwrap();
        let input = out["input"].as_array().unwrap();
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
    }

    #[test]
    fn system_hoists_to_instructions() {
        let req = req_with(vec![
            ItemStreamMessage {
                role: Role::System,
                items: vec![ContentItem::Text {
                    text: "be terse".into(),
                }],
                metadata: ItemMeta::default(),
            },
            user_text("hi"),
        ]);
        let out = to_openai_responses(&req).unwrap();
        assert_eq!(out["instructions"], "be terse");
        assert!(
            !out["input"].to_string().contains("be terse"),
            "instructions extracted, not doubled in input"
        );
    }

    #[test]
    fn reasoning_round_trips_summary_and_encrypted() {
        let req = req_with(vec![
            user_text("q"),
            ItemStreamMessage {
                role: Role::Assistant,
                items: vec![ContentItem::Thinking {
                    text: "line one\nline two".into(),
                    signature: None,
                    encrypted: Some("enc_xyz".into()),
                    redacted_data: None,
                }],
                metadata: ItemMeta::default(),
            },
        ]);
        let out = to_openai_responses(&req).unwrap();
        let input = out["input"].as_array().unwrap();
        assert_eq!(input[1]["type"], "reasoning");
        assert_eq!(input[1]["summary"][0]["text"], "line one");
        assert_eq!(input[1]["summary"][1]["text"], "line two");
        assert_eq!(input[1]["encrypted_content"], "enc_xyz");
    }

    #[test]
    fn anthropic_signed_thinking_drops() {
        let req = req_with(vec![
            user_text("q"),
            ItemStreamMessage {
                role: Role::Assistant,
                items: vec![ContentItem::Thinking {
                    text: "plan".into(),
                    signature: Some("sig".into()),
                    encrypted: None,
                    redacted_data: None,
                }],
                metadata: ItemMeta::default(),
            },
        ]);
        let out = to_openai_responses(&req).unwrap();
        // lone user text collapsed to the string shorthand; the signed
        // thinking item was dropped because Anthropic-signed has no
        // Responses channel.
        assert_eq!(out["input"], "q");
    }

    #[test]
    fn tool_call_and_result_render_flat() {
        let req = req_with(vec![
            user_text("run it"),
            ItemStreamMessage {
                role: Role::Assistant,
                items: vec![ContentItem::ToolCall {
                    id: "fc_1".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({"cmd":"ls"}),
                }],
                metadata: ItemMeta::default(),
            },
            ItemStreamMessage {
                role: Role::Tool,
                items: vec![ContentItem::ToolResult {
                    tool_call_id: "fc_1".into(),
                    content: serde_json::json!("file.txt"),
                    is_error: false,
                }],
                metadata: ItemMeta::default(),
            },
        ]);
        let out = to_openai_responses(&req).unwrap();
        let input = out["input"].as_array().unwrap();
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "fc_1");
        assert_eq!(input[1]["arguments"], "{\"cmd\":\"ls\"}");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["output"], "file.txt");
    }

    #[test]
    fn user_role_tool_result_renders_outside_message() {
        // ToolResult inside a user-role message must still render as a
        // top-level function_call_output, not nested in the message.
        let req = req_with(vec![
            ItemStreamMessage {
                role: Role::Assistant,
                items: vec![ContentItem::ToolCall {
                    id: "fc_1".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({}),
                }],
                metadata: ItemMeta::default(),
            },
            ItemStreamMessage {
                role: Role::User,
                items: vec![ContentItem::ToolResult {
                    tool_call_id: "fc_1".into(),
                    content: serde_json::json!("out"),
                    is_error: false,
                }],
                metadata: ItemMeta::default(),
            },
        ]);
        let out = to_openai_responses(&req).unwrap();
        let input = out["input"].as_array().unwrap();
        assert_eq!(input[1]["type"], "function_call_output");
    }

    #[test]
    fn non_text_tool_result_is_a_hard_error() {
        let req = req_with(vec![
            ItemStreamMessage {
                role: Role::Assistant,
                items: vec![ContentItem::ToolCall {
                    id: "fc_1".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({}),
                }],
                metadata: ItemMeta::default(),
            },
            ItemStreamMessage {
                role: Role::User,
                items: vec![ContentItem::ToolResult {
                    tool_call_id: "fc_1".into(),
                    content: serde_json::json!({"structured": true}),
                    is_error: false,
                }],
                metadata: ItemMeta::default(),
            },
        ]);
        assert!(to_openai_responses(&req).is_err());
    }

    #[test]
    fn tools_render_flat_fields() {
        let mut req = req_with(vec![user_text("hi")]);
        req.tools = vec![crate::items::Tool {
            name: "bash".into(),
            description: Some("run".into()),
            input_schema: serde_json::json!({"type":"object"}),
        }];
        req.tool_choice = Some(ToolChoice::Required);
        let out = to_openai_responses(&req).unwrap();
        assert_eq!(out["tools"][0]["type"], "function");
        assert_eq!(out["tools"][0]["name"], "bash");
        assert!(out["tools"][0].get("function").is_none());
        assert_eq!(out["tool_choice"], "required");
    }

    #[test]
    fn response_format_nests_under_text_format() {
        let mut req = req_with(vec![user_text("hi")]);
        req.response_format = Some(ResponseFormat::JsonSchema {
            name: "answer".into(),
            schema: serde_json::json!({"type":"object"}),
            strict: true,
        });
        let out = to_openai_responses(&req).unwrap();
        assert_eq!(out["text"]["format"]["type"], "json_schema");
        assert_eq!(out["text"]["format"]["strict"], true);
    }

    #[test]
    fn thinking_maps_to_reasoning() {
        let mut req = req_with(vec![user_text("hi")]);
        req.thinking = Some(ThinkingCfg {
            budget_tokens: Some(2048),
            effort: Some("high".into()),
        });
        let out = to_openai_responses(&req).unwrap();
        assert_eq!(out["reasoning"]["effort"], "high");
        assert_eq!(out["reasoning"]["budget_tokens"], 2048);
    }

    #[test]
    fn responses_native_extras_passthrough() {
        let mut req = req_with(vec![user_text("hi")]);
        req.extra
            .insert("previous_response_id".into(), serde_json::json!("resp_1"));
        req.extra
            .insert("service_tier".into(), serde_json::json!("flex"));
        req.extra
            .insert("container".into(), serde_json::json!({"id":"c"})); // anthropic-only: drops
        let out = to_openai_responses(&req).unwrap();
        assert_eq!(out["previous_response_id"], "resp_1");
        assert_eq!(out["service_tier"], "flex");
        assert!(out.get("container").is_none());
    }

    #[test]
    fn parse_render_parse_is_stable() {
        let wire = serde_json::json!({
            "model": "gpt", "input": [
                {"type":"message","role":"user","content":"run it"},
                {"type":"reasoning","encrypted_content":"enc",
                 "summary":[{"type":"summary_text","text":"the plan"}]},
                {"type":"function_call","call_id":"fc_1","name":"bash","arguments":"{\"cmd\":\"ls\"}"},
                {"type":"function_call_output","call_id":"fc_1","output":"file.txt"}
            ],
            "instructions": "be terse",
            "max_output_tokens": 2048,
            "stream": true,
            "temperature": 0.7
        });
        let once = from_openai_responses(&wire).unwrap();
        let rendered = to_openai_responses(&once).unwrap();
        let twice = from_openai_responses(&rendered).unwrap();
        assert_eq!(once, twice);
    }
}
