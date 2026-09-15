//! OpenAI Responses request → canonical `ItemRequest`.
//!
//! `input` accepts either a bare string (one user message) or an array of
//! typed items: `message` (role-based), `function_call`, `function_call_output`,
//! `reasoning`, `item_reference`. We normalise all of them into
//! `ItemStreamMessage`s, preserving item order.

use crate::error::ProxyError;
use crate::items::{
    ContentItem, ItemMeta, ItemRequest, ItemStreamMessage, ResponseFormat, Role, ThinkingCfg, Tool,
    ToolChoice,
};

pub fn from_openai_responses(v: &serde_json::Value) -> Result<ItemRequest, ProxyError> {
    if v["model"].as_str().is_none_or(|s: &str| s.is_empty()) {
        return Err(ProxyError::BadRequest("field required: model".into()));
    }
    let mut messages: Vec<ItemStreamMessage> = Vec::new();
    let mut known_tool_ids: std::collections::HashSet<String> = Default::default();

    // Top-level `instructions` folds into a system message.
    if let Some(s) = v["instructions"].as_str()
        && !s.is_empty()
    {
        messages.push(ItemStreamMessage {
            role: Role::System,
            items: vec![ContentItem::Text {
                text: s.to_string(),
            }],
            metadata: ItemMeta::default(),
        });
    }

    match &v["input"] {
        serde_json::Value::String(s) => {
            messages.push(ItemStreamMessage {
                role: Role::User,
                items: vec![ContentItem::Text { text: s.clone() }],
                metadata: ItemMeta::default(),
            });
        }
        serde_json::Value::Array(items) => {
            for it in items {
                let ty = it["type"].as_str().unwrap_or("message");
                match ty {
                    "message" => {
                        let role = match it["role"].as_str().unwrap_or_default() {
                            "user" => Role::User,
                            "assistant" => Role::Assistant,
                            "system" | "developer" => Role::System,
                            other => {
                                return Err(ProxyError::BadRequest(format!(
                                    "unknown message role {other:?}"
                                )));
                            }
                        };
                        let mut items_out: Vec<ContentItem> = Vec::new();
                        match &it["content"] {
                            serde_json::Value::String(s) => {
                                items_out.push(ContentItem::Text { text: s.clone() });
                            }
                            serde_json::Value::Array(parts) => {
                                for p in parts {
                                    let pty = p["type"].as_str().unwrap_or_default();
                                    match pty {
                                        "input_text" | "output_text" => {
                                            if let Some(t) = p["text"].as_str() {
                                                items_out.push(ContentItem::Text {
                                                    text: t.to_string(),
                                                });
                                            }
                                        }
                                        "input_image" | "input_audio" | "input_file" => {
                                            return Err(ProxyError::BadRequest(
                                                "multimodal content is not supported".into(),
                                            ));
                                        }
                                        "refusal" => {
                                            if let Some(t) = p["refusal"].as_str() {
                                                items_out.push(ContentItem::Refusal {
                                                    text: t.to_string(),
                                                });
                                            }
                                        }
                                        other => {
                                            return Err(ProxyError::BadRequest(format!(
                                                "unsupported content part type {other:?}"
                                            )));
                                        }
                                    }
                                }
                            }
                            serde_json::Value::Null => {}
                            _ => {
                                return Err(ProxyError::BadRequest(
                                    "message content must be string or parts array".into(),
                                ));
                            }
                        }
                        if items_out.is_empty() && role != Role::System {
                            // drop empty entries; Anthropic's spec says these
                            // are client errors — keep the laxer Responses
                            // contract here.
                            continue;
                        }
                        messages.push(ItemStreamMessage {
                            role,
                            items: items_out,
                            metadata: ItemMeta::default(),
                        });
                    }
                    "function_call" => {
                        let name =
                            it["name"]
                                .as_str()
                                .filter(|s| !s.is_empty())
                                .ok_or_else(|| {
                                    ProxyError::BadRequest("function_call requires name".into())
                                })?;
                        let call_id = it["call_id"]
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| {
                                ProxyError::BadRequest("function_call requires call_id".into())
                            })?;
                        let arguments = match &it["arguments"] {
                            // spec: arguments arrive as a JSON-encoded string;
                            // tolerate clients that already sent an object
                            serde_json::Value::String(s) => {
                                serde_json::from_str(s).unwrap_or(serde_json::Value::Null)
                            }
                            other @ serde_json::Value::Object(_) => other.clone(),
                            _ => serde_json::Value::Null,
                        };
                        known_tool_ids.insert(call_id.to_string());
                        messages.push(ItemStreamMessage {
                            role: Role::Assistant,
                            items: vec![ContentItem::ToolCall {
                                id: call_id.to_string(),
                                name: name.to_string(),
                                arguments,
                            }],
                            metadata: ItemMeta::default(),
                        });
                    }
                    "function_call_output" => {
                        let cid = it["call_id"]
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| {
                                ProxyError::BadRequest(
                                    "function_call_output requires call_id".into(),
                                )
                            })?;
                        if !known_tool_ids.contains(cid) {
                            return Err(ProxyError::BadRequest(format!(
                                "function_call_output references unknown call_id {cid:?}"
                            )));
                        }
                        let content = match &it["output"] {
                            serde_json::Value::String(s) => serde_json::Value::String(s.clone()),
                            serde_json::Value::Array(parts) => {
                                // spec's structured form — refuse multimodal
                                // parts instead of flattening them away
                                if parts.iter().any(|p| p["type"] != "input_text") {
                                    return Err(ProxyError::BadRequest(
                                        "function_call_output.output supports only input_text parts".into(),
                                    ));
                                }
                                serde_json::Value::String(
                                    parts
                                        .iter()
                                        .filter_map(|p| p["text"].as_str())
                                        .collect::<Vec<_>>()
                                        .join("\n"),
                                )
                            }
                            _ => {
                                return Err(ProxyError::BadRequest(
                                    "function_call_output.output must be a string or input_text parts".into(),
                                ));
                            }
                        };
                        messages.push(ItemStreamMessage {
                            role: Role::Tool,
                            items: vec![ContentItem::ToolResult {
                                tool_call_id: cid.to_string(),
                                content,
                                is_error: false,
                            }],
                            metadata: ItemMeta::default(),
                        });
                    }
                    "reasoning" => {
                        // OpenAI Responses kind: encrypted thinking in
                        // `encrypted_content`, plaintext summary items in
                        // `summary[]`. Multi-turn CoT continuity (store=false
                        // replays) hinges on the encrypted payload, so an item
                        // carrying it is kept even without summaries — clients
                        // only get summaries if they asked the upstream for
                        // them.
                        let text = it["summary"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .filter_map(|s| s["text"].as_str())
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            })
                            .unwrap_or_default();
                        let encrypted = it["encrypted_content"].as_str();
                        if encrypted.is_some() || !text.is_empty() {
                            messages.push(ItemStreamMessage {
                                role: Role::Assistant,
                                items: vec![ContentItem::Thinking {
                                    text,
                                    signature: None,
                                    encrypted: encrypted.map(str::to_string),
                                    redacted_data: None,
                                }],
                                metadata: ItemMeta::default(),
                            });
                        }
                    }
                    "item_reference" => {
                        // Server-side continuation requires persistent state
                        // doesn't have — refuse explicitly.
                        return Err(ProxyError::BadRequest(
                            "item_reference requires server-side state; not supported".into(),
                        ));
                    }
                    other => {
                        return Err(ProxyError::BadRequest(format!(
                            "unsupported input item type {other:?}"
                        )));
                    }
                }
            }
        }
        serde_json::Value::Null => {
            return Err(ProxyError::BadRequest("input must not be null".into()));
        }
        _ => {
            return Err(ProxyError::BadRequest(
                "input must be a string or array of items".into(),
            ));
        }
    }

    if messages.is_empty() {
        return Err(ProxyError::BadRequest("no usable content".into()));
    }

    let tools = v
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    if t["type"].as_str() != Some("function") {
                        // Web-search, file-search, hosted tools — same policy as
                        // Anthropic: silently drop with a comment rather than
                        // emit malformed names.
                        return None;
                    }
                    Some(Tool {
                        // Responses has flat fields; fall back to function.<…>
                        // for openAI-chat-rolled-into-Responses payloads.
                        name: t["name"]
                            .as_str()
                            .or_else(|| t["function"]["name"].as_str())?
                            .to_string(),
                        description: t["description"]
                            .as_str()
                            .or_else(|| t["function"]["description"].as_str())
                            .map(str::to_string),
                        input_schema: if t["parameters"].is_object() {
                            t["parameters"].clone()
                        } else if t["function"]["parameters"].is_object() {
                            t["function"]["parameters"].clone()
                        } else {
                            serde_json::json!({"type": "object", "properties": {}})
                        },
                    })
                })
                .collect::<Vec<Tool>>()
        })
        .unwrap_or_default();

    let tool_choice = v.get("tool_choice").and_then(|tc| match tc {
        serde_json::Value::String(s) => match s.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "none" => Some(ToolChoice::None),
            "required" => Some(ToolChoice::Required),
            _ => None,
        },
        serde_json::Value::Object(o) => match o.get("type").and_then(|t| t.as_str()) {
            Some("function") => o
                .get("name")
                .and_then(|n| n.as_str())
                .or_else(|| o.get("function").and_then(|f| f["name"].as_str()))
                .map(|n| ToolChoice::Tool {
                    name: n.to_string(),
                }),
            _ => None,
        },
        _ => None,
    });

    let thinking = v
        .get("reasoning")
        .filter(|r| r.is_object())
        .map(|r| ThinkingCfg {
            // clamp instead of truncating: a pathological u64 budget must not
            // wrap into a small u32
            budget_tokens: r["budget_tokens"]
                .as_u64()
                .map(|n| u32::try_from(n).unwrap_or(u32::MAX)),
            effort: r["effort"].as_str().map(str::to_string),
        });

    let response_format = v
        .get("text")
        .and_then(|t| t.get("format"))
        .filter(|f| f.is_object())
        .map(|f| match f["type"].as_str() {
            Some("json_schema") => ResponseFormat::JsonSchema {
                name: f["name"].as_str().unwrap_or("output").to_string(),
                schema: f["schema"].clone(),
                strict: f["strict"].as_bool().unwrap_or(false),
            },
            Some("json_object") => ResponseFormat::JsonObject,
            _ => ResponseFormat::Text,
        });

    let mut extra = serde_json::Map::new();
    for key in [
        "store",
        "background",
        "metadata",
        "service_tier",
        "truncation",
        "parallel_tool_calls",
        "user",
        "stream_options",
        "previous_response_id",
        "conversation",
        "include",
    ] {
        if let Some(val) = v.get(key) {
            extra.insert(key.into(), val.clone());
        }
    }

    let req = ItemRequest {
        model: v["model"].as_str().unwrap_or_default().to_string(),
        messages,
        stream: v["stream"].as_bool().unwrap_or(false),
        max_tokens: v["max_output_tokens"]
            .as_u64()
            .or_else(|| v["max_tokens"].as_u64())
            .map(|n| n as u32),
        temperature: v["temperature"].as_f64(),
        top_p: v["top_p"].as_f64(),
        // Responses takes stop as a string array; accept a bare string the way
        // openai_chat does (common client shape) rather than silently dropping it
        stop_sequences: match &v["stop"] {
            serde_json::Value::String(s) if !s.is_empty() => Some(vec![s.clone()]),
            serde_json::Value::Array(a) => {
                let v: Vec<String> = a
                    .iter()
                    .filter_map(|s| s.as_str().map(str::to_string))
                    .collect();
                (!v.is_empty()).then_some(v)
            }
            _ => None,
        },
        tools,
        tool_choice,
        thinking,
        response_format,
        extra,
    };
    req.validate().map_err(ProxyError::BadRequest)?;
    Ok(req)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_string_input_maps() {
        let v = serde_json::json!({
            "model":"m", "input": "hello"
        });
        let req = from_openai_responses(&v).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, Role::User);
        assert!(matches!(
            req.messages[0].items[0],
            ContentItem::Text { ref text } if text == "hello"
        ));
    }

    #[test]
    fn instructions_folded_to_system() {
        let v = serde_json::json!({
            "model":"m","instructions":"be terse",
            "input":[{"type":"message","role":"user","content":"hi"}]
        });
        let req = from_openai_responses(&v).unwrap();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, Role::System);
        assert!(matches!(
            req.messages[0].items[0],
            ContentItem::Text { ref text } if text == "be terse"
        ));
    }

    #[test]
    fn function_call_with_object_arguments_preserved() {
        let v = serde_json::json!({
            "model":"m",
            "input":[
                {"type":"function_call","call_id":"fc_2","name":"bash","arguments":{"cmd":"ls"}}
            ]
        });
        let req = from_openai_responses(&v).unwrap();
        match &req.messages[0].items[0] {
            ContentItem::ToolCall { arguments, .. } => {
                assert_eq!(arguments["cmd"], "ls");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn function_call_items_become_tool_call() {
        let v = serde_json::json!({
            "model":"m",
            "input":[
                {"type":"function_call","call_id":"fc_1","name":"bash","arguments":"{\"cmd\":\"ls\"}"}
            ]
        });
        let req = from_openai_responses(&v).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, Role::Assistant);
        match &req.messages[0].items[0] {
            ContentItem::ToolCall {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "fc_1");
                assert_eq!(name, "bash");
                assert_eq!(arguments["cmd"], "ls");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn function_call_output_pairs_by_call_id() {
        let v = serde_json::json!({
            "model":"m",
            "input":[
                {"type":"function_call","call_id":"fc_1","name":"bash","arguments":"{}"},
                {"type":"function_call_output","call_id":"fc_1","output":"file.txt"}
            ]
        });
        let req = from_openai_responses(&v).unwrap();
        match &req.messages[1].items[0] {
            ContentItem::ToolResult {
                tool_call_id,
                content,
                ..
            } => {
                assert_eq!(tool_call_id, "fc_1");
                assert_eq!(content, &serde_json::json!("file.txt"));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn orphan_function_call_output_rejected() {
        let v = serde_json::json!({
            "model":"m",
            "input":[{"type":"function_call_output","call_id":"?","output":"x"}]
        });
        assert!(from_openai_responses(&v).is_err());
    }

    #[test]
    fn reasoning_summary_becomes_thinking() {
        let v = serde_json::json!({
            "model":"m",
            "input":[
                {"type":"reasoning","encrypted_content":"enc",
                 "summary":[{"type":"summary_text","text":"the plan"}]}
            ]
        });
        let req = from_openai_responses(&v).unwrap();
        match &req.messages[0].items[0] {
            ContentItem::Thinking {
                text,
                signature,
                encrypted,
                ..
            } => {
                assert_eq!(text, "the plan");
                assert_eq!(signature, &None);
                assert_eq!(encrypted.as_deref(), Some("enc"));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn reasoning_with_encrypted_only_is_kept() {
        // no summary[] — the encrypted payload alone must survive so the next
        // turn can resume the chain-of-thought
        let v = serde_json::json!({
            "model":"m",
            "input":[
                {"type":"message","role":"user","content":"q"},
                {"type":"reasoning","encrypted_content":"enc_only"},
                {"type":"function_call","call_id":"fc_1","name":"bash","arguments":"{}"}
            ]
        });
        let req = from_openai_responses(&v).unwrap();
        assert_eq!(req.messages.len(), 3);
        match &req.messages[1].items[0] {
            ContentItem::Thinking {
                text, encrypted, ..
            } => {
                assert_eq!(text, "");
                assert_eq!(encrypted.as_deref(), Some("enc_only"));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn json_schema_via_text_format() {
        let v = serde_json::json!({
            "model":"m","input":"hi",
            "text":{"format":{"type":"json_schema","name":"answer","schema":{"type":"object"},"strict":true}}
        });
        let req = from_openai_responses(&v).unwrap();
        match req.response_format.unwrap() {
            ResponseFormat::JsonSchema { name, strict, .. } => {
                assert_eq!(name, "answer");
                assert!(strict);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn reasoning_config_maps_to_thinking() {
        let v = serde_json::json!({
            "model":"m","input":"hi",
            "reasoning":{"effort":"high"}
        });
        let req = from_openai_responses(&v).unwrap();
        assert_eq!(req.thinking.unwrap().effort, Some("high".into()));
    }

    #[test]
    fn max_output_tokens_fallback() {
        let v = serde_json::json!({
            "model":"m","input":"hi","max_output_tokens":2048
        });
        let req = from_openai_responses(&v).unwrap();
        assert_eq!(req.max_tokens, Some(2048));
    }

    #[test]
    fn absent_max_output_tokens_stays_uncapped() {
        let v = serde_json::json!({"model":"m","input":"hi"});
        let req = from_openai_responses(&v).unwrap();
        assert_eq!(req.max_tokens, None);
    }

    #[test]
    fn array_output_parts_joined_to_text() {
        let v = serde_json::json!({
            "model":"m",
            "input":[
                {"type":"function_call","call_id":"fc_1","name":"bash","arguments":"{}"},
                {"type":"function_call_output","call_id":"fc_1",
                 "output":[{"type":"input_text","text":"line1"},{"type":"input_text","text":"line2"}]}
            ]
        });
        let req = from_openai_responses(&v).unwrap();
        match &req.messages[1].items[0] {
            ContentItem::ToolResult { content, .. } => {
                assert_eq!(content, &serde_json::json!("line1\nline2"));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn structured_output_with_image_part_rejected() {
        let v = serde_json::json!({
            "model":"m",
            "input":[
                {"type":"function_call","call_id":"fc_1","name":"bash","arguments":"{}"},
                {"type":"function_call_output","call_id":"fc_1",
                 "output":[{"type":"input_image","image_url":"x"}]}
            ]
        });
        assert!(from_openai_responses(&v).is_err());
    }

    #[test]
    fn tool_choice_specific() {
        let v = serde_json::json!({
            "model":"m","input":"hi",
            "tool_choice":{"type":"function","name":"bash"},
        });
        let req = from_openai_responses(&v).unwrap();
        assert_eq!(
            req.tool_choice,
            Some(ToolChoice::Tool {
                name: "bash".into()
            })
        );
    }

    #[test]
    fn extras_persisted() {
        let v = serde_json::json!({
            "model":"m","input":"hi",
            "metadata":{"session":"abc"},
            "previous_response_id":"resp_1"
        });
        let req = from_openai_responses(&v).unwrap();
        assert!(req.extra.contains_key("metadata"));
        assert!(req.extra.contains_key("previous_response_id"));
    }

    #[test]
    fn item_reference_unsupported() {
        let v = serde_json::json!({
            "model":"m",
            "input":[{"type":"item_reference","id":"it_1"}]
        });
        assert!(from_openai_responses(&v).is_err());
    }
}
