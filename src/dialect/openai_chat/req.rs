//! OpenAI chat/completions request → `ItemRequest`.
//!
//! Translates OpenAI's `{role, content | tool_calls | reasoning_content}`
//! messages into the canonical items model. `reasoning_content` becomes
//! `ContentItem::Thinking` with `signature: None`. Signed/encrypted thinking
//! is never present on this wire.

use crate::error::ProxyError;
use crate::items::{
    ContentItem, ItemMeta, ItemRequest, ItemStreamMessage, ResponseFormat, Role, ThinkingCfg, Tool,
    ToolChoice,
};

pub fn from_openai_chat(v: &serde_json::Value) -> Result<ItemRequest, ProxyError> {
    if v["model"].as_str().is_none_or(|s: &str| s.is_empty()) {
        return Err(ProxyError::BadRequest("field required: model".into()));
    }
    let msgs = v["messages"]
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or_else(|| ProxyError::BadRequest("messages must be a non-empty array".into()))?;

    let mut messages: Vec<ItemStreamMessage> = Vec::new();
    let mut known_tool_ids: std::collections::HashSet<String> = Default::default();

    for m in msgs.iter() {
        let role_str = m["role"].as_str().unwrap_or_default();
        let role = match role_str {
            "system" | "developer" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => {
                return Err(ProxyError::BadRequest(format!(
                    "messages role must be system|user|assistant|tool, got {other:?}"
                )));
            }
        };

        let mut items: Vec<ContentItem> = Vec::new();

        // Unsigned thinking upstream → canonical Thinking item.
        if let Some(rc) = m["reasoning_content"].as_str()
            && !rc.is_empty()
        {
            items.push(ContentItem::Thinking {
                text: rc.to_string(),
                signature: None,
                encrypted: None,
                redacted_data: None,
            });
        }

        match &m["content"] {
            serde_json::Value::String(s) => {
                if !s.is_empty() || role == Role::Assistant {
                    items.push(ContentItem::Text { text: s.clone() });
                }
            }
            serde_json::Value::Array(parts) => {
                for p in parts {
                    match p["type"].as_str().unwrap_or_default() {
                        "text" => {
                            if let Some(t) = p["text"].as_str() {
                                items.push(ContentItem::Text {
                                    text: t.to_string(),
                                });
                            }
                        }
                        // Structured content is not yet supported in this dialect
                        // — guard with an error instead of dropping silently.
                        "refusal" => {
                            if let Some(t) = p["refusal"].as_str() {
                                items.push(ContentItem::Refusal {
                                    text: t.to_string(),
                                });
                            }
                        }
                        "image_url" | "input_audio" => {
                            return Err(ProxyError::BadRequest(
                                "multimodal content is not supported".into(),
                            ));
                        }
                        other => {
                            return Err(ProxyError::BadRequest(format!(
                                "unsupported content part type: {other}"
                            )));
                        }
                    }
                }
            }
            serde_json::Value::Null => {
                // tool-call-only assistant turns carry content:null — and a
                // bare null assistant message is wire-legal too; keep an
                // empty text item so validate() accepts the turn
                if role == Role::Assistant && m["tool_calls"].is_null() {
                    items.push(ContentItem::Text {
                        text: String::new(),
                    });
                }
            }
            _ => {
                return Err(ProxyError::BadRequest(
                    "messages.content must be string | array | null".into(),
                ));
            }
        }

        if let Some(tcs) = m["tool_calls"].as_array() {
            for tc in tcs {
                let id = tc["id"].as_str().filter(|s| !s.is_empty()).ok_or_else(|| {
                    ProxyError::BadRequest("tool_calls requires non-empty id".into())
                })?;
                let name = tc["function"]["name"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        ProxyError::BadRequest("tool_calls[i].function.name is required".into())
                    })?;
                let arguments: serde_json::Value = match &tc["function"]["arguments"] {
                    serde_json::Value::String(s) => {
                        serde_json::from_str(s).unwrap_or(serde_json::Value::Null)
                    }
                    // tolerate providers that send already-parsed objects
                    v if v.is_object() => v.clone(),
                    _ => serde_json::Value::Null,
                };
                known_tool_ids.insert(id.to_string());
                items.push(ContentItem::ToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments,
                });
            }
        }

        if role == Role::Tool {
            let tid = m["tool_call_id"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    ProxyError::BadRequest("tool message requires non-empty tool_call_id".into())
                })?;
            if !known_tool_ids.contains(tid) {
                return Err(ProxyError::BadRequest(format!(
                    "tool message references unknown tool_call_id {tid:?}"
                )));
            }
            let content = m["content"]
                .as_str()
                .map(|s: &str| serde_json::Value::String(s.to_string()))
                .or_else(|| {
                    m["content"].as_array().map(|p| {
                        serde_json::Value::String(
                            p.iter()
                                .filter_map(|part| part["text"].as_str())
                                .collect::<Vec<_>>()
                                .join("\n"),
                        )
                    })
                })
                .ok_or_else(|| {
                    ProxyError::BadRequest("tool message content must be string or blocks".into())
                })?;
            items = vec![ContentItem::ToolResult {
                tool_call_id: tid.to_string(),
                content,
                is_error: false,
            }];
        }

        messages.push(ItemStreamMessage {
            role,
            items,
            metadata: ItemMeta {
                name: m["name"].as_str().map(str::to_string),
                ..ItemMeta::default()
            },
        });
    }

    if messages.is_empty() {
        return Err(ProxyError::BadRequest("no usable messages".into()));
    }

    // OpenAI accepts a single string or an array for `stop`.
    let stop = match &v["stop"] {
        serde_json::Value::String(s) if !s.is_empty() => Some(vec![s.clone()]),
        serde_json::Value::Array(a) => {
            let seqs: Vec<String> = a
                .iter()
                .filter_map(|s| s.as_str().map(str::to_string))
                .collect();
            if seqs.is_empty() { None } else { Some(seqs) }
        }
        _ => None,
    };

    let tools = v
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    if t["type"].as_str() != Some("function") {
                        return None;
                    }
                    Some(Tool {
                        name: t["function"]["name"].as_str()?.to_string(),
                        description: t["function"]["description"].as_str().map(str::to_string),
                        input_schema: if t["function"]["parameters"].is_object() {
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
            Some("function") => {
                o.get("function")
                    .and_then(|f| f["name"].as_str())
                    .map(|n| ToolChoice::Tool {
                        name: n.to_string(),
                    })
            }
            _ => None,
        },
        _ => None,
    });

    let thinking = v
        .get("reasoning_effort")
        .and_then(|e| e.as_str())
        .map(|e| ThinkingCfg {
            budget_tokens: None,
            effort: Some(e.to_string()),
        });

    let response_format = v
        .get("response_format")
        .filter(|f| f.is_object())
        .map(|rf| match rf["type"].as_str() {
            Some("json_schema") => ResponseFormat::JsonSchema {
                name: rf["json_schema"]["name"]
                    .as_str()
                    .unwrap_or("output")
                    .to_string(),
                schema: rf["json_schema"]["schema"].clone(),
                strict: rf["json_schema"]["strict"].as_bool().unwrap_or(false),
            },
            Some("json_object") => ResponseFormat::JsonObject,
            _ => ResponseFormat::Text,
        });

    let mut extra = serde_json::Map::new();
    // chat-completions fields we don't model: forward verbatim so the surface
    // behaves like the legacy passthrough (provider dialects whitelist what
    // they forward, so these can only reach OpenAI-shaped upstreams)
    for key in [
        "stream_options",
        "metadata",
        "user",
        "store",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "seed",
        "n",
        "parallel_tool_calls",
        "service_tier",
    ] {
        if let Some(val) = v.get(key) {
            extra.insert(key.into(), val.clone());
        }
    }

    let req = ItemRequest {
        model: v["model"].as_str().unwrap_or_default().to_string(),
        messages,
        stream: v["stream"].as_bool().unwrap_or(false),
        // no cap when the client set none; max_completion_tokens is the
        // field modern OpenAI SDKs actually send for reasoning models
        max_tokens: v["max_tokens"]
            .as_u64()
            .or_else(|| v["max_completion_tokens"].as_u64())
            .map(|n| n as u32),
        temperature: v["temperature"].as_f64(),
        top_p: v["top_p"].as_f64(),
        stop_sequences: stop,
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
    fn plain_user_text_maps_in() {
        let v = serde_json::json!({
            "model":"m",
            "messages":[{"role":"user","content":"hi"}]
        });
        let req = from_openai_chat(&v).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, Role::User);
        assert!(matches!(
            req.messages[0].items[0],
            ContentItem::Text { ref text } if text == "hi"
        ));
    }

    #[test]
    fn assistant_tool_call_folded_to_item() {
        let v = serde_json::json!({
            "model":"m",
            "messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":null,
                 "tool_calls":[{"id":"tc_1","type":"function",
                    "function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}
            ]
        });
        let req = from_openai_chat(&v).unwrap();
        assert_eq!(req.messages.len(), 2);
        match &req.messages[1].items[0] {
            ContentItem::ToolCall {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "tc_1");
                assert_eq!(name, "bash");
                assert_eq!(arguments["command"], "ls");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn reasoning_content_becomes_thinking_item() {
        let v = serde_json::json!({
            "model":"m",
            "messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","reasoning_content":"thinking","content":"done"}
            ]
        });
        let req = from_openai_chat(&v).unwrap();
        match &req.messages[1].items[0] {
            ContentItem::Thinking {
                text, signature, ..
            } => {
                assert_eq!(text, "thinking");
                assert_eq!(signature, &None);
            }
            _ => panic!(),
        }
        match &req.messages[1].items[1] {
            ContentItem::Text { text } => assert_eq!(text, "done"),
            _ => panic!(),
        }
    }

    #[test]
    fn tool_role_message_requires_known_id() {
        let v = serde_json::json!({
            "model":"m",
            "messages":[
                {"role":"tool","tool_call_id":"x","content":"out"}
            ]
        });
        assert!(from_openai_chat(&v).is_err());
    }

    #[test]
    fn parallel_tool_calls_preserved() {
        let v = serde_json::json!({
            "model":"m",
            "messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":null,
                 "tool_calls":[
                    {"id":"a","type":"function","function":{"name":"one","arguments":"{}"}},
                    {"id":"b","type":"function","function":{"name":"two","arguments":"{}"}}
                ]}
            ]
        });
        let req = from_openai_chat(&v).unwrap();
        let calls: Vec<_> = req.messages[1]
            .items
            .iter()
            .filter(|i| matches!(i, ContentItem::ToolCall { .. }))
            .collect();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn response_format_json_schema() {
        let v = serde_json::json!({
            "model":"m","max_tokens":10,
            "response_format":{"type":"json_schema",
                "json_schema":{"name":"out","schema":{"type":"object"},"strict":true}},
            "messages":[{"role":"user","content":"hi"}]
        });
        let req = from_openai_chat(&v).unwrap();
        match req.response_format.unwrap() {
            ResponseFormat::JsonSchema {
                name,
                schema: _,
                strict,
            } => {
                assert_eq!(name, "out");
                assert!(strict);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn refusal_becomes_refusal_item() {
        let v = serde_json::json!({
            "model":"m","max_tokens":10,
            "messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":[{"type":"refusal","refusal":"nope"}]}
            ]
        });
        let req = from_openai_chat(&v).unwrap();
        assert!(matches!(
            req.messages[1].items[0],
            ContentItem::Refusal { .. }
        ));
    }

    #[test]
    fn stop_sequences_list() {
        let v = serde_json::json!({
            "model":"m","max_tokens":10,
            "stop":["\nWAIT","---"],
            "messages":[{"role":"user","content":"hi"}]
        });
        let req = from_openai_chat(&v).unwrap();
        assert_eq!(
            req.stop_sequences,
            Some(vec!["\nWAIT".into(), "---".into()])
        );
    }

    #[test]
    fn reasoning_effort_maps_to_thinking() {
        let v = serde_json::json!({
            "model":"m","max_tokens":10,
            "reasoning_effort":"high",
            "messages":[{"role":"user","content":"hi"}]
        });
        let req = from_openai_chat(&v).unwrap();
        assert_eq!(req.thinking.as_ref().unwrap().effort, Some("high".into()));
    }

    #[test]
    fn unknown_role_rejected() {
        let v = serde_json::json!({
            "model":"m","max_tokens":10,
            "messages":[{"role":"weird","content":"hi"}]
        });
        assert!(from_openai_chat(&v).is_err());
    }
}
