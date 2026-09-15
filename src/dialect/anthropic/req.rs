//! Anthropic Messages request → `ItemRequest`.
//!
//! Strictly-additive sibling of `anthropic_in::to_canonical`. Same validation,
//! same output shape — but produces the items-based canonical type instead of
//! the OpenAI-shaped `ChatRequest`.

use crate::error::ProxyError;
use crate::items::{
    ContentItem, ItemMeta, ItemRequest, ItemStreamMessage, ResponseFormat, Role, ThinkingCfg, Tool,
    ToolChoice,
};

pub fn from_anthropic(v: &serde_json::Value) -> Result<ItemRequest, ProxyError> {
    if v["model"].as_str().is_none_or(|s: &str| s.is_empty()) {
        return Err(ProxyError::BadRequest("field required: model".into()));
    }
    let msgs_in = v["messages"]
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or_else(|| {
            ProxyError::BadRequest("field required: messages (non-empty array)".into())
        })?;

    // max_tokens is optional at parse time: /v1/messages enforces it (Anthropic
    // contract) via require_max_tokens; count_tokens parses without it.
    let max_tokens: Option<u32> = match v.get("max_tokens") {
        None | Some(&serde_json::Value::Null) => None,
        Some(mt) => {
            let n = mt.as_u64().filter(|n| *n >= 1).ok_or_else(|| {
                ProxyError::BadRequest("max_tokens is required and must be >= 1".into())
            })?;
            Some(
                u32::try_from(n)
                    .map_err(|_| ProxyError::BadRequest("max_tokens out of range".into()))?,
            )
        }
    };
    if let Some(t) = v["temperature"].as_f64()
        && !(0.0..=1.0).contains(&t)
    {
        return Err(ProxyError::BadRequest(format!(
            "temperature must be in [0.0, 1.0], got {t}"
        )));
    }
    if let Some(p) = v["top_p"].as_f64()
        && !(0.0..=1.0).contains(&p)
    {
        return Err(ProxyError::BadRequest(format!(
            "top_p must be in [0.0, 1.0], got {p}"
        )));
    }
    if let Some(th) = v.get("thinking").filter(|x| x.is_object())
        && let Some(b) = th["budget_tokens"].as_u64()
        && (b < 1024 || b > u32::MAX as u64 || max_tokens.is_some_and(|mt| b >= mt as u64))
    {
        return Err(ProxyError::BadRequest(format!(
            "thinking.budget_tokens must be in [1024, max_tokens), got {b} (max_tokens={max_tokens:?})"
        )));
    }

    let mut messages: Vec<ItemStreamMessage> = Vec::new();
    let mut known_tool_ids: std::collections::HashSet<String> = Default::default();
    let mut sys_extra: serde_json::Map<String, serde_json::Value> = Default::default();
    let mut extra = serde_json::Map::new();

    if let Some(sys) = v.get("system") {
        let text = match sys {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter(|b| b["type"] == "text")
                .filter_map(|b| b["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        if !text.is_empty() {
            messages.push(ItemStreamMessage {
                role: Role::System,
                items: vec![ContentItem::Text { text }],
                metadata: ItemMeta::default(),
            });
        }
        if let serde_json::Value::Array(arr) = sys {
            let has_cc = arr.iter().any(|b| b.get("cache_control").is_some());
            if has_cc {
                // same carrier key the legacy path and providers/anthropic use
                sys_extra.insert("_sys_blocks".into(), serde_json::Value::Array(arr.clone()));
            }
        }
    }

    for m in msgs_in.iter() {
        let role_str = m["role"].as_str().unwrap_or_default();

        let role = match role_str {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "system" => {
                // Claude Code interleaves `role: "system"` reminders inside
                // messages; Anthropic's strict spec rejects them. We normalise
                // to the canonical System slot (same behaviour as the merged
                // pre-refactor code path).
                let text = match &m["content"] {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Array(blocks) => blocks
                        .iter()
                        .filter(|b| b["type"] == "text")
                        .filter_map(|b| b["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("\n"),
                    _ => String::new(),
                };
                if !text.is_empty() {
                    messages.push(ItemStreamMessage {
                        role: Role::System,
                        items: vec![ContentItem::Text { text }],
                        metadata: ItemMeta::default(),
                    });
                }
                continue;
            }
            other => {
                return Err(ProxyError::BadRequest(format!(
                    "messages role must be ``user``, ``assistant`` or ``system``, got {other:?}"
                )));
            }
        };

        let mut items: Vec<ContentItem> = Vec::new();
        // per-block cache_control, parallel-indexed with `items`
        let mut block_cc: Vec<Option<serde_json::Value>> = Vec::new();

        let content_val = &m["content"];
        match content_val {
            serde_json::Value::Null => {
                if role == Role::Assistant {
                    // legacy tolerance: tool-call-free assistant turns may
                    // carry null content — keep an empty text item.
                    items.push(ContentItem::Text {
                        text: String::new(),
                    });
                    block_cc.push(None);
                } else {
                    return Err(ProxyError::BadRequest(
                        "messages.content must be a string or blocks array, got null".into(),
                    ));
                }
            }
            serde_json::Value::String(s) => {
                // legacy forwards a bare "" user turn verbatim; keep parity so
                // existing clients don't regress at cutover
                items.push(ContentItem::Text { text: s.clone() });
                block_cc.push(None);
            }
            serde_json::Value::Array(blocks) => {
                if blocks.is_empty() {
                    if role == Role::Assistant {
                        // legacy tolerance: an empty-block assistant turn
                        // becomes an empty content string
                        items.push(ContentItem::Text {
                            text: String::new(),
                        });
                        block_cc.push(None);
                    } else {
                        return Err(ProxyError::BadRequest(
                            "messages content must not be an empty array".into(),
                        ));
                    }
                }
                for b in blocks {
                    match b["type"].as_str().unwrap_or_default() {
                        "text" => {
                            if let Some(t) = b["text"].as_str() {
                                items.push(ContentItem::Text {
                                    text: t.to_string(),
                                });
                                block_cc.push(b.get("cache_control").cloned());
                            }
                        }
                        "thinking" | "redacted_thinking" => {
                            // only ever valid on assistant turns on the wire
                            if role == Role::User {
                                return Err(ProxyError::BadRequest(
                                    "thinking blocks are only valid on assistant messages".into(),
                                ));
                            }
                            if b["type"] == "redacted_thinking" {
                                // keep the opaque `data` payload so the block
                                // can round-trip back to Anthropic
                                items.push(ContentItem::Thinking {
                                    text: String::new(),
                                    signature: None,
                                    encrypted: None,
                                    redacted_data: b["data"].as_str().map(str::to_string),
                                });
                            } else {
                                items.push(ContentItem::Thinking {
                                    text: b["thinking"].as_str().unwrap_or_default().to_string(),
                                    signature: b["signature"].as_str().map(str::to_string),
                                    encrypted: None,
                                    redacted_data: None,
                                });
                            }
                            block_cc.push(b.get("cache_control").cloned());
                        }
                        "tool_use" => {
                            // tool_use is only legal on assistant turns; under
                            // role "user" registering its id would authorise
                            // phantom tool_result linkage
                            if role != Role::Assistant {
                                return Err(ProxyError::BadRequest(
                                    "tool_use blocks are only valid on assistant messages".into(),
                                ));
                            }
                            let id = b["id"].as_str().filter(|s| !s.is_empty());
                            let name = b["name"].as_str().filter(|s| !s.is_empty());
                            match (id, name) {
                                (Some(id), Some(name)) => {
                                    known_tool_ids.insert(id.to_string());
                                    items.push(ContentItem::ToolCall {
                                        id: id.to_string(),
                                        name: name.to_string(),
                                        arguments: b["input"].clone(),
                                    });
                                    block_cc.push(b.get("cache_control").cloned());
                                }
                                _ => {
                                    return Err(ProxyError::BadRequest(
                                        "tool_use block requires non-empty id and name".into(),
                                    ));
                                }
                            }
                        }
                        "tool_result" => {
                            let tid = b["tool_use_id"].as_str().ok_or_else(|| {
                                ProxyError::BadRequest(
                                    "tool_result block missing tool_use_id".into(),
                                )
                            })?;
                            if !known_tool_ids.contains(tid) {
                                return Err(ProxyError::BadRequest(format!(
                                    "tool_result references unknown tool_use_id {tid:?}"
                                )));
                            }
                            let content = match &b["content"] {
                                serde_json::Value::String(s) => {
                                    serde_json::Value::String(s.clone())
                                }
                                serde_json::Value::Array(parts) => {
                                    if parts
                                        .iter()
                                        .any(|p| p["type"] == "image" || p["type"] == "document")
                                    {
                                        return Err(ProxyError::BadRequest(
                                            "image/document content inside tool_result is not supported".into(),
                                        ));
                                    }
                                    let text = parts
                                        .iter()
                                        .filter(|p| p["type"] == "text")
                                        .filter_map(|p| p["text"].as_str())
                                        .collect::<Vec<_>>()
                                        .join("\n");
                                    serde_json::Value::String(text)
                                }
                                _ => serde_json::Value::String(String::new()),
                            };
                            items.push(ContentItem::ToolResult {
                                tool_call_id: tid.to_string(),
                                content,
                                is_error: b["is_error"].as_bool().unwrap_or(false),
                            });
                            block_cc.push(b.get("cache_control").cloned());
                        }
                        "image" | "document" => {
                            return Err(ProxyError::BadRequest(
                                "image/document blocks are not supported".into(),
                            ));
                        }
                        // Unknown/forward-compat block types (server_tool_use,
                        // web_search_tool_result, …) are dropped like the
                        // legacy path — Claude Code conversation histories
                        // carry these and must not 400.
                        other => {
                            tracing::debug!(
                                block_type = other,
                                "dropping unknown anthropic content block"
                            );
                        }
                    }
                }
            }
            _ => {
                return Err(ProxyError::BadRequest(
                    "messages.content must be a string or blocks array".into(),
                ));
            }
        }

        // A message whose blocks were all dropped (unknown types) or empty
        // still becomes a message, matching the legacy path's empty-content
        // forward.
        if items.is_empty() && matches!(role, Role::User | Role::Assistant) {
            items.push(ContentItem::Text {
                text: String::new(),
            });
            block_cc.push(None);
        }
        debug_assert_eq!(block_cc.len(), items.len());
        // all-None carries no information; leave the vec empty so serde skips it
        if block_cc.iter().all(Option::is_none) {
            block_cc.clear();
        }
        messages.push(ItemStreamMessage {
            role,
            items,
            metadata: ItemMeta {
                cache_control: block_cc,
                ..ItemMeta::default()
            },
        });
    }

    if messages.is_empty() {
        return Err(ProxyError::BadRequest(
            "no usable content after filtering".into(),
        ));
    }

    let stop = v["stop_sequences"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(str::to_string))
                .collect()
        })
        .filter(|v: &Vec<String>| !v.is_empty());

    let tools = v
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    let t = t.as_object()?;
                    if t.get("type").is_some_and(|ty| !ty.is_null()) {
                        return None; // hosted tools skipped
                    }
                    Some(Tool {
                        name: t["name"].as_str()?.to_string(),
                        description: t
                            .get("description")
                            .and_then(|d| d.as_str())
                            .map(str::to_string),
                        input_schema: if t["input_schema"].is_object() {
                            t["input_schema"].clone()
                        } else {
                            serde_json::json!({"type": "object", "properties": {}})
                        },
                    })
                })
                .collect::<Vec<Tool>>()
        })
        .unwrap_or_default();

    let tool_choice = v.get("tool_choice").and_then(|tc| {
        match tc["type"].as_str() {
            // "auto" is the upstream default — omit, exactly like the legacy path
            Some("none") => Some(ToolChoice::None),
            Some("any") => Some(ToolChoice::Required),
            Some("tool") => tc["name"].as_str().map(|n| ToolChoice::Tool {
                name: n.to_string(),
            }),
            _ => None,
        }
    });
    // Anthropic's disable_parallel_tool_use ↔ OpenAI parallel_tool_calls=false
    if v["tool_choice"]["disable_parallel_tool_use"].as_bool() == Some(true) {
        extra.insert("parallel_tool_calls".into(), serde_json::json!(false));
    }

    let thinking = v
        .get("thinking")
        .filter(|x| x.is_object())
        .map(|th| ThinkingCfg {
            budget_tokens: th["budget_tokens"]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok()),
            effort: th["effort"].as_str().map(str::to_string),
        });

    let response_format = v
        .get("response_format")
        .filter(|f| f.is_object())
        .map(|rf| match rf["type"].as_str() {
            Some("json_schema") => ResponseFormat::JsonSchema {
                name: rf["name"].as_str().unwrap_or("output").to_string(),
                schema: rf["schema"].clone(),
                strict: rf["strict"].as_bool().unwrap_or(false),
            },
            Some("json_object") | Some("json") => ResponseFormat::JsonObject,
            Some("text") | None => ResponseFormat::Text,
            _ => ResponseFormat::Text,
        });

    for key in [
        "container",
        "mcp_servers",
        "service_tier",
        "cache_control",
        "top_k",
        // thinking verbatim: `{"type":"disabled"}` and exotic budget shapes
        // must reach an Anthropic upstream as sent (typed cfg stays in
        // `req.thinking` for dialects that can express it).
        "thinking",
    ] {
        if let Some(val) = v.get(key).filter(|x| !x.is_null()) {
            extra.insert(key.into(), val.clone());
        }
    }
    // usage-classification metadata (Anthropic bills against it) — forward as
    // OpenAI's loosely-defined user field, same as the legacy path
    if let Some(u) = v["metadata"]["user_id"].as_str() {
        extra.insert("user".into(), serde_json::Value::String(u.to_string()));
    }
    if !sys_extra.is_empty() {
        extra.extend(sys_extra);
    }

    let req = ItemRequest {
        model: v["model"].as_str().unwrap_or_default().to_string(),
        messages,
        stream: v["stream"].as_bool().unwrap_or(false),
        max_tokens,
        temperature: v["temperature"].as_f64(),
        top_p: v["top_p"].as_f64(),
        stop_sequences: stop,
        tools,
        tool_choice,
        thinking,
        response_format,
        extra,
    };
    // prefix lives in the wire envelope (type field), not the message body
    req.validate().map_err(ProxyError::BadRequest)?;
    Ok(req)
}

/// Anthropic's Messages API contract requires max_tokens; the count_tokens
/// surface accepts requests without it. Called by the /v1/messages handler
/// after parsing.
pub fn require_max_tokens(req: &ItemRequest) -> Result<(), ProxyError> {
    if req.max_tokens.is_none() {
        return Err(ProxyError::BadRequest(
            "max_tokens is required and must be >= 1".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &'static str, body: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": role, "content": body}],
        })
    }

    #[test]
    fn plain_text_request_round_trips() {
        let req = from_anthropic(&msg("user", serde_json::json!("hello"))).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, Role::User);
        assert_eq!(req.messages[0].items.len(), 1);
        assert!(matches!(
            req.messages[0].items[0],
            ContentItem::Text { ref text } if text == "hello"
        ));
    }

    #[test]
    fn system_field_becomes_canonical_system_role() {
        let v = serde_json::json!({
            "model": "m", "max_tokens": 10,
            "system": "be brief",
            "messages": [{"role":"user","content":"hi"}],
        });
        let req = from_anthropic(&v).unwrap();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, Role::System);
        assert_eq!(req.messages[0].items[0].kind(), "text");
    }

    #[test]
    fn inline_system_role_folds_into_system_slot() {
        let v = serde_json::json!({
            "model": "m", "max_tokens": 10,
            "messages": [
                {"role":"user","content":"first"},
                {"role":"system","content":"REMINDER"},
                {"role":"assistant","content":"ok"}
            ],
        });
        let req = from_anthropic(&v).unwrap();
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[1].role, Role::System);
        assert!(matches!(
            req.messages[1].items[0],
            ContentItem::Text { ref text } if text == "REMINDER"
        ));
    }

    #[test]
    fn tool_use_block_becomes_tool_call_item() {
        let v = serde_json::json!({
            "model": "m", "max_tokens": 10,
            "messages": [
                {"role":"user","content":"run it"},
                {"role":"assistant","content":[
                    {"type":"tool_use","id":"tc_1","name":"bash","input":{"command":"ls"}}
                ]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"tc_1","content":"file.txt"}
                ]},
            ],
        });
        let req = from_anthropic(&v).unwrap();
        assert_eq!(req.messages.len(), 3);
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
        match &req.messages[2].items[0] {
            ContentItem::ToolResult {
                tool_call_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_call_id, "tc_1");
                assert_eq!(content, "file.txt");
                assert!(!is_error);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn orphan_tool_result_rejected() {
        let v = serde_json::json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role":"user","content":[
                {"type":"tool_result","tool_use_id":"who_knows","content":"c"}
            ]}],
        });
        assert!(from_anthropic(&v).is_err());
    }

    #[test]
    fn thinking_blocks_become_thinking_items() {
        let v = serde_json::json!({
            "model": "m", "max_tokens": 10000,
            "thinking": {"type":"enabled","budget_tokens":4096},
            "messages": [{"role":"assistant","content":[
                {"type":"thinking","thinking":"plan","signature":"sig_ABC"},
                {"type":"text","text":"answer"}
            ]}],
        });
        let req = from_anthropic(&v).unwrap();
        assert_eq!(req.messages.len(), 1);
        match &req.messages[0].items[0] {
            ContentItem::Thinking {
                text, signature, ..
            } => {
                assert_eq!(text, "plan");
                assert_eq!(signature.as_deref(), Some("sig_ABC"));
            }
            _ => panic!(),
        }
        assert!(matches!(
            req.messages[0].items[1],
            ContentItem::Text { ref text } if text == "answer"
        ));
        assert_eq!(req.thinking.as_ref().unwrap().budget_tokens, Some(4096));
    }

    #[test]
    fn response_format_normalizes_to_json_schema() {
        let v = serde_json::json!({
            "model":"m", "max_tokens":10,
            "response_format":{"type":"json_schema","name":"reply","schema":{"type":"object"},"strict":true},
            "messages":[{"role":"user","content":"hi"}],
        });
        let req = from_anthropic(&v).unwrap();
        match req.response_format.unwrap() {
            ResponseFormat::JsonSchema {
                name,
                schema: _,
                strict,
            } => {
                assert_eq!(name, "reply");
                assert!(strict);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn json_object_response_format() {
        let v = serde_json::json!({
            "model":"m", "max_tokens":10,
            "response_format":{"type":"json_object"},
            "messages":[{"role":"user","content":"hi"}],
        });
        let req = from_anthropic(&v).unwrap();
        assert_eq!(req.response_format, Some(ResponseFormat::JsonObject));
    }

    #[test]
    fn missing_max_tokens_parses_but_fails_surface_check() {
        let v = serde_json::json!({
            "model": "m",
            "messages": [{"role":"user","content":"hi"}],
        });
        let req = from_anthropic(&v).unwrap();
        assert_eq!(req.max_tokens, None);
        assert!(require_max_tokens(&req).is_err());
    }

    #[test]
    fn max_tokens_overflow_rejected() {
        let v = serde_json::json!({
            "model": "m",
            "max_tokens": 4294967297_u64,
            "messages": [{"role":"user","content":"hi"}],
        });
        assert!(from_anthropic(&v).is_err());
    }

    #[test]
    fn budget_tokens_overflow_rejected() {
        let v = serde_json::json!({
            "model": "m",
            "max_tokens": null,
            "thinking": {"type":"enabled","budget_tokens": 4294967300_u64},
            "messages": [{"role":"user","content":"hi"}],
        });
        assert!(from_anthropic(&v).is_err());
    }

    #[test]
    fn empty_string_user_content_accepted_like_legacy() {
        let req = from_anthropic(&msg("user", serde_json::json!(""))).unwrap();
        assert!(matches!(
            req.messages[0].items[0],
            ContentItem::Text { ref text } if text.is_empty()
        ));
    }

    #[test]
    fn empty_array_assistant_content_accepted_like_legacy() {
        let req = from_anthropic(&msg("assistant", serde_json::json!([]))).unwrap();
        assert!(matches!(
            req.messages[0].items[0],
            ContentItem::Text { ref text } if text.is_empty()
        ));
    }

    #[test]
    fn unknown_blocks_dropped_like_legacy() {
        let v = serde_json::json!({
            "model": "m", "max_tokens": 10,
            "messages": [
                {"role":"user","content":[
                    {"type":"server_tool_use","id":"srv_1","name":"web_search","input":{}},
                    {"type":"web_search_tool_result","tool_use_id":"srv_1","content":[]},
                    {"type":"text","text":"kept"}
                ]},
                {"role":"assistant","content":[
                    {"type":"redacted_thinking","data":"blob"}
                ]},
                {"role":"user","content":[
                    {"type":"web_search_tool_result","tool_use_id":"srv_1","content":[]}
                ]},
            ],
        });
        let req = from_anthropic(&v).unwrap();
        assert_eq!(req.messages[0].items.len(), 1);
        assert!(matches!(
            req.messages[0].items[0],
            ContentItem::Text { ref text } if text == "kept"
        ));
        // unknown-only message survives as an empty text turn
        assert!(matches!(
            req.messages[2].items[0],
            ContentItem::Text { ref text } if text.is_empty()
        ));
    }

    #[test]
    fn tool_use_on_user_role_rejected() {
        let v = serde_json::json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role":"user","content":[
                {"type":"tool_use","id":"tc_1","name":"bash","input":{}}
            ]}],
        });
        assert!(from_anthropic(&v).is_err());
    }

    #[test]
    fn temperature_out_of_range_errors() {
        let v = serde_json::json!({
            "model":"m","max_tokens":10,"temperature": 1.5,
            "messages":[{"role":"user","content":"hi"}],
        });
        assert!(from_anthropic(&v).is_err());
    }

    #[test]
    fn budget_tokens_below_min_errors() {
        let v = serde_json::json!({
            "model":"m","max_tokens":10000,
            "thinking":{"type":"enabled","budget_tokens":512},
            "messages":[{"role":"user","content":"hi"}],
        });
        assert!(from_anthropic(&v).is_err());
    }

    #[test]
    fn empty_content_array_errors() {
        let v = serde_json::json!({
            "model":"m","max_tokens":10,
            "messages":[{"role":"user","content":[]}],
        });
        assert!(from_anthropic(&v).is_err());
    }

    #[test]
    fn unknown_role_errors() {
        let v = serde_json::json!({
            "model":"m","max_tokens":10,
            "messages":[{"role":"developer","content":"x"}],
        });
        assert!(from_anthropic(&v).is_err());
    }

    #[test]
    fn null_content_errors() {
        let v = serde_json::json!({
            "model":"m","max_tokens":10,
            "messages":[{"role":"user","content":null}],
        });
        assert!(from_anthropic(&v).is_err());
    }

    #[test]
    fn extra_fields_preserved() {
        let v = serde_json::json!({
            "model":"m","max_tokens":10,
            "service_tier":"standard",
            "container":{"id":"c"},
            "messages":[{"role":"user","content":"hi"}],
        });
        let req = from_anthropic(&v).unwrap();
        assert_eq!(req.extra["service_tier"], "standard");
        assert!(req.extra.contains_key("container"));
    }
}
