//! OpenAI Responses response body → canonical typed pieces (client
//! direction).
//!
//! Inverse of [`crate::dialect::openai_responses::out::responses_body_from_chat`]:
//! parses a non-streaming `/v1/responses` body into an assistant
//! [`ItemStreamMessage`] (push it onto `ItemRequest::messages` to continue
//! the conversation), the canonical cache-inclusive [`Usage`], and an
//! OpenAI-vocabulary `finish_reason`. The Responses surface reports
//! completion as `status` + `incomplete_details.reason`; the mapping is
//! `max_output_tokens` → `length`, `content_filter` → `content_filter`.
//!
//! CoT continuity: `reasoning` output items keep both the plaintext
//! `summary[]` text (joined with '\n') and the `encrypted_content` payload,
//! so re-rendering the message via [`crate::dialect::openai_responses::req_out`]
//! round-trips back onto the `rs_` chain when `store: false`.

use crate::canonical::Usage;
use crate::error::ProxyError;
use crate::items::{ContentItem, ItemMeta, ItemStreamMessage, Role};

/// The parsed pieces of one Responses body.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponsesReply {
    /// The assistant turn, ready to push onto `ItemRequest::messages`.
    pub message: ItemStreamMessage,
    /// Canonical usage.
    pub usage: Usage,
    /// OpenAI-vocabulary stop reason (`stop` | `length` | `content_filter` |
    /// `tool_calls`).
    pub finish_reason: Option<String>,
}

pub fn parse_response(v: &serde_json::Value) -> Result<ResponsesReply, ProxyError> {
    let output = v["output"].as_array().ok_or_else(|| {
        ProxyError::Transport("malformed responses body: output is not an array".into())
    })?;

    let mut items: Vec<ContentItem> = Vec::new();
    for o in output {
        match o["type"].as_str().unwrap_or_default() {
            "message" => {
                for c in o["content"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                    match c["type"].as_str().unwrap_or_default() {
                        "output_text" => {
                            items.push(ContentItem::Text {
                                text: c["text"].as_str().unwrap_or_default().to_string(),
                            });
                        }
                        "refusal" => {
                            items.push(ContentItem::Refusal {
                                text: c["refusal"].as_str().unwrap_or_default().to_string(),
                            });
                        }
                        other => {
                            tracing::debug!(
                                part_type = other,
                                "skipping unknown message content part"
                            );
                        }
                    }
                }
            }
            "function_call" => {
                // arguments arrive as a JSON-encoded string; tolerate
                // upstreams that already sent the object form.
                let arguments = match &o["arguments"] {
                    serde_json::Value::String(s) => {
                        serde_json::from_str(s).unwrap_or(serde_json::Value::Null)
                    }
                    other @ serde_json::Value::Object(_) => other.clone(),
                    _ => serde_json::Value::Null,
                };
                items.push(ContentItem::ToolCall {
                    // `call_id` is what the client echoes back on the next
                    // turn; `id` is the fc_ item id — prefer the call id.
                    id: o["call_id"]
                        .as_str()
                        .or_else(|| o["id"].as_str())
                        .unwrap_or_default()
                        .to_string(),
                    name: o["name"].as_str().unwrap_or_default().to_string(),
                    arguments,
                });
            }
            "reasoning" => {
                let text = o["summary"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|s| s["text"].as_str())
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                items.push(ContentItem::Thinking {
                    text,
                    signature: None,
                    encrypted: o["encrypted_content"].as_str().map(str::to_string),
                    redacted_data: None,
                });
            }
            // file_search_call / web_search_call / computer_call etc. are
            // forward-compat drops, same policy as the request parser.
            other => {
                tracing::debug!(item_type = other, "skipping unknown responses output item");
            }
        }
    }

    let u = &v["usage"];
    let usage = Usage {
        prompt_tokens: u["input_tokens"].as_u64().unwrap_or(0),
        completion_tokens: u["output_tokens"].as_u64().unwrap_or(0),
        cached_read_tokens: u["input_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0),
        cache_write_tokens: 0,
        reasoning_tokens: u["output_tokens_details"]["reasoning_tokens"].as_u64(),
    };

    let has_tool_calls = items
        .iter()
        .any(|i| matches!(i, ContentItem::ToolCall { .. }));
    let finish_reason = match v["status"].as_str() {
        Some("incomplete") | Some("failed") => match v["incomplete_details"]["reason"].as_str() {
            Some("max_output_tokens") | Some("max_tokens") => Some("length".to_string()),
            Some("content_filter") => Some("content_filter".to_string()),
            _ => Some("stop".to_string()),
        },
        _ => Some(if has_tool_calls {
            "tool_calls".to_string()
        } else {
            "stop".to_string()
        }),
    };

    Ok(ResponsesReply {
        message: ItemStreamMessage {
            role: Role::Assistant,
            items,
            metadata: ItemMeta::default(),
        },
        usage,
        finish_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::{ChatResponse, Usage as CanonUsage};

    fn body(output: serde_json::Value, status: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "resp_1", "object": "response", "created_at": 0,
            "status": status, "model": "gpt", "output": output,
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        })
    }

    #[test]
    fn text_message_parses() {
        let r = parse_response(&body(
            serde_json::json!([
                {"type":"message","role":"assistant","status":"completed",
                 "content":[{"type":"output_text","text":"hello"}]}
            ]),
            "completed",
        ))
        .unwrap();
        assert_eq!(r.message.role, Role::Assistant);
        assert!(matches!(
            &r.message.items[0],
            ContentItem::Text { text } if text == "hello"
        ));
        assert_eq!(r.finish_reason.as_deref(), Some("stop"));
        assert_eq!(r.usage.prompt_tokens, 10);
        assert_eq!(r.usage.completion_tokens, 5);
    }

    #[test]
    fn reasoning_keeps_summary_and_encrypted() {
        let r = parse_response(&body(
            serde_json::json!([
                {"type":"reasoning","summary":[{"type":"summary_text","text":"the plan"}]},
                {"type":"reasoning","encrypted_content":"enc_only","summary":[]}
            ]),
            "completed",
        ))
        .unwrap();
        match &r.message.items[0] {
            ContentItem::Thinking {
                text, encrypted, ..
            } => {
                assert_eq!(text, "the plan");
                assert_eq!(encrypted, &None);
            }
            _ => panic!(),
        }
        match &r.message.items[1] {
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
    fn function_call_prefers_call_id_over_item_id() {
        let r = parse_response(&body(
            serde_json::json!([
                {"type":"function_call","id":"fc_itemA","call_id":"call_1",
                 "name":"bash","arguments":"{\"cmd\":\"ls\"}","status":"completed"}
            ]),
            "completed",
        ))
        .unwrap();
        match &r.message.items[0] {
            ContentItem::ToolCall {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "bash");
                assert_eq!(arguments["cmd"], "ls");
            }
            _ => panic!(),
        }
        assert_eq!(r.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn object_arguments_tolerated() {
        let r = parse_response(&body(
            serde_json::json!([
                {"type":"function_call","call_id":"c1","name":"bash","arguments":{"cmd":"ls"}}
            ]),
            "completed",
        ))
        .unwrap();
        assert!(matches!(
            &r.message.items[0],
            ContentItem::ToolCall { arguments, .. } if arguments["cmd"] == "ls"
        ));
    }

    #[test]
    fn incomplete_max_tokens_maps_to_length() {
        let mut v = body(
            serde_json::json!([
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"partial"}]}
            ]),
            "incomplete",
        );
        v["incomplete_details"] = serde_json::json!({"reason": "max_output_tokens"});
        assert_eq!(
            parse_response(&v).unwrap().finish_reason.as_deref(),
            Some("length")
        );

        v["incomplete_details"] = serde_json::json!({"reason": "content_filter"});
        assert_eq!(
            parse_response(&v).unwrap().finish_reason.as_deref(),
            Some("content_filter")
        );
    }

    #[test]
    fn utility_details_parse() {
        let mut v = body(
            serde_json::json!([
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"x"}]}
            ]),
            "completed",
        );
        v["usage"]["output_tokens_details"] = serde_json::json!({"reasoning_tokens": 42});
        v["usage"]["input_tokens_details"] = serde_json::json!({"cached_tokens": 3});
        let r = parse_response(&v).unwrap();
        assert_eq!(r.usage.reasoning_tokens, Some(42));
        assert_eq!(r.usage.cached_read_tokens, 3);
    }

    #[test]
    fn unknown_items_are_skipped() {
        let r = parse_response(&body(
            serde_json::json!([
                {"type":"web_search_call","id":"ws_1","status":"completed"},
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"kept"}]}
            ]),
            "completed",
        ))
        .unwrap();
        assert_eq!(r.message.items.len(), 1);
    }

    #[test]
    fn missing_output_is_an_error() {
        assert!(parse_response(&serde_json::json!({"status":"completed"})).is_err());
    }

    /// Round-trip against the server-side renderer: canonical ChatResponse →
    /// Responses body → parsed reply must agree.
    #[test]
    fn round_trips_through_responses_body_from_chat() {
        let mut chat = ChatResponse::new(
            "gpt",
            "the answer".into(),
            Some("stop".into()),
            CanonUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                cached_read_tokens: 0,
                cache_write_tokens: 0,
                reasoning_tokens: Some(7),
            },
        );
        chat.choices[0].message.reasoning_content = Some("loose reasoning".into());
        let wire = super::super::out::responses_body_from_chat(&chat, "gpt");
        let r = parse_response(&wire).unwrap();
        let types: Vec<&str> = r
            .message
            .items
            .iter()
            .map(|i| match i {
                ContentItem::Thinking { .. } => "thinking",
                ContentItem::Text { .. } => "text",
                ContentItem::ToolCall { .. } => "tool_call",
                ContentItem::ToolResult { .. } => "tool_result",
                ContentItem::Refusal { .. } => "refusal",
            })
            .collect();
        assert_eq!(types, vec!["thinking", "text"]);
        assert!(matches!(
            &r.message.items[0],
            ContentItem::Thinking { text, .. } if text == "loose reasoning"
        ));
        assert_eq!(r.usage.reasoning_tokens, Some(7));
    }
}
