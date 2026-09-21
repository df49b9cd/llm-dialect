//! Anthropic Messages response body → canonical typed pieces (client
//! direction).
//!
//! Inverse of [`crate::dialect::anthropic::out::response_from_canonical`]:
//! parses a non-streaming `/v1/messages` body into an assistant
//! [`ItemStreamMessage`] (so the turn can be appended directly to an
//! ongoing [`ItemRequest`](crate::items::ItemRequest) conversation), the canonical cache-INCLUSIVE
//! [`Usage`], and the stop reason in OpenAI's `finish_reason` vocabulary
//! (`stop`, `length`, `tool_calls`) — the stop-reason table is the exact
//! inverse of the stream-side outbound mapping.
//!
//! Usage arithmetic: Anthropic's wire reports `input_tokens`
//! cache-EXCLUSIVE; canonical is cache-inclusive, so the cache classes are
//! added in here — mirroring the subtraction in the server-side renderer.

use crate::canonical::Usage;
use crate::error::ProxyError;
use crate::items::{ContentItem, ItemMeta, ItemStreamMessage, Role};

/// The parsed pieces of one Anthropic message response.
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicReply {
    /// The assistant turn, ready to push onto `ItemRequest::messages`.
    pub message: ItemStreamMessage,
    /// Cache-inclusive canonical usage.
    pub usage: Usage,
    /// OpenAI-vocabulary stop reason (`stop` | `length` | `tool_calls`).
    pub finish_reason: Option<String>,
    /// The matched stop sequence text when `stop_reason == "stop_sequence"`.
    pub stop_sequence: Option<String>,
}

pub fn parse_response(v: &serde_json::Value) -> Result<AnthropicReply, ProxyError> {
    let blocks = v["content"].as_array().ok_or_else(|| {
        ProxyError::Transport("malformed anthropic response: content is not an array".into())
    })?;

    let mut items: Vec<ContentItem> = Vec::new();
    for b in blocks {
        match b["type"].as_str().unwrap_or_default() {
            "text" => {
                items.push(ContentItem::Text {
                    text: b["text"].as_str().unwrap_or_default().to_string(),
                });
            }
            "thinking" => {
                items.push(ContentItem::Thinking {
                    text: b["thinking"].as_str().unwrap_or_default().to_string(),
                    signature: b["signature"].as_str().map(str::to_string),
                    encrypted: None,
                    redacted_data: None,
                });
            }
            "redacted_thinking" => {
                items.push(ContentItem::Thinking {
                    text: String::new(),
                    signature: None,
                    encrypted: None,
                    redacted_data: b["data"].as_str().map(str::to_string),
                });
            }
            "tool_use" => {
                items.push(ContentItem::ToolCall {
                    id: b["id"].as_str().unwrap_or_default().to_string(),
                    name: b["name"].as_str().unwrap_or_default().to_string(),
                    arguments: if b["input"].is_null() {
                        serde_json::json!({})
                    } else {
                        b["input"].clone()
                    },
                });
            }
            // Forward-compat: server-tool / citation / etc. blocks a typed
            // client can't express yet are skipped, same as the request-side
            // parser drops unknown blocks.
            other => {
                tracing::debug!(
                    block_type = other,
                    "skipping unknown anthropic response block"
                );
            }
        }
    }

    let u = &v["usage"];
    let fresh = u["input_tokens"].as_u64().unwrap_or(0);
    let cache_write = u["cache_creation_input_tokens"].as_u64().unwrap_or(0);
    let cached_read = u["cache_read_input_tokens"].as_u64().unwrap_or(0);
    let usage = Usage {
        prompt_tokens: fresh + cached_read + cache_write,
        completion_tokens: u["output_tokens"].as_u64().unwrap_or(0),
        cached_read_tokens: cached_read,
        cache_write_tokens: cache_write,
        reasoning_tokens: None,
    };

    let finish_reason = v["stop_reason"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| map_stop_reason_inbound(s).to_string());

    Ok(AnthropicReply {
        message: ItemStreamMessage {
            role: Role::Assistant,
            items,
            metadata: ItemMeta::default(),
        },
        usage,
        finish_reason,
        stop_sequence: v["stop_sequence"].as_str().map(str::to_string),
    })
}

/// Anthropic `stop_reason` → OpenAI `finish_reason` vocabulary. Exact
/// inverse of [`super::stream::map_stop_reason_outbound`]: unknown values
/// degrade to `"stop"`, mirroring that function's wildcard arm.
fn map_stop_reason_inbound(reason: &str) -> &'static str {
    match reason {
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        // end_turn | stop_sequence | pause_turn | refusal | unknown
        _ => "stop",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(content: serde_json::Value, stop_reason: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude", "content": content,
            "stop_reason": stop_reason, "stop_sequence": null,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
    }

    #[test]
    fn text_response_parses_to_assistant_turn() {
        let r = parse_response(&body(
            serde_json::json!([{"type":"text","text":"hello"}]),
            "end_turn",
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
    fn thinking_and_tool_use_parse() {
        let r = parse_response(&body(
            serde_json::json!([
                {"type":"thinking","thinking":"plan","signature":"sig_ABC"},
                {"type":"redacted_thinking","data":"blob"},
                {"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"ls"}}
            ]),
            "tool_use",
        ))
        .unwrap();
        match &r.message.items[0] {
            ContentItem::Thinking {
                text, signature, ..
            } => {
                assert_eq!(text, "plan");
                assert_eq!(signature.as_deref(), Some("sig_ABC"));
            }
            _ => panic!(),
        }
        assert!(matches!(
            &r.message.items[1],
            ContentItem::Thinking { redacted_data: Some(d), .. } if d == "blob"
        ));
        assert!(matches!(
            &r.message.items[2],
            ContentItem::ToolCall { id, .. } if id == "toolu_1"
        ));
        assert_eq!(r.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn cache_tokens_fold_into_prompt_tokens_inclusively() {
        let mut v = body(serde_json::json!([{"type":"text","text":"x"}]), "end_turn");
        v["usage"] = serde_json::json!({
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 2,
        });
        let r = parse_response(&v).unwrap();
        assert_eq!(r.usage.prompt_tokens, 15);
        assert_eq!(r.usage.cache_write_tokens, 3);
        assert_eq!(r.usage.cached_read_tokens, 2);
        assert_eq!(
            r.usage.prompt_tokens - r.usage.cached_read_tokens - r.usage.cache_write_tokens,
            10,
            "fresh = prompt - cached_read - cache_write holds"
        );
    }

    #[test]
    fn stop_reasons_map_and_sequence_surfaces() {
        let v = serde_json::json!({
            "id":"m","type":"message","role":"assistant","model":"c",
            "content":[{"type":"text","text":"x"}],
            "stop_reason":"max_tokens","stop_sequence":null,
            "usage":{"input_tokens":1,"output_tokens":1}
        });
        assert_eq!(
            parse_response(&v).unwrap().finish_reason.as_deref(),
            Some("length")
        );

        let mut v2 = body(
            serde_json::json!([{"type":"text","text":"x"}]),
            "stop_sequence",
        );
        v2["stop_sequence"] = serde_json::json!("<END>");
        let r = parse_response(&v2).unwrap();
        assert_eq!(r.finish_reason.as_deref(), Some("stop"));
        assert_eq!(r.stop_sequence.as_deref(), Some("<END>"));
    }

    #[test]
    fn null_stop_reason_maps_to_none() {
        let mut v = body(serde_json::json!([{"type":"text","text":"x"}]), "end_turn");
        v["stop_reason"] = serde_json::Value::Null;
        assert_eq!(parse_response(&v).unwrap().finish_reason, None);
    }

    #[test]
    fn unknown_blocks_are_skipped() {
        let r = parse_response(&body(
            serde_json::json!([
                {"type":"server_tool_use","id":"srv_1","name":"web_search","input":{}},
                {"type":"text","text":"kept"}
            ]),
            "end_turn",
        ))
        .unwrap();
        assert_eq!(r.message.items.len(), 1);
    }

    #[test]
    fn missing_content_is_an_error() {
        assert!(parse_response(&serde_json::json!({"usage":{}})).is_err());
    }

    /// Round-trip against the server-side renderer: wire body → canonical
    /// chat JSON → Anthropic body → parsed reply must agree.
    #[test]
    fn round_trips_through_response_from_canonical() {
        let chat = serde_json::json!({
            "id": "chatcmpl-x",
            "choices": [{"message": {
                "role": "assistant",
                "content": "done",
                "tool_calls": [{"id":"toolu_9","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]
            }, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 7,
                      "cached_read_tokens": 2, "cache_write_tokens": 1}
        });
        let wire = super::super::out::response_from_canonical(&chat, "claude");
        let r = parse_response(&wire).unwrap();
        // text + tool both survive
        assert_eq!(r.message.items.len(), 2);
        assert_eq!(r.usage.prompt_tokens, 13);
        assert_eq!(r.usage.cached_read_tokens, 2);
        assert_eq!(r.usage.cache_write_tokens, 1);
        assert_eq!(r.finish_reason.as_deref(), Some("tool_calls"));
    }
}
