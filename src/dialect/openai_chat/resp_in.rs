//! OpenAI Chat Completions response body → canonical [`ChatResponse`]
//! (client direction).
//!
//! The chat dialect is the canonical shape — parsing is plain serde into
//! the canonical type. The one normalization the wire needs is usage
//! enrichment: OpenAI reports cached tokens under
//! `usage.prompt_tokens_details.cached_tokens`, and the canonical model
//! lifts that into [`Usage::cached_read_tokens`] so `prompt_tokens`
//! (cache-inclusive per OpenAI's convention) needs no further arithmetic.

use crate::canonical::ChatResponse;
use crate::error::ProxyError;

pub fn parse_response(v: &serde_json::Value) -> Result<ChatResponse, ProxyError> {
    let mut r: ChatResponse = serde_json::from_value(v.clone())
        .map_err(|e| ProxyError::Transport(format!("malformed chat completion body: {e}")))?;
    if let Some(cached) = v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64() {
        r.usage.cached_read_tokens = cached;
    }
    if let Some(reasoning) = v["usage"]["completion_tokens_details"]["reasoning_tokens"].as_u64() {
        r.usage.reasoning_tokens = Some(reasoning);
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_chat_body_parses() {
        let v = serde_json::json!({
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
            "model": "gpt",
            "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let r = parse_response(&v).unwrap();
        assert_eq!(r.choices[0].message.text(), "hi");
        assert_eq!(r.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(r.usage.prompt_tokens, 10);
        assert_eq!(r.usage.cached_read_tokens, 0);
    }

    #[test]
    fn prompt_cache_details_lift_to_canonical() {
        let v = serde_json::json!({
            "id": "c", "object": "chat.completion", "created": 0, "model": "gpt",
            "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "x"}}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18,
                      "prompt_tokens_details": {"cached_tokens": 3},
                      "completion_tokens_details": {"reasoning_tokens": 7}}
        });
        let r = parse_response(&v).unwrap();
        assert_eq!(r.usage.cached_read_tokens, 3);
        assert_eq!(r.usage.reasoning_tokens, Some(7));
    }

    #[test]
    fn tool_calls_parse_verbatim() {
        let v = serde_json::json!({
            "id": "c", "object": "chat.completion", "created": 0, "model": "gpt",
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "bash", "arguments": "{}"}}
                ]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        let r = parse_response(&v).unwrap();
        assert_eq!(
            r.choices[0].message.tool_calls.as_ref().unwrap()[0]["id"],
            "call_1"
        );
    }

    #[test]
    fn malformed_body_is_a_transport_error() {
        assert!(parse_response(&serde_json::json!({"nope": 1})).is_err());
    }
}
