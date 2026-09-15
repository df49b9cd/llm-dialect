//! Canonical `ChatResponse` → OpenAI Responses JSON body (non-stream).
//!
//! Chain-of-thought precedes visible output; terminal status reflects the
//! finish reason (`length`/`content_filter` → `incomplete` with the matching
//! reason, never a false `completed`).

use crate::canonical::ChatResponse;

/// Render the canonical `ChatResponse` as an OpenAI Responses body.
pub fn responses_body_from_chat(chat: &ChatResponse, model: &str) -> serde_json::Value {
    let mut output: Vec<serde_json::Value> = Vec::new();
    for choice in &chat.choices {
        let msg = &choice.message;
        // Chain-of-thought precedes visible output: Anthropic-signed thinking
        // and vLLM-style reasoning_content both become `reasoning` items with
        // summary text. Redacted payloads have no Responses channel — skipped.
        if let Some(blocks) = msg.thinking_blocks.as_ref() {
            for b in blocks {
                if let Some(t) = b["thinking"].as_str().filter(|s| !s.is_empty()) {
                    output.push(serde_json::json!({
                        "type": "reasoning",
                        "id": format!("rs_{}", uuid::Uuid::new_v4().simple()),
                        "summary": [{"type": "summary_text", "text": t}],
                    }));
                }
            }
        }
        if let Some(rc) = msg.reasoning_content.as_deref().filter(|s| !s.is_empty()) {
            output.push(serde_json::json!({
                "type": "reasoning",
                "id": format!("rs_{}", uuid::Uuid::new_v4().simple()),
                "summary": [{"type": "summary_text", "text": rc}],
            }));
        }
        let text = msg.text();
        if !text.is_empty() {
            output.push(serde_json::json!({
                "type": "message",
                "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
                "role": "assistant",
                "status": "completed",
                "content": [{"type":"output_text","text": text, "annotations": []}],
            }));
        }
        if let Some(tcs) = msg.tool_calls.as_ref().and_then(|t| t.as_array()) {
            for tc in tcs {
                // `id` is the Responses item id (`fc_…`); `call_id` is the
                // call identifier the client echoes back — keep them distinct
                // (the chat canonical only carries the call id)
                output.push(serde_json::json!({
                            "type": "function_call",
                            // Responses function_call ids use the fc_ prefix per the schema.
                "id": format!("fc_{}", uuid::Uuid::new_v4().simple()),
                            "call_id": tc["id"],
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"].as_str().unwrap_or("{}"),
                            "status": "completed",
                        }));
            }
        }
    }
    let u = &chat.usage;
    // "length" means the model hit the cap: the Responses contract is
    // status incomplete + incomplete_details.reason, not a false "completed"
    let (status, incomplete) = match chat
        .choices
        .first()
        .and_then(|c| c.finish_reason.as_deref())
    {
        Some("length") => (
            "incomplete",
            Some(serde_json::json!({"reason": "max_output_tokens"})),
        ),
        Some("content_filter") => (
            "incomplete",
            Some(serde_json::json!({"reason": "content_filter"})),
        ),
        _ => ("completed", None),
    };
    let mut resp = serde_json::json!({
        "id": chat.id,
        "object": "response",
        "created_at": chat.created,
        "status": status,
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        },
    });
    if let Some(d) = incomplete {
        resp["incomplete_details"] = d;
    }
    if let Some(r) = u.reasoning_tokens {
        resp["usage"]["output_tokens_details"] = serde_json::json!({"reasoning_tokens": r});
    }
    resp
}
#[cfg(test)]
mod responses_body_tests {
    use super::*;
    use crate::canonical::Usage;

    #[test]
    fn non_stream_body_keeps_chain_of_thought() {
        let mut chat = ChatResponse::new(
            "m",
            "the answer".into(),
            Some("stop".into()),
            Usage::default(),
        );
        chat.choices[0].message.thinking_blocks = Some(vec![
            serde_json::json!({"type":"thinking","thinking":"the plan","signature":"sig"}),
            serde_json::json!({"type":"redacted_thinking","data":"opaque"}),
        ]);
        chat.choices[0].message.reasoning_content = Some("loose reasoning".into());
        let body = responses_body_from_chat(&chat, "m");
        let out = body["output"].as_array().unwrap();
        let types: Vec<&str> = out.iter().map(|i| i["type"].as_str().unwrap()).collect();
        // signed thinking + reasoning_content survive; redacted is skipped
        assert_eq!(types, vec!["reasoning", "reasoning", "message"]);
        assert_eq!(out[0]["summary"][0]["text"], "the plan");
        assert_eq!(out[1]["summary"][0]["text"], "loose reasoning");
        assert_eq!(out[2]["content"][0]["text"], "the answer");
    }
}

#[test]
fn body_includes_reasoning_token_detail_when_present() {
    let mut chat = crate::canonical::ChatResponse::new(
        "m",
        "done".into(),
        Some("stop".into()),
        crate::canonical::Usage::default(),
    );
    chat.usage.reasoning_tokens = Some(42);
    let body = responses_body_from_chat(&chat, "m");
    assert_eq!(
        body["usage"]["output_tokens_details"]["reasoning_tokens"],
        serde_json::json!(42)
    );
    // absent when provider didn't report them
    chat.usage.reasoning_tokens = None;
    let body = responses_body_from_chat(&chat, "m");
    assert!(body["usage"].get("output_tokens_details").is_none());
}
