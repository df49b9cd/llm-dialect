//! Canonical response → Anthropic Messages wire shapes (non-stream
//! direction), plus the Anthropic error JSON mapping.
//!
//! Moved from the legacy `anthropic_in.rs`. Pure serde only: the async
//! error-body reader (`anthropic_error_from`) lives in `proxy_api.rs` —
//! reading an HTTP response body is I/O, not translation.

use super::stream::{anthropic_tool_use_id, map_stop_reason_outbound};
use crate::error::ProxyError;

pub fn response_from_canonical(r: &serde_json::Value, model: &str) -> serde_json::Value {
    let choice = &r["choices"][0];
    let msg = &choice["message"];
    let mut content: Vec<serde_json::Value> = Vec::new();
    // thinking blocks precede text/tool_use on the wire
    if let Some(blocks) = msg["thinking_blocks"].as_array() {
        for b in blocks {
            if b.is_object() {
                content.push(b.clone());
            }
        }
    }
    if let Some(t) = msg["content"].as_str()
        && !t.is_empty()
    {
        content.push(serde_json::json!({"type": "text", "text": t}));
    }
    if let Some(tcs) = msg["tool_calls"].as_array() {
        for (i, tc) in tcs.iter().enumerate() {
            let input: serde_json::Value = tc["function"]["arguments"]
                .as_str()
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or_else(|| serde_json::json!({}));
            let id = anthropic_tool_use_id(tc["id"].as_str(), r["id"].as_str().unwrap_or("0"), i);
            let name = tc["function"]["name"].as_str().unwrap_or("tool");
            content.push(serde_json::json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            }));
        }
    }
    let stop_reason = choice["finish_reason"]
        .as_str()
        .map(|fr| serde_json::json!(map_stop_reason_outbound(fr)))
        .unwrap_or(serde_json::Value::Null);
    let u = &r["usage"];
    // Canonical usage is cache-inclusive while Anthropic's wire reports
    // `input_tokens` fresh-only — subtract the cache classes for the client.
    let cached_read = u["cached_read_tokens"]
        .as_u64()
        .or_else(|| u["cache_read_input_tokens"].as_u64())
        .unwrap_or(0);
    let cache_write = u["cache_write_tokens"]
        .as_u64()
        .or_else(|| u["cache_creation_input_tokens"].as_u64())
        .unwrap_or(0);
    let mut usage = serde_json::json!({
        "input_tokens": u["prompt_tokens"].as_u64().unwrap_or(0).saturating_sub(cached_read + cache_write),
        "output_tokens": u["completion_tokens"].as_u64().unwrap_or(0),
    });
    // the canonical ChatResponse uses `cache_write_tokens`/`cached_read_tokens`;
    // raw upstream payloads pass through with Anthropic's own names instead.
    for (src, dst) in [
        ("cache_write_tokens", "cache_creation_input_tokens"),
        ("cached_read_tokens", "cache_read_input_tokens"),
        ("cache_creation_input_tokens", "cache_creation_input_tokens"),
        ("cache_read_input_tokens", "cache_read_input_tokens"),
    ] {
        if let Some(t) = u.get(src).and_then(|x| x.as_u64())
            && t > 0
        {
            usage[dst] = serde_json::json!(t);
        }
    }
    serde_json::json!({
        "id": r["id"].as_str().unwrap_or_default().replace("chatcmpl-", "msg_"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": usage,
    })
}

/// HTTP status → Anthropic `error.type` string. The single status→type table
/// for this dialect; both error paths consume it.
pub fn anthropic_error_type(status: u16) -> &'static str {
    match status {
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        503 => "overloaded_error",
        400 => "invalid_request_error",
        _ => "api_error",
    }
}

pub fn apply_stop_sequence_echo(
    out: &mut serde_json::Value,
    chat: &serde_json::Value,
    raw_req: &serde_json::Value,
) {
    if chat["choices"][0]["finish_reason"].as_str() != Some("stop") {
        return;
    }
    let Some(stops) = raw_req["stop_sequences"].as_array() else {
        return;
    };
    let stops: Vec<&str> = stops
        .iter()
        .filter_map(|s| s.as_str())
        .filter(|s| !s.is_empty())
        .collect();
    if stops.is_empty() {
        return;
    }
    let choice = &chat["choices"][0];

    // Some upstreams name the match outright (vLLM's `stop_reason`, SGLang's
    // `matched_stop`). Cheapest and most reliable signal when it exists;
    // Dynamo does not send it, hence the text scan below.
    for field in ["stop_reason", "matched_stop"] {
        if let Some(hit) = choice[field].as_str()
            && stops.contains(&hit)
        {
            mark_stop_sequence(out, hit);
            return;
        }
    }

    // Otherwise recognise the sequence in the generated text. `include_stop_str_
    // in_output` keeps it there for vLLM-family upstreams; without that the
    // text is already truncated and nothing can be matched.
    //
    // Reasoning models apply stop sequences to their thinking channel too, so
    // visible content is frequently empty while the thinking block is the one
    // carrying the match — scan both.
    for s in &stops {
        if strip_stop_from_blocks(out, s) {
            mark_stop_sequence(out, s);
            return;
        }
    }
}

/// Anthropic excludes the matched stop sequence from the text it returns.
/// Finds the first content block ending with `stop` and truncates it there;
/// an emptied text block is dropped rather than left as a stray empty block.
fn strip_stop_from_blocks(out: &mut serde_json::Value, stop: &str) -> bool {
    let Some(blocks) = out["content"].as_array_mut() else {
        return false;
    };
    let mut hit = false;
    for b in blocks.iter_mut() {
        let field = match b["type"].as_str() {
            Some("text") => "text",
            Some("thinking") => "thinking",
            _ => continue,
        };
        let Some(text) = b[field].as_str() else {
            continue;
        };
        if let Some(pos) = text.rfind(stop)
            && pos + stop.len() == text.len()
        {
            b[field] = serde_json::json!(&text[..pos]);
            hit = true;
        }
    }
    if hit {
        blocks.retain(|b| b["type"] != "text" || !b["text"].as_str().unwrap_or("").is_empty());
    }
    hit
}

fn mark_stop_sequence(out: &mut serde_json::Value, stop: &str) {
    out["stop_sequence"] = serde_json::json!(stop);
    out["stop_reason"] = serde_json::json!("stop_sequence");
}

/// Anthropic-shaped error body plus the matching HTTP status.
pub fn error_json(e: &ProxyError) -> (u16, serde_json::Value) {
    // Status comes from the shared table; BudgetExceeded is the one
    // deliberate dialect divergence (429 rate_limit_error is Anthropic's
    // billing-quota shape — a 403 permission error would confuse clients).
    let status = match e {
        ProxyError::BudgetExceeded => 429,
        other => other.http_status(),
    };
    // Client-facing messages for upstream failures stay generic: provider
    // error bodies can echo request material and are not client-safe (same
    // rule as state::error_response — the raw body stays in the spend log
    // and tracing).
    let message = match e {
        ProxyError::Upstream { status, .. } => format!("upstream returned status {status}"),
        other => other.to_string(),
    };
    let etype = anthropic_error_type(status);
    (
        status,
        serde_json::json!({
            "type": "error",
            "error": {"type": etype, "message": message},
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo(content: serde_json::Value, stops: serde_json::Value) -> serde_json::Value {
        let chat = serde_json::json!({
            "id": "chatcmpl-a",
            "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        });
        let mut out = serde_json::json!({
            "content": content, "stop_reason": "end_turn", "stop_sequence": null
        });
        apply_stop_sequence_echo(
            &mut out,
            &chat,
            &serde_json::json!({"stop_sequences": stops}),
        );
        out
    }

    #[test]
    fn stop_sequence_in_visible_text_is_reported_and_stripped() {
        let out = echo(
            serde_json::json!([{"type": "text", "text": "one two <END>"}]),
            serde_json::json!(["<END>"]),
        );
        assert_eq!(out["stop_reason"], "stop_sequence");
        assert_eq!(out["stop_sequence"], "<END>");
        assert_eq!(out["content"][0]["text"], "one two ");
    }

    #[test]
    fn stop_sequence_in_thinking_channel_is_reported() {
        // Reasoning models apply stop sequences to their thinking too; the
        // visible text is then empty and only the thinking block carries it.
        let out = echo(
            serde_json::json!([{"type": "thinking", "thinking": "plan: ONE TWO FIVE"}]),
            serde_json::json!(["FIVE"]),
        );
        assert_eq!(out["stop_reason"], "stop_sequence");
        assert_eq!(out["stop_sequence"], "FIVE");
        assert_eq!(out["content"][0]["thinking"], "plan: ONE TWO ");
    }

    #[test]
    fn upstream_stop_reason_field_is_honored_when_present() {
        // vLLM names the match outright; no text scan needed.
        let chat = serde_json::json!({
            "id": "chatcmpl-a",
            "choices": [{
                "message": {"role": "assistant", "content": "whatever"},
                "finish_reason": "stop",
                "stop_reason": "<END>"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        });
        let mut out = serde_json::json!({
            "content": [{"type": "text", "text": "whatever"}],
            "stop_reason": "end_turn", "stop_sequence": null
        });
        apply_stop_sequence_echo(
            &mut out,
            &chat,
            &serde_json::json!({"stop_sequences": ["<END>"]}),
        );
        assert_eq!(out["stop_reason"], "stop_sequence");
        assert_eq!(out["stop_sequence"], "<END>");
    }

    #[test]
    fn stop_sequence_mid_prose_is_not_a_match() {
        // the model mentioned it and kept going — that is a natural end_turn
        let out = echo(
            serde_json::json!([{"type": "text", "text": "I will not say <END> today"}]),
            serde_json::json!(["<END>"]),
        );
        assert_eq!(out["stop_reason"], "end_turn");
        assert_eq!(out["stop_sequence"], serde_json::Value::Null);
    }

    #[test]
    fn emptied_text_block_is_dropped_not_left_blank() {
        let out = echo(
            serde_json::json!([{"type": "text", "text": "<END>"}]),
            serde_json::json!(["<END>"]),
        );
        assert_eq!(out["stop_reason"], "stop_sequence");
        assert_eq!(out["content"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn colliding_upstream_tool_ids_are_namespaced_per_message() {
        // An OpenAI-dialect upstream that numbers tool calls per message
        // ("Read:0") hands back the same id on every turn. Two successive
        // responses must not present the client the same tool_use id, or the
        // client drops the second call as a duplicate and its tool never runs.
        let mk = |msg_id: &str| {
            serde_json::json!({
                "id": msg_id,
                "choices": [{"message": {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "Read:0", "function": {"name": "Read", "arguments": "{}"}}
                ]}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}
            })
        };
        let a = response_from_canonical(&mk("chatcmpl-aaa"), "translate-model");
        let b = response_from_canonical(&mk("chatcmpl-bbb"), "translate-model");
        let id_a = a["content"][0]["id"].as_str().unwrap();
        let id_b = b["content"][0]["id"].as_str().unwrap();
        assert!(id_a.starts_with("toolu_"), "{id_a}");
        assert_ne!(id_a, id_b, "ids must differ across messages");
    }

    #[test]
    fn parallel_tool_calls_in_one_message_keep_distinct_ids() {
        let r = serde_json::json!({
            "id": "chatcmpl-aaa",
            "choices": [{"message": {"role": "assistant", "content": null, "tool_calls": [
                {"id": "get_weather:0", "function": {"name": "get_weather", "arguments": "{}"}},
                {"id": "get_weather:1", "function": {"name": "get_weather", "arguments": "{}"}}
            ]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        });
        let a = response_from_canonical(&r, "translate-model");
        let x = a["content"][0]["id"].as_str().unwrap();
        let y = a["content"][1]["id"].as_str().unwrap();
        assert_ne!(x, y, "parallel calls must stay distinct");
    }

    #[test]
    fn anthropic_tool_ids_pass_through_verbatim() {
        let r = serde_json::json!({
            "id": "chatcmpl-aaa",
            "choices": [{"message": {"role": "assistant", "content": null, "tool_calls": [
                {"id": "toolu_01ABC", "function": {"name": "bash", "arguments": "{}"}}
            ]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        });
        let a = response_from_canonical(&r, "claude");
        assert_eq!(a["content"][0]["id"], "toolu_01ABC");
    }

    #[test]
    fn response_translation() {
        let r = serde_json::json!({
            "id": "chatcmpl-abc",
            "choices": [{"message": {"role": "assistant", "content": "hello", "tool_calls": null}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let a = response_from_canonical(&r, "translate-model");
        assert_eq!(a["type"], "message");
        assert_eq!(a["content"][0]["text"], "hello");
        assert_eq!(a["stop_reason"], "end_turn");
        assert_eq!(a["usage"]["input_tokens"], 10);
    }
}
