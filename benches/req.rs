//! Benches for the request-parse hot path: every wire request flows through
//! these dialect adapters before anything else runs.
//!
//! The other half of AVO scoring (the response/framer path) lives in
//! `benches/framer.rs`; these benches cover what that file's module doc says
//! it does not: req parsing. Standards match the framer bench (long
//! measurement, 200 samples) so the noise floor is meaningful.

use criterion::{Criterion, criterion_group, criterion_main};
use llm_dialect::dialect::anthropic::req::from_anthropic;
use llm_dialect::dialect::openai_chat::req::from_openai_chat;
use llm_dialect::dialect::openai_responses::req::from_openai_responses;
use std::hint::black_box;

/// Realistic Anthropic multi-turn request: system prompt, plain turns,
/// a signed thinking block, a tool_use/tool_result pair, one final plain turn.
fn anthropic_turn() -> serde_json::Value {
    serde_json::json!({
        "model": "claude-opus-4-5",
        "max_tokens": 4096,
        "system": "You are a helpful assistant.",
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "tools": [{
            "name": "Bash",
            "description": "Run a shell command",
            "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}}
        }],
        "messages": [
            {"role": "user", "content": "Summarise this directory listing."},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "The user pasted an ls -la output.", "signature": "sig_abc_def"},
                {"type": "text", "text": "The directory contains two files: report.pdf and notes.md."}
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_bench", "name": "Bash",
                 "input": {"cmd": "ls -la /tmp"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_bench",
                 "content": "total 0\ndrwxr-xr-x 2 user user 4096 ..."}
            ]},
            {"role": "assistant", "content": "Empty /tmp — nothing to clean."},
            {"role": "user", "content": "Thanks!"},
        ]
    })
}

/// Realistic OpenAI chat multi-turn request: system + interleaved
/// user/assistant/tool messages with a tool_calls side-array.
fn openai_chat_turn() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "List the top 3 uses."},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result content"},
            {"role": "assistant", "content": "Use one, use two, use three."},
            {"role": "user", "content": "Thanks."},
        ]
    })
}

/// Realistic OpenAI Responses request: typed items (message, reasoning, function_call).
fn openai_responses_turn() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4o",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Explain this code."}]},
            {"type": "reasoning", "summary": []},
            {"type": "function_call", "name": "read_file", "arguments": "{}", "call_id": "call_2"},
            {"type": "function_call_output", "call_id": "call_2", "output": "fn main() {}"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "It prints nothing."}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Ok."}]},
        ]
    })
}

/// Wire-conformance canary: an empty messages array must reject fast (early
/// return, no allocation-heavy path — this is the absence-of-work guard).
/// Excluded from the geomean by the scorer (canary_* naming, P1.3): its job
/// is the veto — a regression here rejects the candidate outright.
fn canary_empty_messages() -> serde_json::Value {
    serde_json::json!({
        "model": "claude-opus-4-5",
        "max_tokens": 1,
        "messages": []
    })
}

fn bench_req(_c: &mut Criterion) {
    // Same hedging as framer.rs: short default measurements are bimodal at
    // single-digit-µs scale; a longer measurement + more samples gives the
    // AVO noise band a real point estimate to guard.
    let mut cfg = Criterion::default()
        .measurement_time(std::time::Duration::from_secs(20))
        .warm_up_time(std::time::Duration::from_secs(5))
        .sample_size(200);
    let mut group = cfg.benchmark_group("req");

    let a = anthropic_turn();
    group.bench_function("anthropic_turn", |b| {
        b.iter(|| from_anthropic(black_box(&a)).unwrap())
    });

    let oc = openai_chat_turn();
    group.bench_function("openai_chat_turn", |b| {
        b.iter(|| from_openai_chat(black_box(&oc)).unwrap())
    });

    let ors = openai_responses_turn();
    group.bench_function("openai_responses_turn", |b| {
        b.iter(|| from_openai_responses(black_box(&ors)).unwrap())
    });

    group.bench_function("canary_empty_messages", |b| {
        b.iter(|| black_box(from_anthropic(black_box(&canary_empty_messages())).is_err()))
    });

    group.finish();
}

criterion_group!(benches, bench_req);
criterion_main!(benches);
