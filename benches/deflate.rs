//! Benches for the deflate pipeline: the items-canonical → chat-request fold.
//!
//! Exercises the req→deflate chain end-to-end on a realistic multi-turn
//! Anthropic request so the scoring includes *both* the parse shape and the
//! fold shape (not one in isolation). The output side (`out.rs` /
//! `to_sse_json`) stays covered by `benches/framer.rs`.

use criterion::{Criterion, criterion_group, criterion_main};
use llm_dialect::dialect::anthropic::req::from_anthropic;
use llm_dialect::dialect::deflate::items_to_chat_request;
use std::hint::black_box;

/// Same realistic shape as benches/req.rs — kept self-contained so the two
/// scores stay comparable under any future change to either bench file.
fn anthropic_turn_wire() -> serde_json::Value {
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

fn bench_deflate(_c: &mut Criterion) {
    // The deflate input is the items canonical — one fold per request. The
    // chain bench calls both in sequence so per-candidate movement is split
    // between parse and fold by the per-bench metrics (they drift
    // independently and both enter the geomean).
    let mut cfg = Criterion::default()
        .measurement_time(std::time::Duration::from_secs(20))
        .warm_up_time(std::time::Duration::from_secs(5))
        .sample_size(200);
    let mut group = cfg.benchmark_group("deflate");

    // The parse half lives in benches/req.rs for now; deflate's diff lands on
    // the fold itself. Callers wire them sequentially in production, so the
    // bench does the same but reports the half that moves.
    let wire = anthropic_turn_wire();
    group.bench_function("req_to_items_to_deflate_chain", |b| {
        b.iter(|| {
            let items = from_anthropic(black_box(&wire)).unwrap();
            items_to_chat_request(black_box(&items)).unwrap()
        })
    });

    group.finish();
}

criterion_group!(benches, bench_deflate);
criterion_main!(benches);
