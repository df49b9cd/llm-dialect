//! Benches for the pure SSE framer hot path: every streamed token flows
//! through these translation cores. Covers the per-chunk framers for the two
//! client surfaces — the Anthropic framer state machine and the OpenAI chat
//! chunk→SSE renderer — plus a canary guarding the empty-chunk early-out.

use criterion::{Criterion, criterion_group, criterion_main};
use llm_dialect::canonical::{CanonChunk, ThinkingDelta, Usage};
use llm_dialect::dialect::anthropic::stream::{StreamState, chunk_to_sse_events};
use std::hint::black_box;

fn usage(p: u64, c: u64) -> Usage {
    Usage {
        prompt_tokens: p,
        completion_tokens: c,
        cached_read_tokens: 0,
        cache_write_tokens: 0,
        reasoning_tokens: None,
    }
}

fn text_chunk(s: &str) -> CanonChunk {
    CanonChunk {
        delta_text: s.to_string(),
        ..Default::default()
    }
}

/// Realistic Anthropic-upstream turn: thinking preamble, prose deltas, a
/// tool-use open + argument deltas, and a finish+usage terminator. The shape
/// real clients drive through an Anthropic /v1/messages surface.
fn anthropic_turn_chunks() -> Vec<CanonChunk> {
    let mut chunks = vec![CanonChunk {
        input_tokens: Some(18_000),
        ..text_chunk("")
    }];
    for i in 0..40 {
        chunks.push(CanonChunk {
            thinking: Some(ThinkingDelta {
                block_index: 0,
                kind: "thinking",
                text: format!("thinking segment {i} "),
            }),
            ..text_chunk("")
        });
    }
    for i in 0..60 {
        chunks.push(text_chunk(&format!("prose token {i} ")));
    }
    chunks.push(CanonChunk {
        tool_calls: Some(serde_json::json!([
            {"index":0,"id":"toolu_1","type":"function","function":{"name":"Bash","arguments":""}}
        ])),
        ..text_chunk("")
    });
    for i in 0..20 {
        chunks.push(CanonChunk {
            tool_calls: Some(serde_json::json!([
                {"index":0,"function":{"arguments":format!("{{\"cmd\":\"arg {i}\"}}")}}
            ])),
            ..text_chunk("")
        });
    }
    chunks.push(CanonChunk {
        finish_reason: Some("tool_calls".into()),
        usage: Some(usage(18_204, 900)),
        ..text_chunk("")
    });
    chunks
}

/// Frame one full turn through the Anthropic framer state machine.
fn frame_anthropic_turn(chunks: &[CanonChunk]) -> Vec<(String, String)> {
    let mut st = StreamState::new();
    let mut out = Vec::new();
    for c in chunks {
        out.extend(chunk_to_sse_events(
            c,
            "translate-model",
            &mut st,
            "msg_bench",
        ));
    }
    out
}

fn bench_framer(_c: &mut Criterion) {
    // The framer benches are bimodal: a single cold-start / frequency-scaling
    // right-tail sample can shift a short default measurement (5s, 3s warmup)
    // well past noise. A longer measurement + more samples lands a stable
    // point-estimate so the calibrated noise band is meaningful (one outlier
    // a band inflated by one outlier is useless as a veto).
    let mut cfg = Criterion::default()
        .measurement_time(std::time::Duration::from_secs(20))
        .warm_up_time(std::time::Duration::from_secs(5))
        .sample_size(200);
    let mut group = cfg.benchmark_group("framer");

    // Whole-turn framing: the dominant per-request cost shape. 123 chunks
    // (prelude + 40 thinking + 60 prose + tool-open + 20 args + finish).
    let turn = anthropic_turn_chunks();
    group.bench_function("anthropic_turn_full", |b| {
        b.iter(|| frame_anthropic_turn(black_box(&turn)))
    });

    // Per-delta framing on a steady-state prose stream (the classic hot loop:
    // one text delta per chunk, stream state already open).
    group.bench_function("anthropic_text_delta", |b| {
        b.iter(|| {
            let mut st = StreamState::new();
            let prelude = CanonChunk {
                input_tokens: Some(100),
                ..text_chunk("")
            };
            chunk_to_sse_events(black_box(&prelude), "m", &mut st, "msg_bench");
            for i in 0..100 {
                chunk_to_sse_events(
                    black_box(&text_chunk(&format!("delta {i} "))),
                    "m",
                    &mut st,
                    "msg_bench",
                );
            }
        })
    });

    // OpenAI chat surface: CanonChunk → SSE data frames for standard
    // OpenAI-compatible clients.
    group.bench_function("openai_chat_to_sse_json", |b| {
        b.iter(|| {
            let mut out = Vec::new();
            for i in 0..100 {
                let chunk = text_chunk(&format!("delta {i} "));
                if let Some(json) = chunk.to_sse_json("chatcmpl-bench", "m", 0, false) {
                    out.push(json);
                }
            }
            out
        })
    });

    // OpenAI chat surface with tool-call deltas (strict merge-by-index
    // clients omit `content` on these).
    group.bench_function("openai_chat_tool_delta", |b| {
        b.iter(|| {
            let mut out = Vec::new();
            for i in 0..100 {
                let chunk = CanonChunk {
                    tool_calls: Some(serde_json::json!([
                        {"index":0,"type":"function","function":{"arguments":format!("{{\"i\":{i}}}")}}
                    ])),
                    ..text_chunk("")
                };
                if let Some(json) = chunk.to_sse_json("chatcmpl-bench", "m", 0, false) {
                    out.push(json);
                }
            }
            out
        })
    });

    // Canary: the empty-chunk early-out in to_sse_json — a guard on the
    // no-work fast path. Excluded from the geomean by the scorer (canary_*
    // naming); a regression here rejects the candidate outright.
    group.bench_function("canary_empty_chunk_skip", |b| {
        b.iter(|| {
            let chunk = CanonChunk::default();
            black_box(chunk.to_sse_json("chatcmpl-bench", "m", 0, false).is_none())
        })
    });

    group.finish();
}

criterion_group!(benches, bench_framer);
criterion_main!(benches);
