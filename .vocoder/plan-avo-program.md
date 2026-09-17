# AVO program plan — llm-dialect

## Pause note (framer scope — 2026-09-17, after x5)

**Verdict history.** x0 104708 → x1 −74.3% → x2 −8.6% → x3 −4.6% (within-band rollup,
per-bench wins) → x4 rebaseline (scoring v2) → x5 −12.0% geomean driven by chat_to_sse.
Two consecutive non-improving verdicts on the geomean roll-up (x3 partial, r6-era band
widening absorbed the delta) — rule 4 applies; and the structural operator pool is
exhausted: every frame is now single-buffer, all dynamic leaves escape in one pass,
event names are borrowed statics, and the prose-delta profile is 83 ns/chunk of the
remaining turn_turn 13104 ns — thinking/tool/terminal frames dominate and are already
minimal.

**Where the remaining cost lives** — nothing obvious left in the serializer:
- `anthropic_turn_full` 18482 ns: 40 thinking chunks (block-open close-reopen + one
  alloc each) + 20 tool-arg frames + the terminal pair. All single-buffer single-alloc
  already; further cuts mean rethinking block-index bookkeeping, not assembly.
- `anthropic_text_delta` 13022 ns: two String allocs per chunk (frame + escaped text
  member in `write_json_str` on escape-heavy input, none on clean input) — already at
  the allocator floor.
- `openai_chat_tool_delta` 53740 ns: `tcs.to_string()` embed — serde_json's own
  serializer on a Value it just parsed upstream. Real win possible by routing tool_call
  deltas through the same literal-with-escapes path as `text_delta` (the OpenAI wire
  shape for tool deltas is fixed at ~8 fields). That's a small constant-factor chase.

**Decision: pause the framer scope at x5.** Diminishing returns (rule 7) — the slope
flattened across 3 committed versions even before the rebaseline; further wins are
cycle-level fiddling on code that's already 5× faster than x0. If a fresh prompt shows
up (e.g. a Gemini dialect adapter or a resize of the SSE pump), start a new scope
against it rather than grinding this one further.

**Next-scope candidates for when work resumes:** (1) `deflate.rs` items→chat flattening
(request path isn't bench-instrumented at all); (2) a `req.rs` parse-cost scope —
the inbound surface, zero bench coverage today.

## Steering note (after r5 rejection — 2026-09-16, scope: framer)

**Situation.** r5 (one-buffer assembly completion + escape-scan fast path) improved
every primary bench (geomean −13.0%, chat_to_sse −29.3%) but was rejected by the
canary veto: `canary_empty_chunk_skip` +4.21% beyond its ±3.0% band. Re-benching
the canary alone gave 5.948 ns — the delta is real *for that build*, not run noise.
r2 was rejected the same way (+4.34%), and r3 passed only at +2.98%. Pattern: the
canary benches ONE `to_sse_json` empty-chunk call (~5.8 ns). At that scale,
cross-build codegen/layout jitter (which function lands where in the text section)
moves the number ±4% — larger than the canary's calibrated run band (±3%, from
cv ≈ 0.2–0.5% within a build). Every round that touched `canonical.rs` flipped it;
the stream.rs-only round (r4) moved it the other way (−3.4%). The veto is currently
guarding binary layout, not the early-out.

**Redirection options considered:**

1. **Fix the canary's measurement** (scoring change + rebaseline, EVO.md rule 8 /
   the "fix the score function first" gotcha): bench 1000 empty chunks per
   iteration so layout jitter amortizes to noise; ratchet `scoring.version`, land
   a `rebaseline: true` row, recalibrate noise under the new shape. A real
   early-out regression (+10 ns/call class) still vetoes at +100%+; layout
   coin-flips stop mattering. Cost: one scoring round + 2× noise calibration.
2. **Constrain future mutations to `anthropic/stream.rs`** (leave `canonical.rs`
   alone): r4 shows the canary then stays quiet. Cost: forfeits the
   escape-scan fast path — the single biggest remaining win (−29% on the
   chat surface) lives in `canonical.rs`.
3. **Accept the veto and pause the scope at x3** (−77.6% from x0). Safe, but
   leaves a measured −13% geomean on the table and the canary stays a
   coin-flip for any future round.
4. **Widen the canary band to ~5%** to cover observed cross-build variance.
   Cheapest, but weakens the veto for real regressions too and bakes a fudge
   factor into the contract.

**Decision: option 1.** The canary's purpose — guard the empty-chunk early-out —
is preserved; its measurement becomes robust to build-to-build layout. After the
rebaseline, re-land the r5 diff (saved at /tmp/r5.diff) as the next candidate and
let the verdict run under the new calibration. If the re-landed r5 STILL trips the
amortized canary, the regression is real and the candidate stays rejected.
