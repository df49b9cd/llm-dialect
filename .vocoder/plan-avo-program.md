# AVO program plan — llm-dialect

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
