# AVO program plan — llm-dialect

## Pause note (req scope — 2026-09-17, after x0 baseline)

**Why the scope doesn't proceed to mutation rounds.** The first candidate
experiment in `r1-req` was rejected by the calibration itself, and the profile
that preceded it diagnoses why: `serde_json::Value` is the wrong substrate for
this loop.

Measured shape at accept-time:
- `req/anthropic_turn` 1600 ns/request on the 9-block realistic multi-turn shape
  (bench suite in `benches/req.rs`, baseline at `dd53d84`).
- Bare map-lookup share (the naive `b["key"]` scan) is ~28 ns — **2%** of the
  call. The r1 mutation ("bind `b.as_object()` once per block, reuse it for
  every field read") changed 1591→1523 ns: a **−5.0% delta for ~99% of the
  suspected mechanism**. The candidate is accepted on that number (±3% noise
  band would mark it beyond, and the result was stable across restarts) —
  but it doesn't matter. Even a magic zero-cost `Map::get` would recover
  ~30 ns of the 1150-ns baseline.

The other 98%:
- **Memory traffic on allocs:** every content block pushes `ContentItem::Text`
  (a fresh `String`, its own heap header) plus a `block_cc` `Vec<Option<Value>>`
  which is essentially always all-`None` and immediately cleared. The allocs
  dominate because the per-block work (match + a few reads) is sub-100 ns.
  The tool_use × tool_result path alone contributes ~500 ns of the 1308
  (measured in `examples/req_split.rs`); that's `String` init + the
  always-propagating `block_cc` `Vec`.
- **Match dispatch cost per block is ~100 ns** on this crate: 9 blocks, so
  ~900 ns go through the `match ty {…}` arm alone, driven by which shape is
  matched (text vs. thinking vs. tool_use vs. tool_result). With only ~100 ns
  of budget left per block after the string allocs, there's no single
  hard-won operator that saves more than a few ns — the win is proportional
  to structure, not to any specific field access.

**Structural lesson (the conclusion, not a mutation).** `serde_json::Value`
is the correct parser output *for this codebase*: it keeps the shape
"whatever the wire put here" and remains byte-preserving on unknown keys
(forward-compat pragmatism, the same reason the unknown-anthropic-blocks arm
silently swallows server_tool_use). The cost of chasing it into a typed
deserialize (`ItemRequest::{de}derive`) and back out again is structural —
`ItemRequest` has no `serde` derive, and adding one reopens the parse-as
surface this crate deliberately refuses to own (the JSON envelope is routed,
not walked). The slow path is the price of staying pure: accept it.

**Mutation pool that *finds* nothing at this scale but costs a full round
each time anyway:** cache-control Vec hoisting (it's already all-`None`),
role-str interning (no measurable alloc load), match-arm reordering (hot-arm
first is already sub-ns): every one of those is a ~5–10 ns guess on a 1300-ns
budget. The slope is not flattening — it's flat.

**Where the next scope should start instead:** the `deflate.rs` and
`out.rs` paths are unbenched; each writes its own bench surface if they
matter. Nothing in this scope affects the framer or the renderers.

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
