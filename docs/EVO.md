# EVO — Agentic Variation Operators

This repo adopts NVIDIA's **Agentic Variation Operators** (AVO,
arXiv:2603.24517) both as a **discipline** (the commit-only-on-improve loop
below) and, where this repo *is* the vocoder-tui harness, as a **product
feature** (the `/evo` dashboard). This file is the stable reference; the
living ledger lives in `.vocoder/plan-avo-program.md` (vocoder-tui) or your
repo's equivalent plan file. Design constants are tagged **P** (from the
paper) or **N** (our translation).

## The operator

Classical evolutionary search decomposes variation as
`Vary(P) = Generate(Sample(P))` — a fixed sampler picks parents, a model is
confined to one-shot generation, and a surrounding framework manages the loop.
AVO replaces the entire operator with a single self-directed agent run:

```
Vary(P_t) = Agent(P_t, K, f)
```

- **P_t — the lineage.** Committed versions + their scores. Here: git commits
  + `.vocoder/evo/population.json` rows.
- **K — the knowledge base.** Here: `AGENTS.md`, `docs/`, benchmark baselines,
  and reference implementations the agent consults.
- **f — the scoring function.** Here: the correctness gate
  (`cargo build --workspace --all-features` + `cargo fmt --check` +
   `cargo nextest run --workspace --all-features` +
   `cargo clippy --workspace --all-targets --all-features -- -D warnings`)
   gates every candidate; the score tuple (bench geomean,
  warning count, test count) ranks survivors per scope. Computed uniformly by
  `.scripts/evo-score.sh`.

## Process rules (the contract)

1. **Correctness gate is absolute** (P). A candidate that fails the gate is
  score-zero regardless of any other metric. Commit only candidates that pass
  the gate.
2. **Commit only on improvement** (P). A new version is persisted when it
  passes the gate AND its scope-score matches or beats the best committed
  score. Failed attempts stay in the working tree / transcript only.
  **Improvement is measured beyond the noise band** (P0.2): a delta counts
  only when it exceeds the metric's demonstrated noise floor — run
  `.scripts/evo-score.sh --noise N` (N≥3, ideally twice, merged with
  `--noise-merge` keeping the worst band) to calibrate, and treat any delta
  inside `improve_band_pct` as indistinguishable from noise. This guards
  against committing noise wins (the E7 bimodal-bench lesson). Each committed
  row records the active calibration as `noise_band` (P1.6), so the lineage
  itself shows which deltas were beyond demonstrated noise.
3. **One commit per version, scores recorded** (P). Every stable round lands
  as its own git commit with its score tuple appended to `population.json`
  (via `.scripts/evo-score.sh`). The lineage is replayable.
3. **Persist round findings to memory** (AX #13 session culture; the harness
  `memory` tool writes topic files recalled into future sessions). After a
  committed version, write a short note to the `project` target under a topic
  slug like `evo-<scope>` — the measured delta, the mechanism, the gotcha.
  Future AVO sessions in the repo then *recall* it as context, so the
  lineage's hard-won facts compound across sessions and contributors. The
  note is the durable artifact; the population file is the structured score.
4. **Stagnation supervisor** (P mechanism, N cadence). Two consecutive
  non-improving candidates in a scope trigger a steering note before round 3:
  revisit the trajectory, list 2–4 candidate redirections, pick one. The note
  is recorded in the plan file.
5. **Scores are comparable** (N). All scores in one scope come from
  `.scripts/evo-score.sh` on the same machine profile; note hardware/theme
  changes in the version's `note`.
6. **Scope score functions** (N):
  - `perf`: `bench_geomean_ns` (drive down). The scorer ALSO emits every
    individual bench as its own top-level metric (granularity, N): a
    candidate verdict pinpoints exactly WHICH bench regressed beyond its
    band instead of only seeing the roll-up move, and new benches join the
    comparison as soon as baseline and candidate both carry them.
  - `quality`: linter warnings (drive to 0) + test count (drive up)
    + doc-coverage ratchets.
  - `docs`: link integrity + section staleness + AGENTS.md size cap.
  - **Canary benches** (P1.3): a perf scope may designate sentinel
    benches by naming them `canary_*`. The scorer EXCLUDES them from
    `bench_geomean_ns` (the aggregate measures the broad surface, not the
    sentinels it guards) and emits each as its own `canary_<name>_ns`
    metric. `evo-compare` then rejects a candidate that
    regresses ANY canary beyond its own noise band, even when the primary
    geomean improved — a canary is a veto, not an average. Calibrate
    canary bands with `--noise` like any other bench (the canary key
    `canary_<name>_ns` is stable across scorer + calibration + verdict).
7. **Diminishing returns are expected** (P, Fig. 5–6). Early rounds capture
  structural gains; later rounds are cycle-level. When the slope flattens
  across ≥3 committed versions, pause the scope and redirect (rule 4) — fresh
  perspective beats finer grinding.
8. **Scoring is versioned; a changed `f` rebaselines** (P0.3). The population
  file carries a `scoring` block (`version`, `gate`, `f`). Whenever the score
  function changes meaningfully (new metric, re-baselined corpus, scorer
  rewrite), ratchet `scoring.version` AND land a `rebaseline: true` row that
  re-establishes the baseline under the new `f`. Versions before a rebaseline
  were scored under the old `f` and are never compared against versions after
  it (`/evo` resets best-version selection at the most recent rebaseline).
  This formalizes the ad-hoc re-baseline E7 did when its bench corpus changed.
  Each population row also records a hash of the scorer that produced its
  scores (`scorer_sha`, P1.7); `/evo` warns when consecutive rows were scored
  by different scorers with no rebaseline row between them — rule 8 as a
  check, not just discipline.

## Mutation pool (first-mutation operators)

Round 1 of a new scope draws 2–3 operators from this pool as its first
candidate suggestions — the house's proven winners, codified (P1.4). Each
entry: mechanism, when it applies, and the lineage evidence that paid for it.

- **Early-out fast path.** Detect the common no-work case in O(small) and
  return before the expensive pipeline runs. Applies when most inputs need
  none of the work (plain prose through a markdown stitcher, cache-hit
  renders). Evidence: `repair_streaming` Cow::Borrowed fast path (E1 x1,
  −13.5% geomean); marker-absence early-out in mdstitch `stitch()` (E7 x1,
  plain-prose −85.6%, 6.7×).
- **SIMD byte search.** Replace scalar byte loops with memchr/memchr2/
  memchr3 (or equivalent) for trigger/scan passes. Applies to any per-byte
  scan over large text. Evidence: memchr-based trigger scan (E7 x2,
  plain-prose −20.7%; marker-heavy inputs improved too — SIMD hits the first
  trigger faster than scalar reached it).
- **Alloc elision / borrow reuse.** Return borrowed data when the owned copy
  is only needed on the rare path (`Cow::Borrowed` + `into_owned()` at the
  branch). Applies to hot paths that clone unconditionally.
- **Incremental resume.** Carry scan/parse state across deltas so repeat work
  is O(delta), not O(document). Evidence: incremental boundary scan (E1 x2,
  prod incremental path −16.8% at 256 KiB). Gotcha: resume from a LINE/FENCE
  boundary, not mid-token — splitting a marker corrupts parity.
- **Precompute / cache locality.** Hoist per-call recomputation into a
  per-block or per-width cache keyed on a cheap fingerprint.
- **Constant folding / branchless conversion.** Small constant-factor wins;
  take only after the structural operators above are exhausted (diminishing
  returns rule 7).

Profile before mutating: the pool suggests WHERE, the profiler says WHETHER.
Two rejected candidates in a row ⇒ steering note (rule 4), and reconsider the
score function itself before round 3 (the E7 score/optimization mismatch).

## Applying AVO to another repo

```bash
python3 <skill-dir>/scripts/evo-init [target-dir] [--scope <name>] [--lang <rust|node|python|generic>] [--force]
```

Drops a working AVO workspace into `target-dir` (defaults to the current
directory): `.scripts/evo-score.sh`, `.scripts/evo-kickoff.sh`,
`.scripts/evo-candidate.sh`, and this `docs/EVO.md`. The rule-2 verdict is
NOT a script — it is pure JSON math, shipped as
`python3 <skill-dir>/scripts/evo-compare BASELINE CANDIDATE [NOISE]`
(exit 0 improving / 1 neutral / 2 rejected). The `--lang` picks the scorer
template; omit it and the lang is inferred from the target
(`Cargo.toml`→rust, `package.json`→node, `pyproject.toml`/`setup.py`→python,
else generic). `--scope` also seeds `.vocoder/evo/population-<scope>.json` in
one step (init + kickoff combined). `--force` overwrites existing files
(otherwise the kit refuses to clobber a repo that already has them).

The same kit is also shipped by the vocoder harness as `vocoder --evo-init`;
the `.scripts/` files it writes are identical.

## Kicking off a new scope

```bash
.scripts/evo-kickoff.sh <scope> [description]   # seeds population-<scope>.json
```

Then:

1. **Pick ONE primary score** for the scope (perf: `bench_geomean_ns` or a targeted `cargo bench` bench; quality: clippy warnings + test count; docs: staleness metric). Multi-objective scopes drift — the paper's wins came from a clear `f`.
2. **Baseline x0**: `cargo bench` (if perf) then `.scripts/evo-score.sh`; paste the tuple into `x0.scores` in the population file.
3. **Run the operator loop**: consult K → one candidate edit → gate (`.scripts/evo-score.sh --gate`) → score → commit only on improvement.
4. **Steer on stagnation** (rule 4), **stop on diminishing returns** (rule 7).
5. **View**: `VOCODER_EVO_POPULATION=.vocoder/evo/population-<scope>.json` then `/evo` in the vocoder TUI (the env override isolates concurrent scopes).

In a vocoder-tui checkout the loop has first-class commands: `/evo run
<scope>` scaffolds a round (candidate worktree + protocol), `/evo finish
<name>` computes the verdict under the noise band (read-only), `--apply`
merges an improving diff into the main tree, `--record` appends the lineage
row and discards the worktree, and `/evo portfolio` shows every scope's
lineage depth + best version in one table. Elsewhere the same steps are the
four `.scripts/` commands above.

## File layout

```
.vocoder/evo/population.json            # the main lineage (scratch)
.vocoder/evo/population-<scope>.json    # per-scope lineages (evo-kickoff.sh)
.vocoder/evo/noise.json                 # merged noise calibration (optional)
.vocoder/evo/worktrees/                 # candidate worktrees (scratch)
.scripts/evo-score.sh                   # uniform score computation
.scripts/evo-kickoff.sh                 # seed a new scope's lineage
.scripts/evo-candidate.sh               # candidate isolation (git worktrees)
docs/EVO.md                             # this file (the contract)
                                        # rule-2 verdict: `scripts/evo-compare`
                                        # (pure JSON math, no script)
```

## References

- NVIDIA AVO paper: arXiv:2603.24517
- This kit shipped by: the `evo-loop` Claude skill (`scripts/evo-init`),
  or `vocoder --evo-init`.
