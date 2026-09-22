#!/usr/bin/env bash
# evo-kickoff.sh — seed an AVO evolution scope for a repo/crate/feature.
#
# Usage:
#   .scripts/evo-kickoff.sh <scope> [description]
#
# Creates (or extends) .vocoder/evo/population-<scope>.json with an x0
# baseline entry and prints the next steps. Run from the repo root.
# The scoring contract lives in docs/EVO.md; scores are computed by
# .scripts/evo-score.sh so all versions stay comparable.
set -euo pipefail
cd "$(dirname "$0")/.."

scope="${1:-}"
desc="${2:-}"
if [[ -z "$scope" ]]; then
    echo "usage: .scripts/evo-kickoff.sh <scope> [description]" >&2
    echo "e.g.:  .scripts/evo-kickoff.sh parser 'streaming preprocessor hot path'" >&2
    exit 1
fi
# Slug: lowercase alnum + dash.
scope="$(echo "$scope" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/-/g; s/--*/-/g; s/^-//; s/-$//')"

pop=".vocoder/evo/population-${scope}.json"
mkdir -p .vocoder/evo
if [[ -f "$pop" ]]; then
    echo "lineage already exists: $pop (resume with /evo, VOCODER_EVO_POPULATION=$pop)" >&2
    exit 1
fi

commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
cat > "$pop" <<EOF
{
  "schema": "avo-population/v1",
  "lineage": "${scope}${desc:+ — $desc}",
  "scoring": {
    "gate": "see docs/EVO.md — the correctness gate for this repo's toolchain",
    "f": "see docs/EVO.md — scope picks its primary score (bench_geomean_ns / warnings / test_count / docs metric)"
  },
  "x0": {
    "name": "x0",
    "commit": "$commit",
    "phase": "baseline",
    "scores": {},
    "gate": "pending",
    "note": "seeded by evo-kickoff.sh; fill scores with .scripts/evo-score.sh"
  },
  "committed": []
}
EOF

echo "Seeded $pop (x0 @ $commit)."
echo
echo "Kickoff checklist (docs/EVO.md is the contract):"
echo "  1. Define the scope's primary score (one number to drive down/up)."
echo "  2. Baseline:  .scripts/evo-score.sh          # after benches if perf scope"
echo "     then paste the tuple into x0.scores in $pop."
echo "  3. Round loop (the AVO operator):"
echo "       a. consult K (AGENTS.md, docs/, profiler output, lineage)"
echo "       b. implement ONE candidate edit"
echo "       c. gate:   .scripts/evo-score.sh --gate   (pass => proceed)"
echo "       d. score:  .scripts/evo-score.sh"
echo "       e. commit ONLY if score matches/beats best; append to committed[]"
echo "       f. persist round findings to memory (project target, topic evo-<scope>)"
echo "          — a 3-line note: measured delta, mechanism, gotcha."
echo "  4. Two non-improving candidates => steering note before round 3."
echo "  5. View lineage: set VOCODER_EVO_POPULATION=$pop"
echo "     then, if this repo runs the vocoder TUI: /evo"
echo "     (otherwise just read $pop directly — it is the replayable lineage)."
