#!/usr/bin/env bash
# evo-score.sh — compute the AVO score tuple for the CURRENT working tree.
#
# The tuple rides in .vocoder/evo/population.json alongside each committed
# version; both agents and humans call this so scores stay comparable.
#
# Scores (all "lower-is-better" except test_count):
#   bench_geomean_ns   — geometric mean of criterion estimates under
#                        target/criterion/*/new/estimates.json, EXCLUDING
#                        canary_* benches (run `cargo bench` first; absent
#                        when no estimates exist)
#   canary_<name>_ns   — each canary_* bench as its own veto metric (P1.3):
#                        a candidate regressing one beyond its noise band is
#                        rejected even if the primary improves
#   <bench path>       — every primary bench ALSO as its own metric (full
#                        criterion path, e.g. "engine/flush"), band-guarded
#                        individually so a roll-up delta is pinpointable
# Bench values are native NANOSECONDS (3 decimals => ps resolution): the
# pipeline works at ns/ps scale end-to-end, never rounding fast paths to 0.
#   warnings           — cargo clippy --workspace warning count (target: 0)
#   test_count         — passed tests from `cargo nextest run` summary
#                        (absent unless nextest is installed)
#
# Usage:
#   .scripts/evo-score.sh          # print JSON tuple to stdout
#   .scripts/evo-score.sh --gate   # also run the correctness gate first
#                                  # (build + nextest + clippy -D warnings);
#                                  # exit 1 if the gate fails
#   .scripts/evo-score.sh --noise N [bench_filter]
#   .scripts/evo-score.sh --noise-merge OUT CAL1 CAL2 ...
#                                  # combine calibrations; worst band wins
#                                  # run `cargo bench` N times, emit a
#                                  # "noise" object: per-metric CV% + the
#                                  # improvement band (mean +/- 2*sigma).
#                                  # Rule 2 sharpening: a delta counts as an
#                                  # improvement only beyond its band.
#
# This is the `rust` scorer template shipped by `vocoder --evo-init`.
set -euo pipefail
cd "$(dirname "$0")/.."

gate="skip"
noise_runs=""
bench_filter=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --noise)
            noise_runs="${2:-}"
            shift 2
            ;;
        --gate)
            gate="pass"
            shift
            ;;
        --noise-merge)
            # Defer to the merge block after the loop; stash the args.
            merge_out="${2:-}"
            shift 2
            merge_cals=()
            while [[ $# -gt 0 ]]; do
                merge_cals+=("$1")
                shift
            done
            ;;
        *)
            bench_filter="$1"
            shift
            ;;
    esac
done

if [[ -n "${merge_out:-}" ]]; then
    # Merge >=2 --noise calibrations into ONE noise.json taking the WORST
    # (max) band per metric — "beyond demonstrated noise" across every
    # calibration run, not just the luckiest one.
    if [[ ${#merge_cals[@]} -lt 1 ]]; then
        echo "usage: $0 --noise-merge OUT CAL1 CAL2 [...]" >&2
        exit 1
    fi
    python3 - "$merge_out" "${merge_cals[@]}" <<'PYEOF'
import json, sys

out_path, *paths = sys.argv[1:]
merged = {"noise_floor_version": 1, "calibrations": len(paths), "metrics": {}}
for pth in paths:
    with open(pth) as f:
        d = json.load(f)
    for name, m in d.get("metrics", {}).items():
        dst = merged["metrics"].setdefault(name, m.copy())
        if m["improve_band_pct"] > dst["improve_band_pct"]:
            dst["improve_band_pct"] = m["improve_band_pct"]
        if m["range_pct"] > dst.get("range_pct", 0):
            dst["range_pct"] = m["range_pct"]

with open(out_path, "w") as f:
    json.dump(merged, f, indent=2)
print(f"merged {len(paths)} calibrations -> {out_path}")
for name, m in sorted(merged["metrics"].items()):
    print(f"  {name:34} band=+/-{m['improve_band_pct']}%")
PYEOF
    exit 0
fi

if [[ "${noise_runs}" != "" ]]; then

    # --- noise-floor estimation ------------------------------------------
    # Run the benches N times; per benchmark, report CV% of the means and
    # the improvement band. Rule 2: only beyond-band deltas count.
    if ! [[ "$noise_runs" =~ ^[0-9]+$ ]] || (( noise_runs < 3 )); then
        echo "{\"error\": \"--noise requires an integer >= 3\"}" >&2
        exit 1
    fi
    python3 - "$noise_runs" "$bench_filter" <<'PY'
import json, math, pathlib, statistics, subprocess, sys

runs = int(sys.argv[1])
flt = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
crit = pathlib.Path("target/criterion")

# criterion OVERWRITES new/estimates.json on every run — snapshot the means
# after EACH run, or N runs leave exactly one sample per metric. A bench NOT
# selected by the filter never rewrites its file, so sampling is gated on a
# pre-run mtime watermark: only files WRITTEN BY THIS CALIBRATION count.
# Stale files from earlier invocations would otherwise fake cv=0 triplets.
import os as _os

watermark = 0.0
if crit.is_dir():
    for est in crit.rglob("new/estimates.json"):
        try:
            watermark = max(watermark, est.stat().st_mtime)
        except OSError:
            pass

def metric_name_for(rel):
    """Stable metric key for a criterion bench. Canary benches (leaf starts
    with `canary_`) are keyed `canary_<leaf>_ns` — the SAME key the scorer
    emits and evo-compare looks up, so canary noise bands wire end-to-end.
    Primary benches keep their full relative path (legacy-compatible)."""
    leaf = str(rel).split("/")[-1]
    if leaf.startswith("canary_"):
        return "canary_" + "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in leaf.removeprefix("canary_")
        ).lower() + "_ns"
    return str(rel)

def snapshot():
    snap = {}
    for est in crit.rglob("new/estimates.json"):
        if not (est.stat().st_mtime > watermark):
            continue  # not written by this calibration -> stale, skip
        rel = est.parent.parent.relative_to(crit)
        name = metric_name_for(rel)
        try:
            mean_ns = json.loads(est.read_text())["mean"]["point_estimate"]
        except (KeyError, json.JSONDecodeError):
            continue
        snap[name] = mean_ns
    return snap

samples = {}
for i in range(runs):
    cmd = ["cargo", "bench"]
    if flt:
        cmd.append(flt)
    subprocess.run(cmd, capture_output=True)
    for name, ns in snapshot().items():
        samples.setdefault(name, []).append(ns)

# Improvement band (EVO.md rule 2): a candidate delta counts ONLY beyond
#   band = max(SAFETY * observed_range%, 2*sqrt(2)*sigma/sqrt(n), 3%)
# - SAFETY * observed_range%: the nondeterminism THIS calibration actually
#   exhibited, inflated 1.25x — timing distributions are right-tailed and a
#   small-N range understates the tail (self-calibrating: bimodal benches
#   raise their own bar — the E7 lesson). Calibrate twice (`--noise` is
#   itself noisy) and merge with --noise-merge, keeping the worst band.
# - 2*sqrt(2)*sigma/sqrt(n): ~95% two-sided band on the difference of two
#   independent n-run means under H0;
# - 3% floor: guard against zero-variance luck.
out = {}
for name, xs in sorted(samples.items()):
    if len(xs) != runs or len(xs) < 3:
        continue
    mu = statistics.fmean(xs)
    sigma = statistics.pstdev(xs)
    cv = (sigma / mu * 100.0) if mu else 0.0
    SAFETY = 1.25
    range_pct = ((max(xs) - min(xs)) / mu * 100.0) if mu else 0.0
    stat_band = 2.0 * math.sqrt(2.0) * cv / math.sqrt(runs)
    band = max(SAFETY * range_pct, stat_band, 3.0)
    out[name] = {
        "runs": runs,
        "mean_ns": round(mu, 3),
        "cv_pct": round(cv, 2),
        "range_pct": round(range_pct, 2),
        "stat_band_pct": round(stat_band, 2),
        "improve_band_pct": round(band, 2),
        "samples_ns": [round(x, 3) for x in xs],
    }
print(json.dumps({"noise_floor_version": 1, "noise_runs": runs, "metrics": out}, indent=2))
PY
    exit 0
fi

if [[ "${gate:-skip}" == "pass" ]]; then
    gate="pass"
    # --all-features: feature-gated shells (the axum SSE pump, its tests) hold
    # real code paths the default feature set doesn't compile — a candidate
    # passing only on the default subset would silently change the code CI
    # runs. fmt is the same class of gate, cheaper to run than discover at CI.
    cargo build --workspace --all-features >/dev/null 2>&1 || { gate="fail: build"; }
    if [[ "$gate" == "pass" ]] && command -v cargo-fmt >/dev/null 2>&1; then
        cargo fmt --check >/dev/null 2>&1 || { gate="fail: fmt"; }
    fi
    if [[ "$gate" == "pass" ]] && command -v cargo-nextest >/dev/null 2>&1; then
        cargo nextest run --workspace --all-features >/dev/null 2>&1 || { gate="fail: nextest"; }
    fi
    if [[ "$gate" == "pass" ]]; then
        cargo clippy --workspace --all-targets --all-features -- -D warnings >/dev/null 2>&1 || {
            gate="fail: clippy"
        }
    fi
fi

# --- bench metrics (criterion estimates) ---------------------------------
# Primary geomean EXCLUDES canary_* benches (a sentinel that has its own
# veto metric must not also be averaged into the surface it guards). Each
# canary bench is emitted as its own `canary_<name>_ns` top-level metric so
# evo-compare's per-metric verdict rejects the candidate if it regresses
# beyond its own noise band — even when the primary geomean improved (P1.3).
bench_metrics="$(
    python3 - "$PWD" <<'PY'
import json, math, pathlib, sys

def metric_name_for(rel):
    """Stable metric key for a criterion bench. Canary benches (leaf starts
    with `canary_`) are keyed `canary_<leaf>_ns` -- the SAME key the scorer
    emits and evo-compare looks up, so canary noise bands wire end-to-end.
    Primary benches keep their full relative path (legacy-compatible)."""
    leaf = str(rel).split("/")[-1]
    if leaf.startswith("canary_"):
        return "canary_" + "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in leaf.removeprefix("canary_")
        ).lower() + "_ns"
    return str(rel)

root = pathlib.Path(sys.argv[1]) / "target/criterion"
pts = []          # primary bench point estimates (ns), for the geomean
primaries = {}    # primary bench metric key -> point_estimate (ns)
canaries = {}     # canary metric key -> point_estimate (ns)
if root.is_dir():
    for d in root.rglob("new/estimates.json"):
        try:
            ns = json.loads(d.read_text())["mean"]["point_estimate"]
        except (KeyError, json.JSONDecodeError):
            continue
        rel = d.parent.parent.relative_to(root)
        key = metric_name_for(rel)
        if key.startswith("canary_"):
            canaries[key] = ns
        else:
            primaries[key] = ns
            pts.append(ns)
out = []
if pts:
    out.append(f"\"bench_geomean_ns\": {math.exp(sum(map(math.log, pts)) / len(pts)):.3f}")
    # Every primary bench rides as its own band-guarded metric (granularity:
    # the roll-up alone cannot say WHICH bench moved). Keys are the full
    # criterion path; new benches join the verdict once both tuples have them.
    for key, ns in sorted(primaries.items()):
        out.append(f"\"{key}\": {ns:.3f}")
for key, ns in sorted(canaries.items()):
    out.append(f"\"{key}\": {ns:.3f}")
print(", ".join(out))
PY
)"

# --- clippy warning count -------------------------------------------------
warnings="$(cargo clippy --workspace --all-targets 2>&1 | grep -c '^warning' || true)"

# --- test count (nextest if available) -------------------------------------
tests=""
if command -v cargo-nextest >/dev/null 2>&1; then
    tests="$(
        cargo nextest run --workspace 2>&1 | grep -oE '[0-9]+ passed' | tail -1 |
            grep -oE '[0-9]+' || true
    )"
fi

json="\"gate\": \"$gate\""
json="$json, \"warnings\": ${warnings:-0}"
[[ -n "$bench_metrics" ]] && json="$json, $bench_metrics"
[[ -n "$tests" ]] && json="$json, \"test_count\": $tests"
echo "{$json}"
