#!/usr/bin/env bash
# evo-candidate.sh — AVO candidate isolation via git worktrees (P1.1).
#
# Evolutionary candidates mutate in THROWAWAY worktrees so a broken attempt
# never dirties the main tree ("fix the build in the same session" stops
# applying to rejected candidates). Only gate-passing, improving diffs are
# merged back by the supervisor (/evo run, P1.2).
#
# Worktrees live under .vocoder/evo/worktrees/ (gitignored scratch state —
# `git check-ignore .vocoder/evo/worktrees/x` is true in this repo).
#
# Usage:
#   .scripts/evo-candidate.sh create <name> [ref]   # worktree at ref (default HEAD)
#   .scripts/evo-candidate.sh score  <name>         # evo-score.sh in the worktree
#   .scripts/evo-candidate.sh gate   <name>         # evo-score.sh --gate there
#   .scripts/evo-candidate.sh diff   <name>         # git diff of worktree vs its base
#   .scripts/evo-candidate.sh discard <name>        # remove worktree (force) + prune
#   .scripts/evo-candidate.sh list                  # list candidate worktrees
#
# Part of the portable AVO kit shipped by `vocoder --evo-init` (consumed by
# `/evo run` / `/evo finish`).
set -euo pipefail
cd "$(dirname "$0")/.."

WT_ROOT=".vocoder/evo/worktrees"

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

require_name() {
    [[ -n "${1:-}" ]] || { echo "error: <name> required" >&2; usage; }
}

wt_path() { echo "$WT_ROOT/$1"; }

cmd="${1:-}"; shift || true
case "$cmd" in
    create)
        name="${1:-}"; require_name "$name"
        ref="${2:-HEAD}"
        dest="$(wt_path "$name")"
        if [[ -e "$dest" ]]; then
            echo "error: candidate '$name' already exists at $dest" >&2
            exit 1
        fi
        mkdir -p "$WT_ROOT"
        git worktree add "$dest" "$ref" >&2
        echo "$dest"
        ;;
    score)
        name="${1:-}"; require_name "$name"
        dest="$(wt_path "$name")"
        [[ -d "$dest" ]] || { echo "error: no candidate '$name'" >&2; exit 1; }
        ( cd "$dest" && ./.scripts/evo-score.sh )
        ;;
    gate)
        name="${1:-}"; require_name "$name"
        dest="$(wt_path "$name")"
        [[ -d "$dest" ]] || { echo "error: no candidate '$name'" >&2; exit 1; }
        ( cd "$dest" && ./.scripts/evo-score.sh --gate )
        ;;
    diff)
        name="${1:-}"; require_name "$name"
        dest="$(wt_path "$name")"
        [[ -d "$dest" ]] || { echo "error: no candidate '$name'" >&2; exit 1; }
        ( cd "$dest" && git diff HEAD )
        ;;
    discard)
        name="${1:-}"; require_name "$name"
        dest="$(wt_path "$name")"
        if [[ -d "$dest" ]]; then
            git worktree remove --force "$dest" 2>/dev/null || rm -rf "$dest"
        fi
        git worktree prune
        echo "discarded $name"
        ;;
    list)
        git worktree list | grep "$WT_ROOT" || echo "(no candidates)"
        ;;
    *)
        usage
        ;;
esac