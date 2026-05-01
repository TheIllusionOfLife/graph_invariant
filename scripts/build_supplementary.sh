#!/usr/bin/env bash
# Build supplementary.zip for NeurIPS 2026 submission.
#
# Includes:
#   - src/, tests/, scripts/             (Harmony implementation + automation)
#   - analysis/calibration_gate.md       (only Harmony-relevant analysis md)
#   - data/results/, data/results_mlx/   (multi-seed and MLX run summaries)
#   - pyproject.toml, README.md          (project metadata + reviewer entry point)
#
# Excludes:
#   - paper/, docs/, NeurIPS*/           (manuscript + dev-facing docs)
#   - .git*, __pycache__, .DS_Store, __MACOSX, .ipynb_checkpoints (cruft)
#   - .zenodo.json, zenodo_staging/      (deanonymizing artifacts)
#   - analysis/full_experiment_analysis.md, analysis/phase1_v2_analysis.md
#   - analysis/results/, analysis/day1_results/                   (legacy graph_invariant reports)
#
# After building, the script audits the archive for:
#   - structural leaks (paper/, .git, zenodo, __MACOSX, .DS_Store)
#   - identity-string leaks (author username, /Users/ paths, Zenodo DOI)
# and exits non-zero if any leak is found.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT=supplementary.zip
rm -f "$OUT"

zip -r "$OUT" \
    src tests analysis scripts data/results data/results_mlx \
    pyproject.toml README.md \
    -x '*/__pycache__/*' '*.pyc' '*.DS_Store' '*/.ipynb_checkpoints/*' \
       '*.git*' 'paper/*' '*/NeurIPS*/*' '.zenodo.json' 'zenodo_staging/*' \
       '__MACOSX/*' '*/__MACOSX/*' \
       'analysis/full_experiment_analysis.md' \
       'analysis/phase1_v2_analysis.md' \
       'analysis/results/*' \
       'analysis/day1_results/*' \
    > /dev/null

echo "=== Verifying $OUT ==="
unzip -l "$OUT" | tail -3

# Structural leak check.
# Match only the actually-deanonymizing artifacts:
#   - .zenodo.json (deposit metadata with author identity)
#   - zenodo_staging/ (per-release staging directory)
# Generic scripts/*zenodo*.py are kept intentionally — they are parameterised
# upload utilities with no embedded identity.
LEAK_PATTERN='\.zenodo\.json|zenodo_staging/|\.git|paper/|__MACOSX|\.DS_Store'
if unzip -l "$OUT" | awk '{print $NF}' | grep -E "$LEAK_PATTERN" >/dev/null; then
    echo "ERROR: structural leak detected in $OUT" >&2
    unzip -l "$OUT" | awk '{print $NF}' | grep -E "$LEAK_PATTERN" >&2
    exit 1
fi

# Identity-string leak check (binary scan of the archive contents).
if strings "$OUT" | grep -Ei "yuyamukai|mukaiyuya|/Users/yuyamukai|10\.5281/zenodo" >/dev/null; then
    echo "ERROR: identity-string leak detected in $OUT" >&2
    strings "$OUT" | grep -Ei "yuyamukai|mukaiyuya|/Users/yuyamukai|10\.5281/zenodo" | head -5 >&2
    exit 1
fi

# Confirm only Harmony-relevant analysis md files ship.
md_in_zip=$(unzip -l "$OUT" | awk '/analysis\/.*\.md$/ {print $4}' | sort)
expected="analysis/calibration_gate.md"
if [ "$md_in_zip" != "$expected" ]; then
    echo "ERROR: unexpected analysis/*.md files in $OUT" >&2
    echo "Expected: $expected" >&2
    echo "Found:    $md_in_zip" >&2
    exit 1
fi

echo "OK: $OUT is reviewer-safe."
