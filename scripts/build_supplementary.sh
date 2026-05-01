#!/usr/bin/env bash
# Build supplementary.zip for NeurIPS 2026 submission.
#
# Includes:
#   - src/, tests/, scripts/             (Harmony implementation + automation)
#   - analysis/calibration_gate.md       (only Harmony-relevant analysis md)
#   - data/results/, data/results_mlx/   (multi-seed and MLX run summaries)
#   - pyproject.toml, README.md, LICENSE (project metadata + reviewer entry point)
#
# Excludes:
#   - paper/, docs/, NeurIPS*/                       (manuscript + dev-facing docs)
#   - .git*, __pycache__, .DS_Store, __MACOSX        (cruft)
#   - .ipynb_checkpoints/                            (cruft)
#   - .zenodo.json, zenodo_staging/                  (deanonymizing artifacts)
#   - scripts/build_supplementary.sh                 (THIS script — contains leak patterns
#                                                     that include the author username
#                                                     verbatim; must not ship.)
#   - analysis/full_experiment_analysis.md, analysis/phase1_v2_analysis.md
#   - analysis/results/, analysis/day1_results/      (legacy graph_invariant reports)
#
# Audits the source tree (BEFORE zipping, on the uncompressed files) for any
# author-identifying strings, then audits the built archive listing for
# structural leaks. Exits non-zero on any leak.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT=supplementary.zip
rm -f "$OUT"

# Files/dirs that will be staged into the archive.
STAGED_PATHS=(src tests analysis scripts data/results data/results_mlx
              pyproject.toml README.md LICENSE)

# Identity-string patterns. We never write the literal author username in this
# file; instead we read patterns from an external file that is itself excluded
# from the archive. This keeps the script's source tree clean of identity
# literals AND makes the patterns easy to audit.
PATTERN_FILE="$(dirname "$0")/.leak_patterns"
if [[ ! -f "$PATTERN_FILE" ]]; then
    echo "ERROR: $PATTERN_FILE missing — cannot run identity audit" >&2
    exit 1
fi

# Pre-zip identity audit: scan the uncompressed source tree. ZIP compression
# would defeat a post-build `strings` scan, so we must check before packaging.
# Exclude this script itself (it imports the patterns transitively) and the
# pattern file from the scan.
echo "=== Pre-zip identity scan ==="
SCAN_HITS=$(rg -in --no-messages -f "$PATTERN_FILE" \
    --glob '!scripts/build_supplementary.sh' \
    --glob '!scripts/.leak_patterns' \
    "${STAGED_PATHS[@]}" 2>/dev/null || true)
if [[ -n "$SCAN_HITS" ]]; then
    echo "ERROR: identity-string leak detected in source tree:" >&2
    echo "$SCAN_HITS" >&2
    exit 1
fi
echo "Source tree is identity-clean."

# Build the archive.
zip -r "$OUT" "${STAGED_PATHS[@]}" \
    -x '*/__pycache__/*' '*.pyc' '*.DS_Store' '*/.ipynb_checkpoints/*' \
       '*.git*' 'paper/*' '*/NeurIPS*/*' '.zenodo.json' 'zenodo_staging/*' \
       '__MACOSX/*' '*/__MACOSX/*' \
       'scripts/build_supplementary.sh' \
       'scripts/.leak_patterns' \
       'analysis/full_experiment_analysis.md' \
       'analysis/phase1_v2_analysis.md' \
       'analysis/results/*' \
       'analysis/day1_results/*' \
    > /dev/null

echo
echo "=== Verifying $OUT ==="
unzip -l "$OUT" | tail -3

# Structural leak check on archive listing. Use `unzip -Z1` for a clean
# one-path-per-line listing (no header/footer, handles spaces).
# Patterns are anchored where appropriate to avoid matching `wallpaper/` etc.
LEAK_PATTERN='^\.zenodo\.json$|^zenodo_staging/|^paper/|/__MACOSX/|^__MACOSX/|/\.DS_Store$|^\.DS_Store$|/\.git'
if unzip -Z1 "$OUT" | grep -E "$LEAK_PATTERN" >/dev/null; then
    echo "ERROR: structural leak detected in $OUT" >&2
    unzip -Z1 "$OUT" | grep -E "$LEAK_PATTERN" >&2
    exit 1
fi

# Confirm only Harmony-relevant analysis md files ship.
md_in_zip=$(unzip -Z1 "$OUT" | awk -F/ '$1=="analysis" && /\.md$/' | sort)
expected="analysis/calibration_gate.md"
if [[ "$md_in_zip" != "$expected" ]]; then
    echo "ERROR: unexpected analysis/*.md files in $OUT" >&2
    echo "Expected: $expected" >&2
    echo "Found:    $md_in_zip" >&2
    exit 1
fi

# Confirm the build script itself was not bundled.
if unzip -Z1 "$OUT" | grep -q '^scripts/build_supplementary\.sh$\|^scripts/\.leak_patterns$'; then
    echo "ERROR: build script or pattern file leaked into $OUT" >&2
    exit 1
fi

echo "OK: $OUT is reviewer-safe."
