#!/usr/bin/env bash
# Build supplementary.zip for NeurIPS 2026 submission.
#
# Workflow:
#   1. Stage tracked files (via 'git ls-files') to a temp directory, applying
#      explicit per-file/per-glob excludes for off-topic content.
#   2. Anonymise LICENSE in the staged copy (source tree untouched).
#   3. Audit the staged tree for author-identifying strings using
#      'rg --hidden --no-ignore' against scripts/.leak_patterns.
#   4. Zip from the staged tree (so the audit and archive cover the same files).
#   5. Verify the archive structurally and confirm the expected post-conditions.
#
# Includes (from git-tracked content):
#   - src/, tests/, scripts/                          (Harmony implementation)
#   - analysis/calibration_gate.md                    (only Harmony analysis md)
#   - data/results/, data/results_mlx/                (run summaries)
#   - pyproject.toml, README.md, LICENSE              (project metadata)
#
# Excludes:
#   - paper/, docs/, NeurIPS*/                        (manuscript + dev docs;
#                                                      not staged at all because
#                                                      they aren't in the include
#                                                      list passed to git ls-files)
#   - scripts/build_supplementary.sh, scripts/.leak_patterns
#                                                     (this script + its
#                                                      identity patterns; would
#                                                      themselves leak if shipped)
#   - analysis/full_experiment_analysis.md, analysis/phase1_v2_analysis.md
#   - analysis/results/, analysis/day1_results/       (legacy graph_invariant
#                                                      cross-experiment reports)
#   - .zenodo.json, zenodo_staging/                   (deanonymising artifacts)
#
# Untracked files (e.g., __pycache__, .pyc, local notebooks) are skipped by
# virtue of using 'git ls-files' rather than 'find' or 'zip -r' over the live
# source tree.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT=supplementary.zip
STAGE=$(mktemp -d)
trap "rm -rf '$STAGE'" EXIT

PATTERN_FILE="$(pwd)/scripts/.leak_patterns"
if [[ ! -f "$PATTERN_FILE" ]]; then
    echo "ERROR: $PATTERN_FILE missing — cannot run identity audit" >&2
    exit 1
fi

# Stage tracked content. Path arguments to git ls-files act as a positive
# filter; the per-line case statement applies negative filters.
TRACKED=$(git ls-files \
    src tests analysis scripts \
    data/results data/results_mlx \
    pyproject.toml README.md LICENSE)

count=0
while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    case "$f" in
        scripts/build_supplementary.sh|scripts/.leak_patterns) continue ;;
        analysis/full_experiment_analysis.md|analysis/phase1_v2_analysis.md) continue ;;
        analysis/results/*|analysis/day1_results/*) continue ;;
        .zenodo.json|zenodo_staging/*) continue ;;
    esac
    mkdir -p "$STAGE/$(dirname "$f")"
    cp "$f" "$STAGE/$f"
    count=$((count + 1))
done <<< "$TRACKED"
echo "Staged $count files into temp directory."

# Anonymise LICENSE in the staged copy. The source-tree LICENSE keeps the real
# author name (legally required); only the bundled copy is anonymised.
if grep -q "Yuya Mukai" "$STAGE/LICENSE"; then
    sed -i.bak -E 's/(Copyright \(c\) [0-9]+) Yuya Mukai/\1 Anonymous Author(s)/' "$STAGE/LICENSE"
    rm "$STAGE/LICENSE.bak"
fi

# Pre-zip identity audit. --hidden --no-ignore -a means: scan dotfiles, scan
# files that would be gitignored, treat every file as text. This catches
# identity strings in places that vanilla rg would skip.
echo "=== Pre-zip identity scan ==="
SCAN_HITS=$(rg -in --hidden --no-ignore -a -f "$PATTERN_FILE" "$STAGE" 2>/dev/null || true)
if [[ -n "$SCAN_HITS" ]]; then
    echo "ERROR: identity-string leak detected in staged tree:" >&2
    echo "$SCAN_HITS" >&2
    exit 1
fi
echo "Staged tree is identity-clean."

# Build the archive from the staged directory.
rm -f "$OUT"
(cd "$STAGE" && zip -r "$(cd - >/dev/null && pwd)/$OUT" . > /dev/null)

echo
echo "=== Verifying $OUT ==="
unzip -l "$OUT" | tail -3

# Structural leak check on archive listing.
LEAK_PATTERN='^\.zenodo\.json$|^zenodo_staging/|^paper/|/__MACOSX/|^__MACOSX/|/\.DS_Store$|^\.DS_Store$|/\.git'
if unzip -Z1 "$OUT" | grep -E "$LEAK_PATTERN" >/dev/null; then
    echo "ERROR: structural leak detected in $OUT" >&2
    unzip -Z1 "$OUT" | grep -E "$LEAK_PATTERN" >&2
    exit 1
fi

# Only Harmony-relevant analysis md should ship.
md_in_zip=$(unzip -Z1 "$OUT" | awk -F/ '$1=="analysis" && /\.md$/' | sort)
expected="analysis/calibration_gate.md"
if [[ "$md_in_zip" != "$expected" ]]; then
    echo "ERROR: unexpected analysis/*.md files in $OUT" >&2
    echo "Expected: $expected" >&2
    echo "Found:    $md_in_zip" >&2
    exit 1
fi

# Build script and pattern file must not be bundled.
if unzip -Z1 "$OUT" | grep -qE '^scripts/build_supplementary\.sh$|^scripts/\.leak_patterns$'; then
    echo "ERROR: build script or pattern file leaked into $OUT" >&2
    exit 1
fi

# LICENSE in the archive must be anonymised (no author name).
if unzip -p "$OUT" LICENSE | grep -q "Yuya Mukai"; then
    echo "ERROR: LICENSE in $OUT still contains author name" >&2
    exit 1
fi

echo "OK: $OUT is reviewer-safe."
