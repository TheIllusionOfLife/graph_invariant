#!/bin/bash
# Run all experiments sequentially + OOD validation on results
#
# Quick run (default):  bash run_all_experiments.sh
# Full run:             PROFILE=full bash run_all_experiments.sh
#
# Override model:       MODEL=gemma3:4b bash run_all_experiments.sh
# Override generations: MAX_GENS=10 bash run_all_experiments.sh
#
# Logs: experiment_log.txt (via nohup) or stdout (via tee)

set -euo pipefail

CLI="uv run python -m graph_invariant.cli"
PROFILE="${PROFILE:-quick}"

echo "============================================================"
echo " Graph Invariant — Experiment Suite (${PROFILE})"
echo " Started: $(date)"
echo " Branch:  $(git branch --show-current)"
echo "============================================================"
echo ""

# Select config prefix based on profile
if [ "$PROFILE" = "full" ]; then
    MAP_ELITES_CFG="configs/experiment_map_elites_aspl.json"
    ALG_CONN_CFG="configs/experiment_algebraic_connectivity.json"
    UPPER_BOUND_CFG="configs/experiment_upper_bound_aspl.json"
    BENCHMARK_CFG="configs/benchmark_aspl.json"
else
    MAP_ELITES_CFG="configs/quick_map_elites_aspl.json"
    ALG_CONN_CFG="configs/quick_algebraic_connectivity.json"
    UPPER_BOUND_CFG="configs/quick_upper_bound_aspl.json"
    BENCHMARK_CFG="configs/quick_benchmark_aspl.json"
fi

# Track which experiments succeed for OOD validation
SUCCEEDED_DIRS=()

# ── 1. MAP-Elites ASPL ──────────────────────────────────────────────
echo "━━━ [1/4] MAP-Elites ASPL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Config: $MAP_ELITES_CFG"
echo "Start:  $(date)"
if $CLI phase1 --config "$MAP_ELITES_CFG"; then
    dir=$(python3 -c "import json; print(json.load(open('$MAP_ELITES_CFG'))['artifacts_dir'])")
    echo "✓ MAP-Elites ASPL completed at $(date)"
    SUCCEEDED_DIRS+=("$dir")
else
    echo "✗ MAP-Elites ASPL failed at $(date)"
fi
echo ""

# ── 2. Algebraic Connectivity ───────────────────────────────────────
echo "━━━ [2/4] Algebraic Connectivity ━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Config: $ALG_CONN_CFG"
echo "Start:  $(date)"
if $CLI phase1 --config "$ALG_CONN_CFG"; then
    dir=$(python3 -c "import json; print(json.load(open('$ALG_CONN_CFG'))['artifacts_dir'])")
    echo "✓ Algebraic Connectivity completed at $(date)"
    SUCCEEDED_DIRS+=("$dir")
else
    echo "✗ Algebraic Connectivity failed at $(date)"
fi
echo ""

# ── 3. Upper Bound ASPL ─────────────────────────────────────────────
echo "━━━ [3/4] Upper Bound ASPL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Config: $UPPER_BOUND_CFG"
echo "Start:  $(date)"
if $CLI phase1 --config "$UPPER_BOUND_CFG"; then
    dir=$(python3 -c "import json; print(json.load(open('$UPPER_BOUND_CFG'))['artifacts_dir'])")
    echo "✓ Upper Bound ASPL completed at $(date)"
    SUCCEEDED_DIRS+=("$dir")
else
    echo "✗ Upper Bound ASPL failed at $(date)"
fi
echo ""

# ── 4. Benchmark ASPL ───────────────────────────────────────────────
echo "━━━ [4/4] Benchmark ASPL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Config: $BENCHMARK_CFG"
echo "Start:  $(date)"
if $CLI benchmark --config "$BENCHMARK_CFG"; then
    echo "✓ Benchmark ASPL completed at $(date)"
else
    echo "✗ Benchmark ASPL failed at $(date)"
fi
echo ""

# ── 5. OOD Validation on all succeeded experiments ──────────────────
echo "━━━ [5/5] OOD Validation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Start:  $(date)"

for dir in "${SUCCEEDED_DIRS[@]}"; do
    summary="$dir/phase1_summary.json"
    if [ -f "$summary" ]; then
        echo "--- OOD validating: $dir ---"
        ood_output="$dir/ood"
        if $CLI ood-validate --summary "$summary" --output "$ood_output"; then
            echo "✓ OOD validation for $dir completed"
        else
            echo "✗ OOD validation for $dir failed"
        fi
    else
        echo "⚠ No phase1_summary.json in $dir, skipping OOD"
    fi
done

# Also OOD-validate the existing experiment_v2 results
if [ -f "artifacts/experiment_v2/phase1_summary.json" ]; then
    echo "--- OOD validating: artifacts/experiment_v2 (existing) ---"
    if $CLI ood-validate --summary "artifacts/experiment_v2/phase1_summary.json" --output "artifacts/experiment_v2/ood"; then
        echo "✓ OOD validation for experiment_v2 completed"
    else
        echo "✗ OOD validation for experiment_v2 failed"
    fi
fi

echo ""

# ── 6. Generate reports ─────────────────────────────────────────────
echo "━━━ Reports ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for dir in "${SUCCEEDED_DIRS[@]}" "artifacts/experiment_v2"; do
    if [ -f "$dir/phase1_summary.json" ]; then
        echo "--- Report: $dir ---"
        $CLI report --artifacts "$dir" || echo "⚠ Report generation failed for $dir"
    fi
done

echo ""
echo "============================================================"
echo " All experiments finished at $(date)"
echo "============================================================"
