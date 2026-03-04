# MLX Long-Rerun Estimate (2026-03-04)

## Scope
Estimate wall-clock time for the long rerun set after switching to Qwen 3.5 via MLX, assuming:
- Harmony discovery reruns for 5 domains
- 10-seed evaluation refresh
- Block-A recomputation (multi-seed tables, stats, backtesting, downstream/frequency, Wikidata ablation, reproducibility table)
- Single machine, same class of hardware documented in paper

## Evidence Anchors
- Factor decomposition runs (3 configs x 5 domains) with `mlx-community/Qwen3.5-35B-A3B-4bit` via `mlx_lm` are documented as ~2.5 hours total (`paper/sections/appendix.tex`).
- The same appendix reports MLX throughput near ~46 tok/s (`paper/sections/appendix.tex`).
- Block-A composition and defaults are documented in `scripts/run_block_a.py` and `scripts/run_multi_seed.py` (default 10 seeds).

Derived unit estimate from the factor decomposition anchor:
- 2.5h / 15 runs ~= 10 minutes per domain-run at this scale.

## Revised Estimate (MLX Pipeline)
For 5 domains x 10 seeds = 50 domain-runs:
- Core reruns: ~50 x 10 min ~= 500 min ~= 8.3 h
- Block-A postprocessing: ~40-50 min (including stats)
- Analysis + figure + paper build: ~15-30 min

Estimated wall-clock:
- Best case: 8-10 h
- Expected: 10-14 h
- Worst case: 18-30 h (retry-heavy generations, intermittent hangs, manual recovery)

## Notes on Uncertainty
- Retry loops and proposal rejection dynamics can dominate tail latency.
- If domain count increases from 5 to 7, scale core rerun time by ~1.4x.
- If running with parallel workers >1, wall-clock may drop but contention can reduce per-worker throughput.

## Pre-run Checkpointing Recommendation
Before the full long rerun, run a short calibration slice (1 domain x 2 seeds) and compute observed min/domain-run.
Then rescale:

`Total hours ~= observed_min_per_domain_run * (num_domains * num_seeds) / 60 + overhead_hours`

This gives a same-day correction before committing the full overnight budget.
