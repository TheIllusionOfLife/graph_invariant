# Harmony Metric Calibration Gate — ✅ PASSED

Pre-registered gate: ≥10% improvement over frequency baseline on ≥2 domains,
bootstrap 95% CI lower bound > frequency mean on both domains,
and all 6 weight configurations show consistent direction.

## Datasets

| Domain | Entities | Edges |
|--------|----------|-------|
| linear_algebra | 50 | 75 |
| periodic_table | 153 | 326 |

## Baselines

| Domain | Random | Frequency | DistMult alone |
|--------|--------|-----------|---------------|
| linear_algebra | 0.2667 | 0.4667 | 0.3333 |
| periodic_table | 0.1385 | 0.3231 | 0.2000 |

## Gate Check (n_bootstrap=200, seed=42)

| Domain | Harmony mean | CI95 half-width | CI lower | Freq mean | Improvement | CI > Freq | Pass? |
|--------|-------------|-----------------|----------|-----------|-------------|-----------|-------|
| linear_algebra | 0.6112 | 0.0081 | 0.6032 | 0.4667 | 31.0% ✓ | ✓ | ✅ |
| periodic_table | 0.5335 | 0.0021 | 0.5314 | 0.3231 | 65.1% ✓ | ✓ | ✅ |

## Ablation Results

### linear_algebra

| Component | Mean | Std | CI95 half-width | Δ vs full |
|-----------|------|-----|-----------------|-----------|
| full | 0.6112 | 0.0581 | 0.0081 | +0.0000 |
| w/o_comp | 0.6516 | 0.0757 | 0.0105 | +0.0404 |
| w/o_coh | 0.5192 | 0.0557 | 0.0077 | -0.0921 |
| w/o_sym | 0.6684 | 0.0679 | 0.0094 | +0.0571 |
| w/o_gen | 0.5848 | 0.0652 | 0.0090 | -0.0265 |

### periodic_table

| Component | Mean | Std | CI95 half-width | Δ vs full |
|-----------|------|-----|-----------------|-----------|
| full | 0.5335 | 0.0152 | 0.0021 | +0.0000 |
| w/o_comp | 0.5278 | 0.0215 | 0.0030 | -0.0057 |
| w/o_coh | 0.3768 | 0.0233 | 0.0032 | -0.1567 |
| w/o_sym | 0.7091 | 0.0225 | 0.0031 | +0.1756 |
| w/o_gen | 0.5173 | 0.0017 | 0.0002 | -0.0162 |

## Weight Grid Consistency

All 6 configurations beat frequency baseline: ✅

### linear_algebra

| α | β | γ | δ | Harmony | Frequency | Improvement | Beats freq? |
|---|---|---|---|---------|-----------|-------------|-------------|
| 0.3 | 0.1 | 0.25 | 0.25 | 0.4791 | 0.4667 | 2.7% | ✓ |
| 0.3 | 0.3 | 0.25 | 0.25 | 0.5466 | 0.4667 | 17.1% | ✓ |
| 0.5 | 0.1 | 0.25 | 0.25 | 0.4836 | 0.4667 | 3.6% | ✓ |
| 0.5 | 0.3 | 0.25 | 0.25 | 0.5400 | 0.4667 | 15.7% | ✓ |
| 0.7 | 0.1 | 0.25 | 0.25 | 0.4867 | 0.4667 | 4.3% | ✓ |
| 0.7 | 0.3 | 0.25 | 0.25 | 0.5352 | 0.4667 | 14.7% | ✓ |

### periodic_table

| α | β | γ | δ | Harmony | Frequency | Improvement | Beats freq? |
|---|---|---|---|---------|-----------|-------------|-------------|
| 0.3 | 0.1 | 0.25 | 0.25 | 0.3550 | 0.3231 | 9.9% | ✓ |
| 0.3 | 0.3 | 0.25 | 0.25 | 0.4723 | 0.3231 | 46.2% | ✓ |
| 0.5 | 0.1 | 0.25 | 0.25 | 0.3932 | 0.3231 | 21.7% | ✓ |
| 0.5 | 0.3 | 0.25 | 0.25 | 0.4865 | 0.3231 | 50.6% | ✓ |
| 0.7 | 0.1 | 0.25 | 0.25 | 0.4196 | 0.3231 | 29.9% | ✓ |
| 0.7 | 0.3 | 0.25 | 0.25 | 0.4970 | 0.3231 | 53.8% | ✓ |

## Final Verdict

**✅ PASSED**
