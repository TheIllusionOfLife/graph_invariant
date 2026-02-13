# Phase 1 v2 Analysis: Island-Specific Prompts + Novelty Gate

## 1. Experiment Config Summary

Changes from v1 (experiment_05):
- **Island-specific prompt strategies**: REFINEMENT (islands 0, 2), COMBINATION (island 1), NOVEL (island 3)
- **Anti-pattern warnings**: All prompts now include FORBIDDEN block (no BFS/DFS, no single-feature return)
- **Formula examples**: All prompts include 3 example `def new_invariant(s)` formulas
- **Novelty gate**: threshold=0.15 rejects candidates trivially correlated with known invariants
- **Score weights**: alpha=0.5 (was 0.6), gamma=0.3 (was 0.2) — increased novelty emphasis
- **Baselines enabled**: `run_baselines=true` (was false)
- **Prompt/response logging**: enabled for post-hoc analysis

## 2. Results (Single-Seed Exploratory Run)

| Metric | Value |
|--------|-------|
| Seed | 42 |
| Stop reason | early_stop (gen 12 of 20) |
| Generations completed | 12 |
| Candidates evaluated (LLM calls) | 159 |
| Candidates rejected (events) | 117 |
| Acceptance rate | 26.4% (42/159) |
| Best val Spearman | **0.9370** |
| Best test Spearman | **0.9215** |
| Best train Spearman | 0.9221 |
| Sanity Spearman | 0.5000 |
| Best novelty bonus | 0.181 |
| Best total score | 0.549 |

### Per-Island Performance

| Island | Strategy | Temp | Evaluated | Best Spearman | Best Novelty | Best Total |
|--------|----------|------|-----------|---------------|--------------|------------|
| 0 | REFINEMENT | 0.3 | 52 | 0.725 | 0.320 | 0.508 |
| 1 | COMBINATION | 0.3 | 47 | 0.738 | 0.283 | 0.503 |
| 2 | REFINEMENT | 0.8 | 45 | **0.937** | 0.181 | **0.549** |
| 3 | NOVEL | 1.2 | 15 | -0.570 | 0.324 | 0.417 |

**Winner**: Island 2 (REFINEMENT at temp=0.8) — moderate temperature with refinement strategy.

## 3. Best Formula Discovered

```python
def new_invariant(s):
    n          = s.get('n', 0)
    avg_deg    = s.get('avg_degree', 0.0) or 1.0
    trans      = s.get('transitivity', 0.0) or 0.0
    avg_clust  = s.get('avg_clustering', 0.0) or 0.0
    std_deg    = s.get('std_degree', 0.0) or 1.0
    density    = s.get('density', 0.0) or 1e-6
    assort     = s.get('degree_assortativity', 0.0) or 0.0
    eps = 1e-6
    base = np.log(max(n, 2)) / np.log(max(avg_deg + 1.0, 2.0))
    clustering_adj = 1.0 + 0.5 * (trans + avg_clust)
    variance_adj = 1.0 / (1.0 + std_deg / (avg_deg + 1.0))
    density_adj = np.sqrt(1.0 / (density + eps))
    assort_adj = 1.0 + 0.5 * assort
    return base * clustering_adj * variance_adj * density_adj * assort_adj
```

> **Note on defensive patterns**: This formula is LLM-generated and preserved verbatim. The `or` guards (e.g., `s.get('avg_degree', 0.0) or 1.0`) are over-defensive — the `or` replaces falsy `0.0` with a non-zero default, which may mask real zero-valued features. The `eps` variable is redundant given the `density or 1e-6` guard. These quirks are retained for reproducibility; simplifying them could change output values.

### Mathematical Interpretation

The formula approximates average shortest path length as:

```text
ASPL ≈ [log(n) / log(k+1)] × [1 + 0.5(C_t + C_avg)] × [1/(1 + σ_k/(k+1))] × √(1/ρ) × [1 + 0.5r]
```

Where:
- **Base**: `log(n)/log(k+1)` — random graph diameter scaling (Chung-Lu model)
- **Clustering correction**: Higher clustering → longer paths (detours around triangles)
- **Degree variance correction**: Higher degree heterogeneity → shorter paths (hubs act as shortcuts)
- **Density correction**: Sparser graphs → longer paths (fewer route options)
- **Assortativity correction**: Positive assortativity → longer paths (high-degree nodes connect to each other, leaving low-degree periphery disconnected)

## 4. Baseline Comparison

| Method | Val Spearman | Test Spearman | Notes |
|--------|-------------|---------------|-------|
| **LLM (best)** | **0.937** | **0.922** | Single interpretable formula |
| Linear Regression | 0.975 | 0.975 | 9-feature linear model |
| Random Forest | skipped | skipped | scikit-learn not available in sandbox |
| PySR | skipped | skipped | Julia not installed |

Linear regression outperforms (0.975 vs 0.937) but uses all 9 features as a black-box linear combination. The LLM formula provides a single closed-form expression with interpretable physical meaning — this interpretability tradeoff is the research goal.

## 5. Novelty Assessment

### Novelty Gate Performance

| Rejection Reason | Count | % of Total |
|-----------------|-------|------------|
| below_novelty_threshold | 53 | 45.3% |
| no_valid_train_predictions | 40 | 34.2% |
| below_train_threshold | 24 | 20.5% |
| **Total rejections** | **117** | |

The novelty gate is the primary filter, rejecting 45% of all failed candidates.

### Correlation with Known Invariants (Validation Set)

| Known Invariant | |ρ| (point) | CI Upper (95%) |
|----------------|------------|----------------|
| density | 0.819 | 0.855 |
| diameter | 0.807 | 0.846 |
| algebraic_connectivity | 0.784 | 0.834 |
| spectral_radius | 0.626 | 0.699 |
| average_degree | 0.604 | 0.671 |
| max_degree | 0.555 | 0.633 |
| transitivity | 0.240 | 0.362 |
| clustering_coefficient | 0.227 | 0.344 |
| degree_assortativity | 0.048 | 0.174 |

**Novelty CI threshold (0.7) not passed**: The formula is highly correlated with density (0.855) and diameter (0.846). This is expected — average path length is fundamentally linked to these graph properties. The formula captures this relationship through its density and structural terms.

> **Two novelty thresholds serve different purposes**: The **novelty gate** (threshold=0.15) operates *during* generation to reject trivially-correlated candidates in real time (e.g., `return s['diameter']`). The **novelty CI threshold** (0.7) is a stricter *post-hoc* assessment that checks whether the best formula's 95% CI upper bound for |ρ| with any known invariant stays below 0.7 — confirming the formula captures genuinely new structure rather than proxying a known quantity.

## 6. Failure Mode Analysis

### Self-Correction Stats
- Attempted repairs: 36
- Successful repairs: 14 (38.9% success rate)
- Failed repairs: 22

### Rejection Categories
1. **below_novelty_threshold (53)**: LLM frequently produces `return s['diameter']` or similar trivial mappings. The 0.15 gate correctly catches these.
2. **no_valid_train_predictions (40)**: Code fails sandbox execution — primarily `import` statements (forbidden), syntax errors, or runtime exceptions. Self-correction repairs ~39% of these.
3. **below_train_threshold (24)**: Formulas execute but have poor correlation with target (train_signal < 0.3).

### Stagnation Pattern
- Best formula found in generation 1 (very early)
- No improvement from gen 2-12 (10 generations of stagnation → early stop at gen 12)
- Suggests the REFINEMENT strategy at temp=0.8 found a strong formula quickly but couldn't iterate further

### Island 3 (NOVEL) Performance
- Only 15 candidates accepted (vs 45-52 for other islands)
- Best spearman was negative (-0.570), meaning the formula inversely correlates
- High novelty scores (0.324) confirm it explores different space
- Low acceptance rate suggests temp=1.2 produces mostly invalid or trivial code

## 7. Recommendation: CONTINUE to Phase 2

### Criteria Assessment

| Criterion | Met? | Evidence |
|-----------|------|----------|
| Val Spearman ≥ 0.85 | YES | 0.937 |
| Test Spearman ≥ 0.80 | YES | 0.922 |
| Interpretable formula | YES | Closed-form with physical meaning |
| Novelty gate functional | YES | 45% of rejections from gate |
| Strategy differentiation | YES | Different islands produce different results |

### Phase 2 Recommendations

1. **Run multi-seed benchmark** (5 seeds) to assess statistical significance and formula diversity
2. **Lower novelty CI threshold** or accept that ASPL formulas will correlate with density/diameter by nature
3. **Increase NOVEL island success rate**: Consider temp=1.0 instead of 1.2, or add more constrained guidance
4. **Install Julia/PySR** for proper symbolic regression baseline comparison
5. **Add Random Forest baseline** (install scikit-learn in sandbox deps)
6. **Scale to Phase 2 graph sizes** (N=30-1000) to test formula scaling behavior
7. **Investigate stagnation**: The best formula appeared in gen 1 — consider whether the island model can better explore the space with longer runs or different migration strategies
