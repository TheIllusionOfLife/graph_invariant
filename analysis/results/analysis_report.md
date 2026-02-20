# Cross-Experiment Analysis Report

## Experiment Comparison

| Experiment | Mode | Success | Val Spearman | Test Spearman | Generations |
| --- | --- | --- | --- | --- | --- |
| experiment_map_elites_aspl | correlation | True | 0.9353 | 0.9471 | 30 |
| experiment_algebraic_connectivity | correlation | False | 0.7645 | 0.7779 | 20 |
| experiment_upper_bound_aspl | upper_bound | False | 0.4227 | 0.3590 | 20 |
| experiment_v2 | unknown | True | 0.9370 | 0.9215 | 12 |
| benchmark/benchmark_20260215T230550Z | unknown | False | N/A | N/A | None |

## Multi-Seed Aggregates

| Experiment Group | Seeds | Val mean±std | Val CI95 | Test mean±std | Test CI95 |
| --- | --- | --- | --- | --- | --- |
| benchmark/benchmark_20260215T230550Z | 5 | 0.9265 ± 0.0118 | ±0.0103 | 0.9206 ± 0.0304 | ±0.0266 |

## experiment_map_elites_aspl

- Fitness mode: correlation
- Success: True
- Stop reason: max_generations_reached
- Final generation: 30
- Validation Spearman: 0.9353
- Test Spearman: 0.9471
- Best formula AST nodes: 358

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.426 -> 0.552
- MAP-Elites coverage: 2 -> 5 cells

### Acceptance Funnel

- Final generation attempted: 30
- Final generation accepted: 14
- Final generation acceptance rate: 0.467

### Repair Breakdown

- Repair attempts: 164
- Repair successes: 68
- Repair failures: 96

### Baselines

- linear_regression: val=0.9748, test=0.9749
- random_forest: val=0.9608, test=0.9508
- PySR: val=0.9817, test=0.9747

### OOD Generalization

- large_random: spearman=0.9568 (100/100 valid)
- extreme_params: spearman=0.9263 (50/50 valid)
- special_topology: spearman=0.5000 (8/8 valid)

### Best Candidate Code

```python
def new_invariant(s):
    eps = 1e-12

    # Basic graph quantities
    n          = s['n']
    m          = s['m']
    density    = s['density']
    avg_deg    = s['avg_degree']
    max_deg    = s['max_degree']
    min_deg    = s['min_degree']
    std_deg    = s['std_degree']
    avg_clust  = s['avg_clustering']
    trans      = s['transitivity']
    assort     = s['degree_assortativity']
    num_tri    = s['num_triangles']
    degrees    = s['degrees']

    # ----- Multiplicative components -----
    base = pow(n, 0.5) / (avg_deg + 1)                    # size vs average degree
    dens_factor = 1 / (density + eps)                    # sparsity
    clu_factor  = pow(1 + avg_clust, 0.6)                # clustering
    trans_factor = pow(1 + trans, 0.5)                   # transitivity
    assort_factor = 1 + abs(assort)                      # assortativity
    tri_factor = 1 + num_tri / (m + 1)                   # triangles
    std_factor = 1 + std_deg / (avg_deg + eps)           # degree spread
    size_edge_factor = pow(n / (m + 1), 0.5)             # size‑edge interaction

    # Degree‑sequence shape via harmonic mean
    nonzero = [d for d in degrees if d > 0]
    if nonzero:
        harmonic = len(nonzero) / sum(1.0 / d for d in nonzero)
    else:
        harmonic = 1.0
    deg_shape_factor = harmonic / (avg_deg + eps)

    # Relative average to maximum degree
    rel_avg_max = 1 + avg_deg / (max_deg + 1)

    # ----- Combine all factors -----
    estimate = (base * dens_factor * clu_factor * trans_factor *
                assort_factor * tri_factor * std_factor *
                size_edge_factor * deg_shape_factor * rel_avg_max)

    # Clamp to realistic bounds
    if estimate < 1.0:
        estimate = 1.0
    elif estimate > n:
        estimate = float(n)

    return estimate
```

## experiment_algebraic_connectivity

- Fitness mode: correlation
- Success: False
- Stop reason: max_generations_reached
- Final generation: 20
- Validation Spearman: 0.7645
- Test Spearman: 0.7779
- Best formula AST nodes: 278

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.447 -> 0.494

### Acceptance Funnel

- Final generation attempted: 23
- Final generation accepted: 19
- Final generation acceptance rate: 0.826

### Repair Breakdown

- Repair attempts: 75
- Repair successes: 36
- Repair failures: 39

### Baselines

- linear_regression: val=0.8577, test=0.8663
- random_forest: val=0.9297, test=0.9300
- PySR: val=0.9409, test=0.9375

### Best Candidate Code

```python
def new_invariant(s):
    """
    Lightweight estimate of algebraic connectivity from pre‑computed
    graph statistics.

    The score is a product of several dimensionless factors that
    capture the main structural influences on the second‑smallest
    Laplacian eigenvalue.  Each component is strictly positive and
    uses only built‑in arithmetic, so the function can be called
    without any external libraries.

    Parameters
    ----------
    s : dict
        Pre‑computed graph features.  Expected keys:

        n, m, density, avg_degree, max_degree, min_degree,
        std_degree, avg_clustering, transitivity,
        degree_assortativity, num_triangles, degrees

    Returns
    -------
    float
        Proxy for the algebraic connectivity.  Larger values
        indicate a graph expected to have a larger Fiedler value.
    """
    # small constant to keep denominators safe
    eps = 1e-12

    n          = float(s['n'])
    avg_deg    = float(s['avg_degree'])
    max_deg    = float(s['max_degree'])
    min_deg    = float(s['min_degree'])
    std_deg    = float(s['std_degree'])
    density    = float(s['density'])
    num_tri    = float(s['num_triangles'])
    avg_clust  = float(s['avg_clustering'])
    trans      = float(s['transitivity'])
    assort     = float(s['degree_assortativity'])

    # ---- 1. Size normalisation ----
    size_term = n ** 0.5                     # √n

    # ---- 2. Degree regularity ----
    # More regular degree distributions (low std) raise connectivity
    degree_term = (avg_deg + 1.0) / (std_deg + 1.0 + eps)

    # ---- 3. Triangle richness ----
    max_tri_possible = n * (n - 1.0) * (n - 2.0) / 6.0
    tri_frac = num_tri / (max_tri_possible + eps)
    triangle_term = 1.0 + tri_frac ** 0.25

    # ---- 4. Edge density effect ----
    density_term = density ** 0.5             # √density

    # ---- 5. Penalty for high clustering ----
    clustering_pen = 1.0 / (1.0 + avg_clust ** 1.5)

    # ---- 6. Penalty for high transitivity ----
    trans_pen = 1.0 / (1.0 + trans ** 1.2)

    # ---- 7. Penalty for strong assortativity ----
    assort_pen = 1.0 / (1.0 + abs(assort) ** 0.9)

    # ---- combine multiplicatively ----
    score = (size_term *
             degree_term *
             triangle_term *
             density_term *
             clustering_pen *
             trans_pen *
             assort_pen)

    return score
```

## experiment_upper_bound_aspl

- Fitness mode: upper_bound
- Success: False
- Stop reason: max_generations_reached
- Final generation: 20
- Validation Spearman: 0.4227
- Test Spearman: 0.3590
- Best formula AST nodes: 386

### Bounds Diagnostics

- candidate_mean_gap_max: 79.7868812761218
- candidate_mean_gap_min: 0.13880288873952462
- candidate_satisfaction_max: 1.0
- candidate_satisfaction_min: 0.175
- test_bound_score: 0.4986280800131158
- test_mean_gap: 0.6937665954753548
- test_satisfaction_rate: 0.844559585492228
- val_bound_score: 0.5137132073914047
- val_mean_gap: 0.6944596882445031
- val_satisfaction_rate: 0.8704663212435233
- Convergence: 0.228 -> 0.453

### Acceptance Funnel

- Final generation attempted: 21
- Final generation accepted: 20
- Final generation acceptance rate: 0.952

### Repair Breakdown

- Repair attempts: 56
- Repair successes: 27
- Repair failures: 29

### OOD Generalization

- large_random: spearman=N/A (97/100 valid)
- extreme_params: spearman=N/A (49/50 valid)
- special_topology: spearman=N/A (8/8 valid)

### Best Candidate Code

```python
def new_invariant(s):
    n = s.get('n', 0)
    if n <= 1:
        return 0.0

    # Path‑graph bound (maximum ASPL over all connected graphs)
    path_bound = (n + 1) / 3.0

    # Density bound: if density > 0.5, diameter ≤ 2
    density = s.get('density', 0.0)
    density_bound = 2.0 if density > 0.5 else n - 1

    # Moore bound using maximum degree Δ
    max_deg = s.get('max_degree', 1)
    moore_max = n - 1
    if max_deg > 2:
        nodes = 1
        power = 1
        radius = 0
        deg = int(max_deg)
        while nodes < n:
            nodes += deg * power
            power *= (deg - 1)
            radius += 1
        moore_max = radius + 1

    # Moore bound using minimum degree δ
    min_deg = s.get('min_degree', 1)
    moore_min = n - 1
    if min_deg > 1:
        nodes = 1
        power = 1
        radius = 0
        deg = int(min_deg)
        while nodes < n:
            nodes += deg * power
            power *= (deg - 1)
            radius += 1
        moore_min = radius + 1

    # Moore bound using average degree
    avg_deg = s.get('avg_degree', 1)
    moore_avg = n - 1
    if avg_deg > 1:
        nodes = 1
        power = 1
        radius = 0
        deg = int(avg_deg)
        while nodes < n:
            nodes += deg * power
            power *= (deg - 1)
            radius += 1
        moore_avg = radius + 1

    # Trivial bound
    trivial = n - 1

    # Choose the tightest guaranteed upper bound
    final = path_bound
    if density_bound < final:
        final = density_bound
    if moore_max < final:
        final = moore_max
    if moore_min < final:
        final = moore_min
    if moore_avg < final:
        final = moore_avg
    if trivial < final:
        final = trivial

    return float(final)
```

## experiment_v2

- Fitness mode: unknown
- Success: True
- Stop reason: early_stop
- Final generation: 12
- Validation Spearman: 0.9370
- Test Spearman: 0.9215
- Best formula AST nodes: 208

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.497 -> 0.549

### Acceptance Funnel

- Final generation attempted: 21
- Final generation accepted: 15
- Final generation acceptance rate: 0.714

### Repair Breakdown

- Repair attempts: 36
- Repair successes: 14
- Repair failures: 22

### Baselines

- linear_regression: val=0.9748, test=0.9749
- random_forest: val=N/A, test=N/A

### OOD Generalization

- large_random: spearman=0.8856 (100/100 valid)
- extreme_params: spearman=0.9805 (50/50 valid)
- special_topology: spearman=0.7857 (8/8 valid)

### Best Candidate Code

```python
def new_invariant(s):
    """
    Estimate the average shortest path length from pre‑computed graph features.
    The formula combines random‑graph theory with corrections for clustering,
    degree heterogeneity, density, and assortativity.  All operations are
    safeguarded against division by zero and missing values.
    """
    # Basic graph statistics (safe defaults)
    n          = s.get('n', 0)
    avg_deg    = s.get('avg_degree', 0.0) or 1.0          # avoid zero
    trans      = s.get('transitivity', 0.0) or 0.0
    avg_clust  = s.get('avg_clustering', 0.0) or 0.0
    std_deg    = s.get('std_degree', 0.0) or 1.0
    density    = s.get('density', 0.0) or 1e-6            # avoid zero
    assort     = s.get('degree_assortativity', 0.0) or 0.0

    eps = 1e-6

    # 1. Random‑graph baseline: log(n)/log(avg_deg+1)
    base = np.log(max(n, 2)) / np.log(max(avg_deg + 1.0, 2.0))

    # 2. Clustering adjustment: more triangles → longer paths
    clustering_adj = 1.0 + 0.5 * (trans + avg_clust)

    # 3. Degree variance adjustment: higher variance → shorter paths
    variance_adj = 1.0 / (1.0 + std_deg / (avg_deg + 1.0))

    # 4. Density adjustment: sparser graphs tend to have longer paths
    density_adj = np.sqrt(1.0 / (density + eps))

    # 5. Assortativity adjustment: positive assortativity can lengthen paths
    assort_adj = 1.0 + 0.5 * assort  # range roughly [0.5, 1.5]

    return base * clustering_adj * variance_adj * density_adj * assort_adj
```

## benchmark/benchmark_20260215T230550Z

- Fitness mode: unknown
- Success: False
- Stop reason: N/A
- Final generation: N/A

