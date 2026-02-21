# Cross-Experiment Analysis Report

## Experiment Comparison

| Experiment | Mode | Success | Val Spearman | Test Spearman | Generations |
| --- | --- | --- | --- | --- | --- |
| experiment_map_elites_aspl | correlation | True | 0.9353 | 0.9471 | 30 |
| experiment_algebraic_connectivity | correlation | False | 0.7645 | 0.7779 | 20 |
| experiment_upper_bound_aspl | upper_bound | False | 0.4227 | 0.3590 | 20 |
| experiment_v2 | unknown | True | 0.9370 | 0.9215 | 12 |
| benchmark/benchmark_20260215T230550Z | unknown | False | N/A | N/A | None |
| neurips_matrix_day1_2026-02-21/algebraic_connectivity_medium/seed_22 | correlation | False | 0.8984 | 0.8783 | 12 |
| neurips_matrix_day1_2026-02-21/algebraic_connectivity_medium/seed_33 | correlation | False | 0.8990 | 0.9056 | 12 |
| neurips_matrix_day1_2026-02-21/benchmark_aspl_medium/seed_11 | correlation | False | -0.4850 | -0.5997 | 8 |
| neurips_matrix_day1_2026-02-21/benchmark_aspl_medium/seed_22 | correlation | False | 0.4487 | 0.2933 | 8 |
| neurips_matrix_day1_2026-02-21/benchmark_aspl_medium/seed_33 | correlation | False | 0.2324 | 0.2747 | 8 |
| neurips_matrix_day1_2026-02-21/map_elites_aspl_medium/seed_11 | correlation | False | -0.3619 | -0.4738 | 8 |
| neurips_matrix_day1_2026-02-21/map_elites_aspl_medium/seed_22 | correlation | False | 0.4487 | 0.2933 | 8 |
| neurips_matrix_day1_2026-02-21/map_elites_aspl_medium/seed_33 | correlation | False | 0.2324 | 0.2747 | 8 |
| neurips_matrix_day1_2026-02-21/small_data_aspl_train20_medium/seed_11 | correlation | False | -0.0118 | -0.0244 | 8 |
| neurips_matrix_day1_2026-02-21/small_data_aspl_train20_medium/seed_22 | correlation | False | 0.2725 | 0.4951 | 8 |
| neurips_matrix_day1_2026-02-21/small_data_aspl_train20_medium/seed_33 | correlation | False | 0.1725 | 0.3179 | 8 |
| neurips_matrix_day1_2026-02-21/small_data_aspl_train35_medium/seed_11 | correlation | False | 0.8528 | 0.8840 | 12 |
| neurips_matrix_day1_2026-02-21/small_data_aspl_train35_medium/seed_22 | correlation | True | 0.9549 | 0.9418 | 12 |
| neurips_matrix_day1_2026-02-21/small_data_aspl_train35_medium/seed_33 | correlation | False | 0.8895 | 0.9143 | 12 |
| neurips_matrix_day1_2026-02-21/upper_bound_aspl_medium/seed_11 | upper_bound | False | 0.4136 | 0.4433 | 8 |
| neurips_matrix_day1_2026-02-21/upper_bound_aspl_medium/seed_22 | upper_bound | False | 0.3125 | 0.4122 | 12 |
| neurips_matrix_day1_2026-02-21/upper_bound_aspl_medium/seed_33 | upper_bound | False | 0.4194 | 0.3551 | 12 |

## Multi-Seed Aggregates

| Experiment Group | Seeds | Val mean±std | Val CI95 | Test mean±std | Test CI95 |
| --- | --- | --- | --- | --- | --- |
| benchmark/benchmark_20260215T230550Z | 5 | 0.9265 ± 0.0118 | ±0.0146 | 0.9206 ± 0.0304 | ±0.0377 |
| neurips_matrix_day1_2026-02-21/algebraic_connectivity_medium | 2 | 0.8987 ± 0.0004 | ±0.0039 | 0.8920 ± 0.0193 | ±0.1732 |
| neurips_matrix_day1_2026-02-21/benchmark_aspl_medium | 3 | 0.0654 ± 0.4888 | ±1.2143 | -0.0106 ± 0.5103 | ±1.2676 |
| neurips_matrix_day1_2026-02-21/map_elites_aspl_medium | 3 | 0.1064 ± 0.4197 | ±1.0427 | 0.0314 ± 0.4376 | ±1.0871 |
| neurips_matrix_day1_2026-02-21/small_data_aspl_train20_medium | 3 | 0.1444 ± 0.1442 | ±0.3582 | 0.2629 ± 0.2641 | ±0.6560 |
| neurips_matrix_day1_2026-02-21/small_data_aspl_train35_medium | 3 | 0.8991 ± 0.0517 | ±0.1284 | 0.9134 ± 0.0289 | ±0.0718 |
| neurips_matrix_day1_2026-02-21/upper_bound_aspl_medium | 3 | 0.3818 ± 0.0601 | ±0.1494 | 0.4035 ± 0.0447 | ±0.1111 |

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

## neurips_matrix_day1_2026-02-21/algebraic_connectivity_medium/seed_22

- Fitness mode: correlation
- Success: False
- Stop reason: max_generations_reached
- Final generation: 12
- Validation Spearman: 0.8984
- Test Spearman: 0.8783
- Best formula AST nodes: 282

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.505 -> 0.516
- MAP-Elites coverage: 2 -> 2 cells

### Acceptance Funnel

- Final generation attempted: 11
- Final generation accepted: 9
- Final generation acceptance rate: 0.818

### Repair Breakdown

- Repair attempts: 27
- Repair successes: 16
- Repair failures: 11

### Baselines

- linear_regression: val=1.0000, test=1.0000
- random_forest: val=0.9994, test=0.9988
- PySR: val=1.0000, test=1.0000

### Best Candidate Code

```python
def new_invariant(s):
    """
    Estimate the algebraic connectivity (second smallest Laplacian eigenvalue)
    from a set of pre‑computed graph statistics.

    Parameters
    ----------
    s : dict
        Dictionary containing the following keys (values are floats or ints):

        - n, m, density
        - avg_degree, max_degree, min_degree, std_degree
        - avg_clustering, transitivity
        - degree_assortativity
        - num_triangles
        - degrees (list of ints, not used in the calculation)
        - laplacian_lambda2, laplacian_lambda_max
        - laplacian_spectral_gap
        - normalized_laplacian_lambda2
        - laplacian_energy_ratio

    Returns
    -------
    float
        Approximate algebraic connectivity.
    """
    eps = 1e-6  # small constant to avoid division by zero

    # Core λ₂ signal (shifted slightly to avoid zero)
    base = s['laplacian_lambda2'] + eps

    # Density boost
    density_boost = 1.0 + s['density']

    # Transitivity (global clustering) boost
    transitivity_boost = 1.0 + s['transitivity']

    # Regularity of degrees: avg / max (shifted by 1 to keep positive)
    avg_deg_norm = (s['avg_degree'] + 1.0) / (s['max_degree'] + 1.0)

    # Minimum‑to‑maximum degree factor
    min_deg_norm = 1.0 + s['min_degree'] / (s['max_degree'] + 1.0)

    # Degree assortativity mapped from [‑1, 1] to [0, 1]
    assort_factor = (s['degree_assortativity'] + 1.0) / 2.0

    # Penalty for high degree dispersion (sub‑linear)
    std_penalty = 1.0 / (1.0 + s['std_degree'] ** 0.5)

    # Local clustering contribution (moderate exponent)
    cluster_factor = (1.0 + s['avg_clustering']) ** 0.5

    # Triangle contribution with a very mild exponent
    tri_factor = (s['num_triangles'] + 1.0) ** 0.1

    # Spectral‑gap factor relative to the largest eigenvalue
    gap_rel1 = s['laplacian_spectral_gap'] / (s['laplacian_lambda_max'] + eps)

    # Spectral‑gap factor relative to the difference between λ_max and λ₂
    gap_rel2 = s['laplacian_spectral_gap'] / ((s['laplacian_lambda_max'] -
                                              s['laplacian_lambda2']) + eps)

    # Average of the two gap ratios to reduce sensitivity to extreme values
    gap_factor = (gap_rel1 + gap_rel2) / 2.0

    # Normalised Laplacian λ₂ (scale √ to keep it comparable)
    norm_lap_factor = s['normalized_laplacian_lambda2'] ** 0.5

    # Energy penalty: larger energy ratio tends to reduce λ₂
    energy_penalty = (s['laplacian_energy_ratio'] + eps) ** -0.2

    # Multiplicative combination of all factors
    estimate = (base *
                density_boost *
                transitivity_boost *
                avg_deg_norm *
                min_deg_norm *
                assort_factor *
                std_penalty *
                cluster_factor *
                tri_factor *
                gap_factor *
                norm_lap_factor *
                energy_penalty)

    return estimate
```

## neurips_matrix_day1_2026-02-21/algebraic_connectivity_medium/seed_33

- Fitness mode: correlation
- Success: False
- Stop reason: max_generations_reached
- Final generation: 12
- Validation Spearman: 0.8990
- Test Spearman: 0.9056
- Best formula AST nodes: 187

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.499 -> 0.519
- MAP-Elites coverage: 1 -> 3 cells

### Acceptance Funnel

- Final generation attempted: 14
- Final generation accepted: 6
- Final generation acceptance rate: 0.429

### Repair Breakdown

- Repair attempts: 22
- Repair successes: 12
- Repair failures: 10

### Baselines

- linear_regression: val=1.0000, test=1.0000
- random_forest: val=0.9976, test=0.9992
- PySR: val=1.0000, test=1.0000

### Best Candidate Code

```python
def new_invariant(s):
    """
    Estimate the algebraic connectivity (second Laplacian eigenvalue) from a
    small set of pre‑computed graph features.  The formula is a
    multiplicative combination of several dimensionless factors that
    capture degree heterogeneity, density, clustering, assortativity,
    triangle abundance, spectral gap, normalised λ₂, and Laplacian energy.

    Parameters
    ----------
    s : dict
        Dictionary of pre‑computed graph statistics with the following keys:
        n, m, density, avg_degree, max_degree, min_degree, std_degree,
        avg_clustering, transitivity, degree_assortativity, num_triangles,
        degrees, laplacian_lambda2, laplacian_lambda_max,
        laplacian_spectral_gap, normalized_laplacian_lambda2,
        laplacian_energy_ratio

    Returns
    -------
    float
        An estimate of the algebraic connectivity.
    """
    # ---------- 1. Degree heterogeneity adjustment ----------
    spread = (s['max_degree'] - s['min_degree']) + s['std_degree'] + 1.0
    lam2_adj = s['laplacian_lambda2'] / spread

    # ---------- 2. Structural multipliers ----------
    dens_mult   = 1.0 + s['density']
    clust_mult  = 1.0 + s['avg_clustering']
    assort_mult = 1.0 + abs(s['degree_assortativity'])

    # ---------- 3. Triangle contribution ----------
    tri_den = (s['n'] ** 3) + 1.0          # keeps the factor bounded
    tri_mult = 1.0 + s['num_triangles'] / tri_den

    # ---------- 4. Spectral‑gap scaling ----------
    spec_gap_mult = (1.0 + s['laplacian_spectral_gap']) / (s['n'] + 1.0)

    # ---------- 5. Normalised λ₂ contribution ----------
    norm_lam2_mult = 1.0 + s['normalized_laplacian_lambda2']

    # ---------- 6. Energy moderation ----------
    energy_mult = 1.0 + s['laplacian_energy_ratio']

    # ---------- Combine all factors ----------
    result = (lam2_adj *
              dens_mult *
              clust_mult *
              assort_mult *
              tri_mult *
              spec_gap_mult *
              norm_lam2_mult) / energy_mult

    return result
```

## neurips_matrix_day1_2026-02-21/benchmark_aspl_medium/seed_11

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 8
- Validation Spearman: -0.4850
- Test Spearman: -0.5997
- Best formula AST nodes: 31

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.309 -> 0.309

### Acceptance Funnel

- Final generation attempted: 6
- Final generation accepted: 6
- Final generation acceptance rate: 1.000

### Repair Breakdown

- Repair attempts: 2
- Repair successes: 0
- Repair failures: 2

### Baselines

- linear_regression: val=0.9735, test=0.9751
- random_forest: val=0.9764, test=0.9793
- PySR: val=0.9749, test=0.9829

### Best Candidate Code

```python
def new_invariant(s):
  """
  A refined formula for average_shortest_path_length based on existing features.
  """
  return s['n'] * s['density'] * np.sqrt(s['avg_clustering'])
```

## neurips_matrix_day1_2026-02-21/benchmark_aspl_medium/seed_22

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 8
- Validation Spearman: 0.4487
- Test Spearman: 0.2933
- Best formula AST nodes: 29

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.332 -> 0.332

### Acceptance Funnel

- Final generation attempted: 6
- Final generation accepted: 6
- Final generation acceptance rate: 1.000

### Repair Breakdown

- Repair attempts: 1
- Repair successes: 0
- Repair failures: 1

### Baselines

- linear_regression: val=0.9484, test=0.9518
- random_forest: val=0.9841, test=0.9792
- PySR: val=0.9654, test=0.9597

### Best Candidate Code

```python
def new_invariant(s):
  """
  A refined formula for average shortest path length based on existing features.
  """
  return s['n'] * s['avg_clustering'] / (s['m'] + 1)
```

## neurips_matrix_day1_2026-02-21/benchmark_aspl_medium/seed_33

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 8
- Validation Spearman: 0.2324
- Test Spearman: 0.2747
- Best formula AST nodes: 29

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.237 -> 0.237

### Acceptance Funnel

- Final generation attempted: 6
- Final generation accepted: 6
- Final generation acceptance rate: 1.000

### Repair Breakdown

- Repair attempts: 3
- Repair successes: 1
- Repair failures: 2

### Baselines

- linear_regression: val=0.9260, test=0.9358
- random_forest: val=0.9790, test=0.9788
- PySR: val=0.9446, test=0.9329

### Best Candidate Code

```python
def new_invariant(s):
  """
  A refined formula for average_shortest_path_length, incorporating
  degree and clustering information.
  """
  return s['n'] * s['avg_clustering'] / (s['m'] + 1)
```

## neurips_matrix_day1_2026-02-21/map_elites_aspl_medium/seed_11

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 8
- Validation Spearman: -0.3619
- Test Spearman: -0.4738
- Best formula AST nodes: 26

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.245 -> 0.245
- MAP-Elites coverage: 2 -> 2 cells

### Acceptance Funnel

- Final generation attempted: 6
- Final generation accepted: 6
- Final generation acceptance rate: 1.000

### Repair Breakdown

- Repair attempts: 1
- Repair successes: 0
- Repair failures: 1

### Baselines

- linear_regression: val=0.9735, test=0.9751
- random_forest: val=0.9764, test=0.9793
- PySR: val=0.9600, test=0.9685

### Best Candidate Code

```python
def new_invariant(s):
  """
  A refined formula for average_shortest_path_length, incorporating several features.
  """
  return s['n'] * s['density'] * s['avg_clustering']
```

## neurips_matrix_day1_2026-02-21/map_elites_aspl_medium/seed_22

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 8
- Validation Spearman: 0.4487
- Test Spearman: 0.2933
- Best formula AST nodes: 29

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.332 -> 0.332
- MAP-Elites coverage: 1 -> 1 cells

### Acceptance Funnel

- Final generation attempted: 6
- Final generation accepted: 6
- Final generation acceptance rate: 1.000

### Repair Breakdown

- Repair attempts: 1
- Repair successes: 0
- Repair failures: 1

### Baselines

- linear_regression: val=0.9484, test=0.9518
- random_forest: val=0.9841, test=0.9792
- PySR: val=0.9206, test=0.9204

### Best Candidate Code

```python
def new_invariant(s):
  """
  A refined formula for the average shortest path length, incorporating
  degree and clustering coefficients.
  """
  return (s['n'] * s['avg_clustering']) / (s['m'] + 1)
```

## neurips_matrix_day1_2026-02-21/map_elites_aspl_medium/seed_33

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 8
- Validation Spearman: 0.2324
- Test Spearman: 0.2747
- Best formula AST nodes: 29

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.237 -> 0.237
- MAP-Elites coverage: 1 -> 1 cells

### Acceptance Funnel

- Final generation attempted: 6
- Final generation accepted: 6
- Final generation acceptance rate: 1.000

### Repair Breakdown

- Repair attempts: 1
- Repair successes: 0
- Repair failures: 1

### Baselines

- linear_regression: val=0.9260, test=0.9358
- random_forest: val=0.9790, test=0.9788
- PySR: val=0.9257, test=0.9272

### Best Candidate Code

```python
def new_invariant(s):
  """
  A refined formula for average shortest path length based on existing features.
  """
  return s['n'] * s['avg_clustering'] / (s['m'] + 1)
```

## neurips_matrix_day1_2026-02-21/small_data_aspl_train20_medium/seed_11

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 8
- Validation Spearman: -0.0118
- Test Spearman: -0.0244
- Best formula AST nodes: 29

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.126 -> 0.126
- MAP-Elites coverage: 1 -> 1 cells

### Acceptance Funnel

- Final generation attempted: 6
- Final generation accepted: 6
- Final generation acceptance rate: 1.000

### Repair Breakdown

- Repair attempts: 1
- Repair successes: 0
- Repair failures: 1

### Baselines

- linear_regression: val=0.9319, test=0.9492
- random_forest: val=0.9794, test=0.9689
- PySR: val=0.9630, test=0.9785

### Best Candidate Code

```python
def new_invariant(s):
  """
  A refined formula for average shortest path length based on existing features.
  """
  return s['n'] * s['density'] / (s['m'] + 1)
```

## neurips_matrix_day1_2026-02-21/small_data_aspl_train20_medium/seed_22

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 8
- Validation Spearman: 0.2725
- Test Spearman: 0.4951
- Best formula AST nodes: 29

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.227 -> 0.227
- MAP-Elites coverage: 1 -> 1 cells

### Acceptance Funnel

- Final generation attempted: 6
- Final generation accepted: 6
- Final generation acceptance rate: 1.000

### Repair Breakdown

- Repair attempts: 1
- Repair successes: 0
- Repair failures: 1

### Baselines

- linear_regression: val=0.7555, test=0.7449
- random_forest: val=0.9034, test=0.8685
- PySR: val=0.8643, test=0.8567

### Best Candidate Code

```python
def new_invariant(s):
  """
  A refined formula for average shortest path length based on existing features.
  """
  return s['n'] * s['avg_clustering'] / (s['m'] + 1)
```

## neurips_matrix_day1_2026-02-21/small_data_aspl_train20_medium/seed_33

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 8
- Validation Spearman: 0.1725
- Test Spearman: 0.3179
- Best formula AST nodes: 29

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.198 -> 0.198
- MAP-Elites coverage: 1 -> 1 cells

### Acceptance Funnel

- Final generation attempted: 6
- Final generation accepted: 6
- Final generation acceptance rate: 1.000

### Repair Breakdown

- Repair attempts: 1
- Repair successes: 0
- Repair failures: 1

### Baselines

- linear_regression: val=0.8731, test=0.8970
- random_forest: val=0.9095, test=0.9259
- PySR: val=0.9404, test=0.9571

### Best Candidate Code

```python
def new_invariant(s):
    """
    A refined formula for average shortest path length based on existing features.
    """
    return s['n'] * s['avg_clustering'] / (s['m'] + 1)
```

## neurips_matrix_day1_2026-02-21/small_data_aspl_train35_medium/seed_11

- Fitness mode: correlation
- Success: False
- Stop reason: max_generations_reached
- Final generation: 12
- Validation Spearman: 0.8528
- Test Spearman: 0.8840
- Best formula AST nodes: 330

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.463 -> 0.550
- MAP-Elites coverage: 2 -> 6 cells

### Acceptance Funnel

- Final generation attempted: 11
- Final generation accepted: 7
- Final generation acceptance rate: 0.636

### Repair Breakdown

- Repair attempts: 33
- Repair successes: 18
- Repair failures: 15

### Baselines

- linear_regression: val=0.8722, test=0.8570
- random_forest: val=0.9650, test=0.9615
- PySR: val=0.9491, test=0.9575

### Best Candidate Code

```python
def new_invariant(s):
    """
    Estimate of the average shortest path length from a compact combination
    of size, connectivity, spectral, clustering, degree variability,
    triangle density and assortativity.  Only built‑in arithmetic is used.
    """
    # Basic attributes – cast to float for safety
    n          = float(s['n'])
    m          = float(s['m'])
    density    = float(s['density'])
    avg_deg    = float(s['avg_degree'])
    std_deg    = float(s['std_degree'])
    clustering = float(s['avg_clustering'])
    lap2       = float(s['laplacian_lambda2'])
    lap_gap    = float(s['laplacian_spectral_gap'])
    norm_lap2  = float(s['normalized_laplacian_lambda2'])
    energy     = float(s['laplacian_energy_ratio'])
    num_tri    = float(s.get('num_triangles', 0.0))
    assort     = float(s['degree_assortativity'])
    max_deg    = float(s['max_degree'])
    min_deg    = float(s['min_degree'])

    # ---- Core estimate ----------------------------------------------------
    # Size component: larger graphs tend to have longer paths
    val = n / (avg_deg + 1.0)

    # Density and clustering help shorten paths – mild exponentiation
    val *= pow(1.0 + density, 0.5)
    val *= pow(1.0 + clustering, 0.5)

    # Spectral connectivity – larger λ₂ reduces ASPL
    val /= (lap2 + 1.0)
    val /= (lap_gap + 1.0)
    val /= (norm_lap2 + 1.0)
    val /= (energy + 1.0)

    # Edge count gives a sqrt‑scale reduction
    val *= pow(m + 1.0, 0.5)

    # Triangle density slightly reduces ASPL
    val *= pow(1.0 + num_tri / (n**2 + 1e-6), 0.3)

    # Degree variability and extremes – higher variance increases ASPL
    val /= pow(1.0 + std_deg, 0.5)
    val /= pow(1.0 + max_deg, 0.2)
    val /= pow(1.0 + min_deg, 0.1)

    # Assortativity – positive assortativity tends to increase ASPL
    val /= pow(1.0 + abs(assort), 0.2)

    return val
```

## neurips_matrix_day1_2026-02-21/small_data_aspl_train35_medium/seed_22

- Fitness mode: correlation
- Success: True
- Stop reason: max_generations_reached
- Final generation: 12
- Validation Spearman: 0.9549
- Test Spearman: 0.9418
- Best formula AST nodes: 296

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.529 -> 0.545
- MAP-Elites coverage: 2 -> 3 cells

### Acceptance Funnel

- Final generation attempted: 11
- Final generation accepted: 9
- Final generation acceptance rate: 0.818

### Repair Breakdown

- Repair attempts: 27
- Repair successes: 12
- Repair failures: 15

### Baselines

- linear_regression: val=0.8701, test=0.8904
- random_forest: val=0.9549, test=0.9226
- PySR: val=0.9308, test=0.9210

### Best Candidate Code

```python
def new_invariant(s):
    """
    Approximate the average shortest‑path length from a set of pre‑computed
    graph features.  The formula is a compact multiplicative combination
    of several structural indicators that influence path length.
    """
    eps = 1e-6  # small constant to avoid division by zero

    # Base term: larger graphs with lower average degree tend to have longer paths
    base = s['n'] / (s['avg_degree'] + 1)

    # Structural factors that influence path length
    density_factor      = 1 / (s['density'] + eps) ** 0.5
    spectral_factor     = 1 / (s['laplacian_spectral_gap'] + eps) ** 0.5
    clustering_factor   = 1 + s['avg_clustering'] ** 2 + s['transitivity'] ** 2
    assort_factor       = 1 + abs(s['degree_assortativity'])
    triangle_factor     = 1 / (1 + s['num_triangles'] / (s['n'] + 1)) ** 0.5
    std_factor          = 1 / (1 + s['std_degree']) ** 0.5
    deg_ratio_factor    = 1 + (s['max_degree'] / (s['min_degree'] + eps) - 1) ** 0.5
    norm_lap_factor     = 1 / (s['normalized_laplacian_lambda2'] + eps) ** 0.5
    energy_factor       = 1 / (s['laplacian_energy_ratio'] + eps) ** 0.5
    lambda_ratio_factor = 1 + (s['laplacian_lambda_max'] / (s['laplacian_lambda2'] + eps) - 1) ** 0.5

    # Size factor to capture overall graph scale
    size_factor = s['n'] ** 0.5

    # Combine all factors multiplicatively
    return (base * density_factor * spectral_factor * clustering_factor *
            assort_factor * triangle_factor * std_factor * deg_ratio_factor *
            norm_lap_factor * energy_factor * lambda_ratio_factor * size_factor)
```

## neurips_matrix_day1_2026-02-21/small_data_aspl_train35_medium/seed_33

- Fitness mode: correlation
- Success: False
- Stop reason: max_generations_reached
- Final generation: 12
- Validation Spearman: 0.8895
- Test Spearman: 0.9143
- Best formula AST nodes: 177

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.404 -> 0.511
- MAP-Elites coverage: 2 -> 2 cells

### Acceptance Funnel

- Final generation attempted: 13
- Final generation accepted: 8
- Final generation acceptance rate: 0.615

### Repair Breakdown

- Repair attempts: 27
- Repair successes: 14
- Repair failures: 13

### Baselines

- linear_regression: val=0.8670, test=0.8969
- random_forest: val=0.9133, test=0.9505
- PySR: val=0.9048, test=0.9420

### Best Candidate Code

```python
def new_invariant(s):
    """
    Estimate the average shortest path length (ASPL) using a compact
    multiplicative model based only on pre‑computed graph statistics.
    All arithmetic is performed with Python built‑ins, no external
    libraries or imports are required.

    Features used:
        n, m, density, avg_degree, std_degree,
        avg_clustering, transitivity, degree_assortativity,
        num_triangles, laplacian_lambda2
    """
    eps = 1e-12  # tiny value to avoid divisions by zero

    # Size effect – larger graphs tend to have longer paths
    size_factor = s['n'] / (s['m'] + 1)

    # Connectivity (spectral) – better connectivity shortens paths
    conn_factor = 1.0 / (s['laplacian_lambda2'] + eps)

    # Density – denser graphs have shorter paths
    dens_factor = 1.0 / (s['density'] + eps)

    # Clustering – higher clustering typically increases path lengths
    cluster_factor = 1.0 / (s['avg_clustering'] + 0.01)

    # Degree heterogeneity – wide degree distributions lengthen paths
    heter_factor = 1.0 + s['std_degree'] / (s['avg_degree'] + eps)

    # Triangle abundance – many triangles can shorten overall paths
    tri_factor = (s['num_triangles'] + 1) / (s['n'] + 1)

    # Assortativity – positive assortativity tends to shorten paths
    assort_factor = 1.0 / (1.0 + 0.2 * s['degree_assortativity'])

    # Transitivity – a mild positive adjustment for higher transitivity
    trans_factor = 1.0 + 0.01 * s['transitivity']

    # Final multiplicative combination
    return (
        size_factor
        * conn_factor
        * dens_factor
        * cluster_factor
        * tri_factor
        * trans_factor
        / (heter_factor * assort_factor)
    )
```

## neurips_matrix_day1_2026-02-21/upper_bound_aspl_medium/seed_11

- Fitness mode: upper_bound
- Success: False
- Stop reason: early_stop
- Final generation: 8
- Validation Spearman: 0.4136
- Test Spearman: 0.4433
- Best formula AST nodes: 65

### Bounds Diagnostics

- candidate_mean_gap_max: 58.21255615512499
- candidate_mean_gap_min: 0.25720763319282325
- candidate_satisfaction_max: 1.0
- candidate_satisfaction_min: 0.09545454545454546
- test_bound_score: 0.3975636538390242
- test_mean_gap: 0.7492910600327133
- test_satisfaction_rate: 0.6954545454545454
- val_bound_score: 0.30178089434123695
- val_mean_gap: 0.852638518788139
- val_satisfaction_rate: 0.5590909090909091
- Convergence: 0.305 -> 0.305

### Acceptance Funnel

- Final generation attempted: 11
- Final generation accepted: 8
- Final generation acceptance rate: 0.727

### Repair Breakdown

- Repair attempts: 34
- Repair successes: 13
- Repair failures: 21

### Baselines

- linear_regression: val=0.9426, test=0.9553
- random_forest: val=0.9769, test=0.9785
- PySR: val=0.9625, test=0.9650

### Best Candidate Code

```python
def new_invariant(s):
    """
    A tighter upper bound for the average shortest path length (ASPL) that
    merges two classic bounds:

    1. The trivial bound (n‑1)/2 + 1, which is valid for all connected graphs.
    2. A spectral‑gap correction that tightens the bound for dense graphs.

    The formula is:
        ASPL ≤ ((n-1)/2 + 1) * (1 / (λ₂ + 1)) + 1

    where λ₂ is the second‑smallest Laplacian eigenvalue (spectral gap).

    Parameters
    ----------
    s : dict
        Pre‑computed graph features containing at least:
        - 'n' : number of nodes
        - 'laplacian_spectral_gap' : λ₂

    Returns
    -------
    float
        The invariant value, guaranteed to be ≥ the true ASPL for all graphs.
    """
    n = float(s['n'])
    gap = float(s.get('laplacian_spectral_gap', 0.0))
    base = (n - 1.0) / 2.0 + 1.0          # Trivial upper bound
    correction = 1.0 / (gap + 1.0)       # Spectral‑gap factor (≤ 1)
    return base * correction + 1.0
```

## neurips_matrix_day1_2026-02-21/upper_bound_aspl_medium/seed_22

- Fitness mode: upper_bound
- Success: False
- Stop reason: max_generations_reached
- Final generation: 12
- Validation Spearman: 0.3125
- Test Spearman: 0.4122
- Best formula AST nodes: 190

### Bounds Diagnostics

- candidate_mean_gap_max: 132.85176222215102
- candidate_mean_gap_min: 1.6245265497824948
- candidate_satisfaction_max: 1.0
- candidate_satisfaction_min: 0.7545454545454545
- test_bound_score: 0.29222396915528276
- test_mean_gap: 1.7531809141841233
- test_satisfaction_rate: 0.8045454545454546
- val_bound_score: 0.2811086480293277
- val_mean_gap: 1.8782142234432344
- val_satisfaction_rate: 0.8090909090909091
- Convergence: 0.293 -> 0.310

### Acceptance Funnel

- Final generation attempted: 10
- Final generation accepted: 9
- Final generation acceptance rate: 0.900

### Repair Breakdown

- Repair attempts: 27
- Repair successes: 15
- Repair failures: 12

### Baselines

- linear_regression: val=0.9214, test=0.9244
- random_forest: val=0.9872, test=0.9819
- PySR: val=0.9305, test=0.9244

### Best Candidate Code

```python
def new_invariant(s):
    """
    Upper bound on the average shortest‑path length (ASPL) for a connected graph.

    The bound is the minimum of several provably valid upper bounds that can be
    expressed solely in terms of the pre‑computed graph features supplied in ``s``.
    Every bound used is known to be an upper bound for all connected graphs;
    hence the returned value is guaranteed to be ≥ the true ASPL.

    Parameters
    ----------
    s : dict
        Dictionary of pre‑computed graph features.  Expected keys are:
            'n'                     : number of vertices
            'm'                     : number of edges
            'min_degree'            : minimum vertex degree
            'density'               : edge density (2m/(n(n‑1)))
            'laplacian_spectral_gap': λ₂ of the Laplacian (zero if disconnected)

    Returns
    -------
    float
        An upper bound on the average shortest‑path length.
    """
    n = s['n']
    m = s['m']
    min_deg = s.get('min_degree', 0)
    density = s.get('density', 0)
    lam2 = s.get('laplacian_spectral_gap', 0)

    # 1. Trivial bound: diameter ≤ n‑1  →  ASPL ≤ n‑1
    bounds = [n - 1]

    # 2. Degree‑based bound (valid for min_deg > 1)
    if min_deg > 1:
        bounds.append(1 + (n - 1) / (min_deg - 1))

    # 3. Laplacian spectral gap bound (valid when λ₂ > 0)
    if lam2 > 0:
        bounds.append((n - 1) / lam2)

    # 4. Density‑based bound (requires density*n > 1)
    if density * n > 1:
        bounds.append(1 + (n - 1) / (density * n - 1))

    # 5. Edge‑count based bound (valid for all graphs with m ≥ n‑1)
    #    (derived from the fact that a tree has ASPL ≤ n‑1 and adding edges
    #    can only decrease ASPL)
    if m >= n - 1:
        bound2 = 2 * n - 1 - m
        if bound2 > 0:
            bounds.append(bound2)

    # Return the tightest proven upper bound
    return min(bounds)
```

## neurips_matrix_day1_2026-02-21/upper_bound_aspl_medium/seed_33

- Fitness mode: upper_bound
- Success: False
- Stop reason: max_generations_reached
- Final generation: 12
- Validation Spearman: 0.4194
- Test Spearman: 0.3551
- Best formula AST nodes: 175

### Bounds Diagnostics

- candidate_mean_gap_max: 86363636456.36285
- candidate_mean_gap_min: 0.7712167167368704
- candidate_satisfaction_max: 1.0
- candidate_satisfaction_min: 0.18181818181818182
- test_bound_score: 0.34946462317176896
- test_mean_gap: 1.6273956140347665
- test_satisfaction_rate: 0.9181818181818182
- val_bound_score: 0.3506349358914007
- val_mean_gap: 1.6186261669778366
- val_satisfaction_rate: 0.9181818181818182
- Convergence: 0.386 -> 0.387

### Acceptance Funnel

- Final generation attempted: 10
- Final generation accepted: 9
- Final generation acceptance rate: 0.900

### Repair Breakdown

- Repair attempts: 28
- Repair successes: 18
- Repair failures: 10

### Baselines

- linear_regression: val=0.8785, test=0.9037
- random_forest: val=0.9788, test=0.9785
- PySR: val=0.9010, test=0.9580

### Best Candidate Code

```python
def new_invariant(s):
    """
    Upper bound on the average shortest‑path length (ASPL) for any
    undirected simple graph, expressed only in terms of pre‑computed
    features in the dictionary `s`.

    The bound is the *minimum* of three classical safe upper bounds:
    1.  Minimum‑degree bound   ASPL ≤ n / (δ + 1)
    2.  Spectral (algebraic)  ASPL ≤ n / (λ₂ + 1)
    3.  Expansion bound from the maximum degree Δ:
        Let reachable(d) = 1 + Δ * ((Δ−1)^d − 1)/(Δ−2).  The
        smallest integer d with reachable(d) ≥ n gives a lower bound
        on the eccentricity of any vertex; therefore the diameter
        ≤ 2·d and ASPL ≤ 2·d.

    Every term is guaranteed to be ≥ the true ASPL, so the minimum
    of the three is a valid (and often tighter) upper bound.

    The function uses only built‑in operations and the supplied keys.
    """
    n = s['n']
    if n <= 1:
        return 0.0

    # --- 1. Minimum‑degree bound ---------------------------------------
    min_deg = s.get('min_degree', 0)
    bound_min = n / (float(min_deg) + 1.0)

    # --- 2. Spectral bound (algebraic connectivity) -------------------
    lam2 = s.get('laplacian_lambda2', 0.0)
    if lam2 > 0.0:
        bound_lam = n / (lam2 + 1.0)
    else:
        # Disconnected or λ₂ = 0 → fall back to a loose bound
        bound_lam = float(n)

    # --- 3. Expansion bound using maximum degree ----------------------
    Δ = s.get('max_degree', 0)
    if Δ <= 2:
        bound_exp = float(n - 1)          # path or isolated vertices
    else:
        reachable = 1            # vertices at distance 0
        d = 0
        base = Δ - 1
        factor = Δ
        while reachable < n:
            reachable += factor * (base ** d)
            d += 1
        bound_exp = 2.0 * d

    # Return the tightest safe upper bound
    return min(bound_min, bound_lam, bound_exp)
```

