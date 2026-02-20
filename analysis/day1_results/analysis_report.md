# Cross-Experiment Analysis Report

## Experiment Comparison

| Experiment | Mode | Success | Val Spearman | Test Spearman | Generations |
| --- | --- | --- | --- | --- | --- |
| neurips_matrix/map_elites_aspl_medium/seed_11 | correlation | False | -0.4622 | -0.5648 | 10 |
| neurips_matrix/map_elites_aspl_screen/seed_11 | correlation | False | -0.3255 | -0.4252 | 5 |
| neurips_matrix/map_elites_aspl_screen/seed_22 | correlation | False | 0.3619 | 0.4687 | 5 |
| neurips_matrix/map_elites_aspl_screen/seed_33 | correlation | False | 0.1917 | 0.3824 | 6 |
| neurips_matrix/small_data_aspl_train20_screen/seed_11 | correlation | False | 0.0521 | -0.0911 | 6 |
| neurips_matrix/small_data_aspl_train20_screen/seed_22 | correlation | False | -0.3173 | -0.3018 | 5 |
| neurips_matrix/small_data_aspl_train20_screen/seed_33 | correlation | False | 0.2206 | 0.2304 | 5 |
| neurips_matrix/upper_bound_aspl_screen/seed_11 | upper_bound | False | 0.0690 | 0.1828 | 5 |
| neurips_matrix/upper_bound_aspl_screen/seed_22 | upper_bound | False | -0.2561 | -0.3506 | 6 |
| neurips_matrix/upper_bound_aspl_screen/seed_33 | upper_bound | False | -0.0286 | -0.0326 | 6 |

## Multi-Seed Aggregates

| Experiment Group | Seeds | Val mean±std | Val CI95 | Test mean±std | Test CI95 |
| --- | --- | --- | --- | --- | --- |
| neurips_matrix/map_elites_aspl_medium | 1 | -0.4622 ± 0.0000 | ±0.0000 | -0.5648 ± 0.0000 | ±0.0000 |
| neurips_matrix/map_elites_aspl_screen | 3 | 0.0760 ± 0.3580 | ±0.4051 | 0.1420 ± 0.4930 | ±0.5579 |
| neurips_matrix/small_data_aspl_train20_screen | 3 | -0.0149 ± 0.2751 | ±0.3113 | -0.0542 ± 0.2680 | ±0.3033 |
| neurips_matrix/upper_bound_aspl_screen | 3 | -0.0719 ± 0.1668 | ±0.1887 | -0.0668 ± 0.2683 | ±0.3036 |

## neurips_matrix/map_elites_aspl_medium/seed_11

- Fitness mode: correlation
- Success: False
- Stop reason: max_generations_reached
- Final generation: 10
- Validation Spearman: -0.4622
- Test Spearman: -0.5648
- Best formula AST nodes: 22

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.245 -> 0.298
- MAP-Elites coverage: 1 -> 2 cells

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
- PySR: val=0.9741, test=0.9710

### Best Candidate Code

```python
def new_invariant(s):
  """
  A refined formula for the average shortest path length invariant.
  """
  return s['n'] * s['density']**0.5
```

## neurips_matrix/map_elites_aspl_screen/seed_11

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 5
- Validation Spearman: -0.3255
- Test Spearman: -0.4252
- Best formula AST nodes: 26

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

- Final generation attempted: 4
- Final generation accepted: 2
- Final generation acceptance rate: 0.500

### Repair Breakdown

- Repair attempts: 0
- Repair successes: 0
- Repair failures: 0

### Best Candidate Code

```python
def new_invariant(s):
    """
    Combines elements from the top 2 formulas into a new one.
    """
    return s['n'] * s['density'] * s['avg_clustering']
```

## neurips_matrix/map_elites_aspl_screen/seed_22

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 5
- Validation Spearman: 0.3619
- Test Spearman: 0.4687
- Best formula AST nodes: 29

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.282 -> 0.282
- MAP-Elites coverage: 1 -> 1 cells

### Acceptance Funnel

- Final generation attempted: 4
- Final generation accepted: 4
- Final generation acceptance rate: 1.000

### Repair Breakdown

- Repair attempts: 0
- Repair successes: 0
- Repair failures: 0

### Best Candidate Code

```python
def new_invariant(s):
  """
  A refined formula for the average shortest path length, incorporating
  degree and clustering coefficients.
  """
  return (s['n'] * s['avg_clustering']) / (s['m'] + 1)
```

## neurips_matrix/map_elites_aspl_screen/seed_33

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 6
- Validation Spearman: 0.1917
- Test Spearman: 0.3824
- Best formula AST nodes: 32

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.000 -> 0.307
- MAP-Elites coverage: 0 -> 1 cells

### Acceptance Funnel

- Final generation attempted: 4
- Final generation accepted: 2
- Final generation acceptance rate: 0.500

### Repair Breakdown

- Repair attempts: 0
- Repair successes: 0
- Repair failures: 0

### Best Candidate Code

```python
def new_invariant(s):
  """
  Combines elements from the top 2 formulas into a new one.
  """
  return (s['n'] * s['avg_clustering'])**0.5 / (s['m'] + 1)
```

## neurips_matrix/small_data_aspl_train20_screen/seed_11

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 6
- Validation Spearman: 0.0521
- Test Spearman: -0.0911
- Best formula AST nodes: 29

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.000 -> 0.173
- MAP-Elites coverage: 0 -> 1 cells

### Acceptance Funnel

- Final generation attempted: 4
- Final generation accepted: 4
- Final generation acceptance rate: 1.000

### Repair Breakdown

- Repair attempts: 0
- Repair successes: 0
- Repair failures: 0

### Best Candidate Code

```python
def new_invariant(s):
  """
  A refined formula for average_shortest_path_length based on existing features.
  """
  return s['n'] * s['density'] / (s['m'] + 1)
```

## neurips_matrix/small_data_aspl_train20_screen/seed_22

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 5
- Validation Spearman: -0.3173
- Test Spearman: -0.3018
- Best formula AST nodes: 26

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.229 -> 0.229
- MAP-Elites coverage: 1 -> 1 cells

### Acceptance Funnel

- Final generation attempted: 4
- Final generation accepted: 2
- Final generation acceptance rate: 0.500

### Repair Breakdown

- Repair attempts: 0
- Repair successes: 0
- Repair failures: 0

### Best Candidate Code

```python
def new_invariant(s):
    """
    Combines elements from the top 2 formulas into a new one.
    Merge their strengths into a single improved formula.
    """
    return s['n'] * s['density'] * s['avg_clustering']
```

## neurips_matrix/small_data_aspl_train20_screen/seed_33

- Fitness mode: correlation
- Success: False
- Stop reason: early_stop
- Final generation: 5
- Validation Spearman: 0.2206
- Test Spearman: 0.2304
- Best formula AST nodes: 29

### Bounds Diagnostics

- test_bound_score: None
- test_mean_gap: None
- test_satisfaction_rate: None
- val_bound_score: None
- val_mean_gap: None
- val_satisfaction_rate: None
- Convergence: 0.231 -> 0.231
- MAP-Elites coverage: 1 -> 1 cells

### Acceptance Funnel

- Final generation attempted: 4
- Final generation accepted: 4
- Final generation acceptance rate: 1.000

### Repair Breakdown

- Repair attempts: 0
- Repair successes: 0
- Repair failures: 0

### Best Candidate Code

```python
def new_invariant(s):
  """
  A refined formula for the average shortest path length, incorporating
  degree and clustering coefficients.
  """
  return (s['n'] * s['avg_clustering']) / (s['m'] + 1)
```

## neurips_matrix/upper_bound_aspl_screen/seed_11

- Fitness mode: upper_bound
- Success: False
- Stop reason: early_stop
- Final generation: 5
- Validation Spearman: 0.0690
- Test Spearman: 0.1828
- Best formula AST nodes: 22

### Bounds Diagnostics

- candidate_mean_gap_max: 50.18619716329207
- candidate_mean_gap_min: 30.275319347600064
- candidate_satisfaction_max: 1.0
- candidate_satisfaction_min: 0.9125
- test_bound_score: 0.01837111691375646
- test_mean_gap: 53.43327178714925
- test_satisfaction_rate: 1.0
- val_bound_score: 0.019536516784199494
- val_mean_gap: 50.18619716329207
- val_satisfaction_rate: 1.0
- Convergence: 0.159 -> 0.159

### Acceptance Funnel

- Final generation attempted: 4
- Final generation accepted: 4
- Final generation acceptance rate: 1.000

### Repair Breakdown

- Repair attempts: 0
- Repair successes: 0
- Repair failures: 0

### Best Candidate Code

```python
def new_invariant(s):
  """
  A tighter bound for average shortest path length based on node count and density.
  """
  return s['n'] / (s['density'] + 1)
```

## neurips_matrix/upper_bound_aspl_screen/seed_22

- Fitness mode: upper_bound
- Success: False
- Stop reason: early_stop
- Final generation: 6
- Validation Spearman: -0.2561
- Test Spearman: -0.3506
- Best formula AST nodes: 31

### Bounds Diagnostics

- candidate_mean_gap_max: 29.1951071527241
- candidate_mean_gap_min: 1.1918695832167403
- candidate_satisfaction_max: 1.0
- candidate_satisfaction_min: 0.8
- test_bound_score: 0.3693804652824606
- test_mean_gap: 1.199629044753873
- test_satisfaction_rate: 0.8125
- val_bound_score: 0.36498521906852566
- val_mean_gap: 1.1918695832167403
- val_satisfaction_rate: 0.8
- Convergence: 0.000 -> 0.263

### Acceptance Funnel

- Final generation attempted: 4
- Final generation accepted: 4
- Final generation acceptance rate: 1.000

### Repair Breakdown

- Repair attempts: 0
- Repair successes: 0
- Repair failures: 0

### Best Candidate Code

```python
def new_invariant(s):
  """
  A tighter bound for average_shortest_path_length based on graph density and node count.
  """
  return (s['density'] + 1) * (s['n'] / 2)**(1/3)
```

## neurips_matrix/upper_bound_aspl_screen/seed_33

- Fitness mode: upper_bound
- Success: False
- Stop reason: max_generations_reached
- Final generation: 6
- Validation Spearman: -0.0286
- Test Spearman: -0.0326
- Best formula AST nodes: 28

### Bounds Diagnostics

- candidate_mean_gap_max: 367.81597483511297
- candidate_mean_gap_min: 55.97291133072556
- candidate_satisfaction_max: 1.0
- candidate_satisfaction_min: 1.0
- test_bound_score: 0.017036710497676397
- test_mean_gap: 57.696777182214134
- test_satisfaction_rate: 1.0
- val_bound_score: 0.017552201153896427
- val_mean_gap: 55.97291133072556
- val_satisfaction_rate: 1.0
- Convergence: 0.000 -> 0.156

### Acceptance Funnel

- Final generation attempted: 4
- Final generation accepted: 3
- Final generation acceptance rate: 0.750

### Repair Breakdown

- Repair attempts: 0
- Repair successes: 0
- Repair failures: 0

### Best Candidate Code

```python
def new_invariant(s):
  return s['n'] * (1 + s['avg_clustering'])**(-1/3)
```

