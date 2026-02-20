# Day1 Screening Selection (2026-02-20)

| Config | Seeds | Val Spearman mean±std | Test Spearman mean±std | Val Bound Score mean±std | Val Satisfaction mean±std |
|---|---:|---|---|---|---|
| map_elites_aspl_screen | 3 | 0.0760 ± 0.2923 | 0.1420 ± 0.4026 | N/A ± N/A | N/A ± N/A |
| upper_bound_aspl_screen | 3 | -0.0719 ± 0.1362 | -0.0668 ± 0.2191 | 0.1340 ± 0.1633 | 0.9333 ± 0.0943 |
| small_data_aspl_train20_screen | 3 | -0.0149 ± 0.2246 | -0.0542 ± 0.2188 | N/A ± N/A | N/A ± N/A |

## Promotion Decision
- Promote `map_elites_aspl` and `small_data_aspl_train20` to medium confirmatory runs (2 seeds) for same-day delivery.
- Keep `upper_bound_aspl` screening results for bounds diagnostics in this PR; full confirmatory bounds reruns deferred to next cycle.
