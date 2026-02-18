# NeurIPS Readiness Status (as of 2026-02-18)

## Executive Verdict

The project is now a strong and improving systems/research prototype, but it is **not yet NeurIPS-acceptance ready**.  
Current status is best described as: **interesting systems paper with meaningful infrastructure progress, but major scientific-evidence gaps remain open**.

## Evidence Base

- Peer review A: `paper/peer_review_A_on_2026-02-18.md`
- Peer review B: `paper/peer_review_B_on_2026-02-18.md`
- Recent merged PRs: #34, #33, #32, #31, #30, #29
  - #34: https://github.com/TheIllusionOfLife/graph_invariant/pull/34
  - #33: https://github.com/TheIllusionOfLife/graph_invariant/pull/33
  - #32: https://github.com/TheIllusionOfLife/graph_invariant/pull/32
  - #31: https://github.com/TheIllusionOfLife/graph_invariant/pull/31
  - #30: https://github.com/TheIllusionOfLife/graph_invariant/pull/30
  - #29: https://github.com/TheIllusionOfLife/graph_invariant/pull/29

## Reviewer Concern Taxonomy

The two reviews converge on these high-impact concerns:

1. Novelty framing risk
- “new invariant discovery” is overstated when formulas are compositions over pre-computed feature dictionaries (`paper/peer_review_A_on_2026-02-18.md:98`, `paper/peer_review_B_on_2026-02-18.md:17`).

2. Baseline competitiveness risk
- LLM does not beat strongest baselines in reported ASPL setting (`paper/peer_review_A_on_2026-02-18.md:125`, `paper/peer_review_B_on_2026-02-18.md:23`).

3. Experimental scale/statistical power risk
- Scale appears small for NeurIPS-level empirical confidence (`paper/peer_review_A_on_2026-02-18.md:155`).

4. Theory/validity risk
- Claims around “discovery” and “provably valid” bounds are not yet supported by proofs and failure diagnostics (`paper/peer_review_A_on_2026-02-18.md:170`, `paper/peer_review_B_on_2026-02-18.md:57`).

5. Calibration/ablation/reproducibility detail risk
- Need sensitivity/ablation analysis and stronger determinism/prompt/model detail (`paper/peer_review_B_on_2026-02-18.md:30`, `paper/peer_review_B_on_2026-02-18.md:67`).

## PR Impact Ledger (#34 to #29)

| PR | Main Change | Impact Type | Reviewer Gap Coverage |
|---|---|---|---|
| #34 | Spectral feature pack, dual MAP-Elites, OOD topology controls, data policy, test expansion | Enabler (major) | Partial support for scale/analysis readiness; no direct closure of core novelty/theory gaps |
| #33 | Sandbox DoS hardening, AST/recursion/resource-limit handling | Direct fix | Improves execution safety credibility, not scientific claim strength |
| #32 | Cache safe numpy namespace in sandbox | Enabler (perf) | Better throughput for larger runs; no direct scientific closure |
| #31 | Block sandbox escape via generator frame introspection | Direct fix | Security robustness, not claim/evidence closure |
| #30 | Extract `_handle_rejection` for maintainability | No direct impact | Code quality improvement only |
| #29 | Replace fragile forbidden-pattern checks with AST-based guardrails | Direct fix | Security/reliability robustness, not core acceptance blockers |

## NeurIPS Gap Matrix

| Area | Reviewer Concern | Current Evidence | Status | Acceptance Risk | Concrete Next Action | Priority |
|---|---|---|---|---|---|---|
| Framing/claims | “Invariant discovery” claim exceeds evidence when using fixed feature dictionary | A/B both raise this directly | Open | Critical | Reframe title/abstract/method claims to “interpretable compositions/approximations over known invariants,” and explicitly separate composition novelty from structural-information novelty | P0 |
| Baselines | LLM not clearly superior on key benchmark | Review A baseline table, Review B leakage concern | Open | Critical | Add targeted regimes where LLM should win (small-data, constrained search, noisy targets, nonlinearity), with pre-registered success criteria | P0 |
| Scale/power | Dataset and run scale too small | Review A asks for 500-1000 train and broader settings | Open | Critical | Execute full matrix with multiple seeds per key setup; report variance/CIs and stability | P0 |
| Theory/bounds validity | “Provably valid” language not justified by empirical pass rate | Review B upper-bound critique + Review A proof gap | Open | Critical | Replace over-claiming text now; add theorem-backed subclasses or explicit empirical-only framing + counterexamples/failure analysis | P0 |
| Objective calibration | Novelty gate and alpha/beta/gamma sensitivity unclear | Review B novelty/objective concerns | Open | High | Add sensitivity sweeps for novelty threshold and objective weights, include Pareto tradeoff tables/plots | P1 |
| MAP-Elites evidence | Coverage low and interpretation underdeveloped | Review B coverage critique | Partial | High | Run matched-budget ablation MAP-Elites vs no-MAP-Elites across seeds; add descriptor diagnostics | P1 |
| Reproducibility package | Exact model/runtime/prompt determinism details incomplete | Review B reproducibility request | Partial | High | Document full model identifier/version, decoding params, seeds, prompt templates, and determinism caveats | P1 |
| Security/execution robustness | Sandbox attack and DoS vectors | PR #29/#31/#33 merged | Resolved (for current known vectors) | Medium | Keep regression tests and threat-model notes in appendix | P2 |
| Data governance and archival policy | Need DOI-backed reproducibility artifacts | PR #34 added policy/docs | Partial | High | Complete Zenodo publish + DOI sync in paper/docs | P0 |

## Post-Merge Recommendation Status (from PR #34)

PR #34 included five explicit post-merge recommendations. Current status:

1. Lock release point via tag
- Status: **Not started**
- Evidence: no tags present (`git tag --list` -> `tag_count=0`).

2. Execute long experiments (full matrix, ~5 seeds)
- Status: **Not started / not evidenced**
- Evidence: no explicit run log update tied to post-merge checklist; existing figure/analysis outputs are timestamped earlier than latest merge window.

3. Publish raw data to Zenodo with DOI/checksum metadata
- Status: **Partial**
- Evidence:
  - policy exists (`docs/DATA_POLICY.md:21`)
  - README references DOI expectations (`README.md:147`)
  - actual dataset citation still placeholder (`paper/references.bib:114`).

4. Regenerate paper outputs from archived data
- Status: **Not started / not evidenced**
- Evidence: report/figure artifacts currently appear pre-checklist and no archived-data regeneration trail is recorded.

5. Final paper sync pass (DOI consistency across docs/paper)
- Status: **Not started**
- Evidence: DOI placeholders remain (`paper/references.bib:118`).

## Current Project Status Summary

### What improved substantially

- Infrastructure maturity improved sharply in one day:
  - security hardening and sandbox robustness (PR #29/#31/#33)
  - runtime scaling enablers and expanded experiment architecture (PR #32/#34)
  - governance scaffolding for research-grade archival/citation (PR #34)

### What remains the acceptance bottleneck

- The central scientific questions from both reviews remain largely open:
  - claim framing vs evidence
  - stronger-than-baseline positioning in at least one regime
  - larger and more rigorous empirical evaluation
  - proof-level or carefully bounded empirical claims for “discovery” and bounds validity

## Experiment Efficiency and Comparison Update

Input from current project operation:
- A PR #18 long experiment reportedly required about **48 hours**.

### Comparison: Previous Long-Run Style vs Planned NeurIPS Evidence Runs

| Item | Previous long-run style (PR #18 scale) | Planned NeurIPS-ready program |
|---|---|---|
| Primary objective | Obtain one strong end-to-end run | Build acceptance-grade evidence across claims |
| Run design | Single heavy fixed-budget run | Staged program (pilot -> filter -> finalist long runs) |
| Runtime use | High wall-clock per configuration | Compute concentrated on finalists only |
| Statistical strength | Limited if few seeds/configs | Multi-seed variance and confidence reporting |
| Reviewer coverage | Partial | Directly addresses ablation/sensitivity/robustness concerns |
| Expected value per compute hour | Moderate | Higher (more decisions informed per hour) |

### Efficiency Improvement Strategy (Required Before Full Resubmission Matrix)

Priority order:

1. Algorithm/runtime refinement (first)
- Profile hotspot components (sandbox eval path, feature extraction, scoring, archive insertion, model call path).
- Reduce wasted compute via staged candidate evaluation and stronger early rejection.
- Cache or reuse expensive intermediate results where deterministically valid.

2. Parameter optimization (second)
- Run small design-of-experiments sweeps on population, generations, migration interval, and archive bins.
- Optimize for quality-per-hour rather than raw absolute score in one run.
- Promote only promising configs to expensive long runs.

3. Cloud compute scaling (third)
- Use after local algorithmic efficiency is improved and configs are tuned.
- Apply mainly to parallel seeds/configs to accelerate statistical evidence collection.
- Keep pinned environment/model settings for reproducibility parity.

### Operational Decision Rule

- Do **not** use 48-hour-scale runs as the default.
- Reserve long runs for final shortlisted configurations.
- Gate progression with staged criteria:
  - Stage 1: short pilot runs for pruning,
  - Stage 2: short multi-seed stability checks,
  - Stage 3: full-duration runs only for finalists included in paper tables.

### Experiment Kill/Promotion Policy (Value-First)

Goal:
- Accept long-running (48h+) experiments when they are decision-critical and likely valuable.
- Avoid consuming iteration budget on runs that cannot change submission decisions.

Pre-run value contract (required before launch):

1. Decision target
- The run must map to one explicit paper decision:
  - claims framing evidence,
  - baseline-advantage claim,
  - ablation/sensitivity claim,
  - final reported number.

2. Success gate
- Define one primary gate and one fallback gate before launch.
- Recommended default gates for correlation tasks:
  - Primary: test Spearman rho improves by >= 0.01 over current best non-LLM baseline in target regime.
  - Fallback: same-level rho with materially lower complexity or better OOD stability.

3. Stop-if-useless condition
- If the run cannot satisfy either gate even in best-case continuation, terminate.

Checkpoint schedule (for each long run):

1. 10% budget checkpoint
- Required: validation trend slope positive and no major instability.
- Kill if:
  - metric trend is flat/negative across recent windows, and
  - novelty/diversity signal is collapsing.

2. 25% budget checkpoint
- Required: at least 50% of target gap to primary gate is closed.
- Kill if:
  - progress < 30% of required gap and no compensating gains in simplicity/OOD.

3. 50% budget checkpoint
- Required: >= 80% of target gap closed or fallback gate likely.
- Kill if:
  - still < 60% of target gap and confidence interval overlap indicates low chance of meaningful win.

Promotion rules:

1. Stage 1 -> Stage 2
- Promote config only if pilot run hits 25% checkpoint requirements.

2. Stage 2 -> Stage 3 (full 48h+)
- Promote only if short multi-seed evidence (recommended 3 seeds) shows:
  - median improvement direction consistent, and
  - no catastrophic seed variance.

3. Stage 3 -> final paper table
- Accept only if full-run metrics satisfy pre-registered gate and are reproducible on confirmatory rerun/seed set.

Minimum experiment portfolio per iteration cycle:

1. Screening bucket
- 6-12 short runs (cheap) for parameter/algorithm candidates.

2. Stability bucket
- Top 2-3 configs with short multi-seed checks.

3. Finalist bucket
- 1-2 long runs (48h+) reserved for claim-critical configs.

Timeline guardrail (from 2026-02-18 to likely May 2026 deadline):

1. Cycle 1 (weeks 1-3)
- Build calibrated checkpoints and prune weak configs aggressively.

2. Cycle 2 (weeks 4-7)
- Run finalist long experiments for core claims.

3. Cycle 3 (weeks 8-10)
- Confirmatory reruns, figure/table lock, and paper text synchronization.

Hard rule:
- No new long run starts unless it has a written pre-run value contract and checkpoint thresholds.

## Prioritized Resubmission Roadmap

### P0 (must close before submission)

1. Claim reframing patch
- Remove/soften over-claims in abstract, intro, and bounds language.
- Explicitly define “composition discovery” scope.

2. Full experiment matrix with variance
- ASPL correlation, algebraic connectivity, bounds mode, dual MAP-Elites.
- Multi-seed statistics and confidence intervals.

3. Competitive differentiation experiment
- Add at least one task regime where LLM method is clearly preferable to PySR/Linear under agreed metric.

4. Zenodo publication completion
- Publish artifacts, add DOI, checksums, and version metadata; then synchronize all docs/paper citations.

### P1 (high leverage)

1. Sensitivity and ablation package
- Feature ablations
- Novelty threshold sensitivity
- Objective weight sensitivity
- MAP-Elites ablation under matched compute

2. Bounds diagnostics
- Enumerate failures and where bound expressions violate expectations.
- Restrict claim domain where validity can be defended.

3. Reproducibility appendix completion
- Full prompt templates and model serving details.
- Seeded rerun stability discussion.

### P2 (polish)

1. Improve formula transparency
- Publish full discovered formulas (not ellipsized excerpts).

2. Strengthen topology/OOD reporting detail
- Explicit category counts and split composition.

## Decision-Complete Next Sprint Definition

If one sprint is available before submission, use this implementation order:

1. Paper claim reframing edits (fast, immediate risk reduction).
2. Run long experiment matrix with 5 seeds for key configs.
3. Run ablation/sensitivity bundle.
4. Publish Zenodo dataset and update `paper/references.bib`.
5. Regenerate figures/tables from archived data and finalize consistency pass.

## Assumptions

- This status reflects repository and PR state available on 2026-02-18.
- No external unpublished experiment logs were considered.
- “Recent PRs” was interpreted as last six merged PRs (#34 to #29).
