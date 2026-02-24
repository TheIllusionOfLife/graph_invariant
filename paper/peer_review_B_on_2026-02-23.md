
# Peer review (researcher-style) — *LLM-Driven Discovery of Interpretable Graph Invariants via Island-Model Evolution*

**Overall score:** **6 / 10 (Weak Accept)**  
**Confidence:** **3 / 5** (paper is clear and experiments are fairly complete, but I can’t validate the underlying code/artifacts here)

## Summary
The paper proposes an open-source framework that uses LLM-driven evolutionary search (four-island setup with heterogeneous prompting/temperatures) plus a MAP-Elites archive to discover **interpretable closed-form formulas** over a fixed set of pre-computed graph features/invariants. Candidates are evaluated in a sandbox with static checks and resource limits, scored via a composite objective (Spearman correlation + simplicity + novelty relative to known invariants), and optionally repaired through an LLM self-correction loop. Experiments span multiple synthetic graph generators, include out-of-distribution (OOD) evaluation, and compare against linear regression, random forest, and PySR.

## Strengths
- **Clear problem framing and positioning.** The paper is explicit about the interpretability goal and how it differs from (i) opaque GNN predictors and (ii) accuracy-only symbolic regression.
- **System design is coherent and reasonably well-motivated.** The island model (refinement/combination vs novelty), MAP-Elites diversity maintenance, and self-correction are complementary mechanisms rather than a grab-bag of tricks.
- **Safety-aware evaluation loop.** The sandboxing + static analysis + resource limits are good engineering practice for executing LLM-generated code.
- **Evaluation is multi-faceted.** You report (a) in-distribution validation/test, (b) OOD categories (large random, extreme params, special topologies), and (c) multi-seed variability for a benchmark setting.
- **Honest reporting of limitations and tradeoffs.** Results acknowledge that accuracy trails strong baselines on ASPL and that diversity/novelty objectives can reduce peak correlation.

## Weaknesses / concerns
- **Incremental empirical performance vs baselines (for ASPL).** On the headline ASPL setting, the reported test Spearman for the best LLM run (0.947) trails PySR and linear regression (both 0.975) and is close to random forest (0.951). As written, the primary value is interpretability and the discovery mechanism itself, not predictive performance.
- **Interpretability claims could be more operationalized.** The paper asserts “interpretable” but mostly measures simplicity via AST node count. This may not align with human interpretability (e.g., nested min/max, piecewise behavior, or ad-hoc exponents can be hard to reason about even if the AST is short). Some qualitative examples are included, but a more systematic interpretability evaluation would strengthen the claims.
- **Novelty metric may conflate “not strongly correlated with a short list” with “novel.”** The bootstrap CI gate over 13 known invariants is reasonable, but it may reject genuinely useful formulas that correlate with a known invariant for principled reasons, and it may accept “novelty” that is just noise on limited graph families.
- **Coverage of MAP-Elites archive seems modest.** Reported coverage reaches 5/25 cells (20%). It’s not obvious whether MAP-Elites is really contributing substantially beyond island heterogeneity, or whether the archive discretization/axes need tuning.
- **OOD degradation on special topologies is a red flag for “invariant discovery.”** For the MAP-Elites ASPL formula, OOD $\rho$ drops to 0.500 on special topologies (barbell/grid/etc.). This suggests the approach is learning a distribution-specific proxy rather than a broadly valid structural relationship—fine for predictive modeling, but less compelling for “invariant” discovery unless scoped carefully.
- **Limited ablations / attribution.** The paper narrates why each component matters (islands, MAP-Elites, self-correction), but there is not a clean ablation table quantifying marginal contributions (e.g., islands-only vs islands+MAP-Elites, with/without novelty gate, with/without self-correction) under matched compute.

## Questions for the authors
1. **What is the compute budget comparison vs PySR?** PySR is given a 60s timeout (and other settings), but the LLM system likely uses substantially more wall-clock time. A compute-normalized comparison (or at least total GPU/CPU-hours) would help interpret results.
2. **Do discovered formulas rediscover known approximations/bounds?** For ASPL and algebraic connectivity there are classical inequalities/approximations. A direct comparison (symbolic simplification + discussion) could strengthen the “mathematical insight” narrative.
3. **How sensitive are results to the feature set?** Since candidates only compose precomputed features, the feature dictionary effectively defines the hypothesis class. What happens if you remove key features (e.g., density, clustering, degree sequence stats)?
4. **What is the failure/repair breakdown?** Self-correction repairs 41–48% of failures, but it would be useful to know the distribution across syntax errors vs sandbox violations vs timeouts vs numerical issues.

## Suggestions (actionable)
- Add a **compute/efficiency section**: wall-clock per experiment, number of LLM calls, acceptance rate over time, and a compute-normalized baseline comparison.
- Provide an **ablation study** with at least: (i) no MAP-Elites, (ii) no novelty bonus/gate, (iii) no self-correction, (iv) single-island vs multi-island—each with matched budgets/seeds.
- Strengthen interpretability evidence: e.g., **human rating** on 20 formulas, or objective proxies like symbolic simplification length, number of distinct input features used, monotonicity checks, or stability under perturbations.
- Clarify scope: call the outputs “**interpretable predictors over a feature set**” rather than “graph invariants” unless you explicitly argue invariance to labeling and discuss whether the composed features are themselves invariants (they mostly are, but some implementation details can matter).
- For bounds mode: add **stress tests on adversarial / extremal graphs** (within computational feasibility) and report the tightness/satisfaction tradeoff more granularly (e.g., satisfaction vs gap quantiles).

## Score justification (NeurIPS-style)
I’d lean **weak accept** because the paper offers a solid, reproducible system contribution that combines multiple search/diversity mechanisms in a principled way and evaluates across several regimes with OOD checks. However, the predictive performance is not better than strong baselines on the main target, the interpretability and novelty claims are not yet rigorously quantified, and component-wise attribution is underdeveloped.

