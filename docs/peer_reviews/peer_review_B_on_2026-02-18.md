# Review: *LLM-Driven Discovery of Interpretable Graph Invariants via Island-Model Evolution*

Date: 2026-02-18  
Scope: Technical clarity, experimental design, claims vs. evidence, reproducibility, and presentation.

## Summary
The paper presents an open-source framework that uses LLM-driven evolutionary search (four islands with heterogeneous prompting/temperatures), MAP-Elites quality-diversity archiving, a composite objective (accuracy + simplicity + novelty), and an LLM-based self-correction loop to discover interpretable closed-form graph formulas that correlate with targets such as average shortest path length (ASPL) and algebraic connectivity, and to find upper bounds for ASPL. Results indicate competitive performance with baselines (PySR, random forest, linear regression) while emphasizing interpretability and novelty.

## Strengths
- **Clear system decomposition.** The method section is well-structured (dataset → island evolution → sandbox → scoring → archive → migration → self-correction) and conveys the pipeline end-to-end.
- **Interpretability focus is explicit.** The paper consistently motivates why closed-form formulas are valuable compared to black-box predictors (e.g., random forests, GNNs).
- **Diversity mechanisms are thoughtfully integrated.** Island-model heterogeneity + MAP-Elites is a coherent approach for exploring a large, rugged search space.
- **Security-conscious evaluation design.** The sandbox + static checks + resource limits are appropriate and described concretely.
- **Multiple evaluation modes.** Including correlation-mode, bounds-mode, and OOD evaluation makes the experimental story richer than “just fit the data.”

## Major concerns / questions (substantive)
### 1) “Invariant” vs “predictor of an invariant”
The system discovers formulas over **pre-computed features** (e.g., density, degree stats, clustering, triangles). These features are themselves graph invariants, but the discovered formula is therefore a *function of a chosen feature set* rather than operating directly on the graph structure.
- If the intention is “discover new graph invariants,” it would help to sharpen terminology: the method is discovering **compositions of known invariants/features** (which is still valuable), but it is not discovering invariants that depend on information not already in the feature dictionary.
- Suggest clarifying early (abstract/introduction) that the novelty is in the *functional form/composition* and in the automated search procedure, not in accessing new graph information.

### 2) Potential leakage / target dependence via the feature set
- The feature dictionary includes several properties that may correlate strongly with ASPL (e.g., density, degree moments, clustering/transitivity). This is fine, but it means the problem can become “learn ASPL from standard graph statistics.” The fact that **linear regression performs extremely well** (reported test $\rho=0.975$) suggests the feature set nearly linearizes the target.
- Because the strongest baselines already do very well, it becomes harder to argue that the method discovers qualitatively new insight rather than a rearrangement of known correlates.

What would strengthen the claim:
- An ablation showing performance when restricting the feature set (e.g., degrees-only; without clustering; without triangles; etc.).
- Reporting the final discovered formulas explicitly (not just “key components … $\ldots$”) so readers can judge novelty/insight.

### 3) Novelty metric: definition and calibration
- The novelty score is based on a bootstrap CI upper bound of correlation with a set of 13 known invariants; the “novelty gate” threshold is $\theta_{\text{gate}}=0.15$.
- This is a reasonable idea, but novelty-by-(low correlation) can:
  1) penalize formulas that are genuinely novel but happen to correlate with known invariants on the sampled graph distribution, and  
  2) reward formulas that are simply noisy or distribution-specific.

Questions:
- How sensitive are results to $\theta_{\text{gate}}$ and to the list of 13 invariants?
- Are those invariants computed on the same graphs and pre-processing as the candidate outputs? (It sounds like yes, but the paper would benefit from spelling it out.)
- Do you compute novelty on validation graphs only, or train+val, and could that induce overfitting to the novelty test itself?

### 4) Composite score vs. reported success criteria
- The paper reports success thresholds in terms of Spearman $\rho$ (e.g., $\rho \ge 0.85$), but search is performed using a weighted composite objective with novelty and simplicity terms.
- Since baselines are reported in terms of $\rho$, the paper could better justify the chosen weights $(\alpha,\beta,\gamma)=(0.6,0.2,0.2)$ and how they affect the final $\rho$ vs interpretability tradeoff.

What would strengthen:
- A Pareto-style plot or table: best $\rho$ at varying complexity levels; or show the best formula under (i) $\alpha=1$ and (ii) the composite objective.
- A brief sensitivity analysis for the weights.

### 5) MAP-Elites coverage seems low
- Archive coverage grows from 2 to 5 cells out of 25 (20%) over 30 generations. That’s modest and raises questions:
  - Are the behavioral descriptors (simplicity, novelty bonus) too coarse or too strict?
  - Is the search budget too small (population sizes, generation counts)?
  - Does the novelty gate filter out too many candidates early, causing sparse coverage?

It may still help performance, but it would be good to add interpretation: why coverage is low, whether it matters, and whether richer descriptors would improve exploration.

### 6) Bounds mode: validity and evaluation
- The upper-bound ASPL experiment reports an 87% satisfaction rate and suggests “provably valid inequality for most graph families,” but bounds mode (as described) does not prove validity; it empirically checks on sampled graphs.
- The “Best Discovered Formulas” table suggests combining known bounds (e.g., path-graph bound $(n+1)/3$, Moore-bound arguments) with a min operator. This is plausible, but:
  - If the expression is a **minimum of bounds**, it is *not necessarily* an upper bound unless each constituent term is itself a valid upper bound for all graphs in scope.
  - Empirical satisfaction < 100% suggests it is not universally valid (or the evaluation is approximate / has edge cases).

Recommendation:
- Be precise: “empirically valid on X% of graphs in our evaluation set,” unless you provide a proof or restrict to a class of graphs where each term is known to be a valid bound.
- Include examples/counterexamples where it fails and discuss why.

### 7) Reproducibility details: model, prompts, and determinism
- The paper states a local model `gpt-oss:20b` via Ollama; good for reproducibility, but readers will want:
  - The exact model identifier/version, decoding parameters beyond temperature (top-p, repetition penalties, max tokens), and prompt templates.
  - How randomness is controlled (seeds for dataset + LLM sampling + evolutionary operations).
  - Whether results are stable across reruns with the *same* seed (LLM sampling may introduce nondeterminism depending on serving stack).

## Minor issues / presentation suggestions
- **Provide the actual discovered formulas.** Currently many are shown as partial expressions with ellipses; at least the best final formulas per experiment should be included in full (possibly in appendix/supplement).
- **Clarify feature count consistency.** Method lists many features; experiments mention “12 features excluding the target.” A short table listing all features used would remove ambiguity.
- **Explain novelty invariants list.** The novelty test references “13 known invariants” (diameter, radius, Wiener index, spectral radius, algebraic connectivity, etc.). A bulleted list (and how each is computed) would help.
- **OOD “special topology” set.** The description includes deterministic graphs and a few named real-world graphs. It would help to report the number of graphs in this category and whether the correlation is computed over a mixture of sizes and types.
- **Interpretation claims.** When stating that the upper-bound formula uses Moore-bound arguments or that components are “provably valid,” consider either citing known theorems or rephrasing as “inspired by.”

## Suggested additional experiments (high impact)
1) **Feature ablations**: remove clustering/transitivity, remove triangle-based features, degrees-only, etc. Report effect on LLM vs baselines.
2) **Weight sensitivity**: vary $(\alpha,\beta,\gamma)$ and show how interpretability and $\rho$ trade off.
3) **MAP-Elites vs no MAP-Elites** under identical compute budget, with multiple seeds (not just one MAP-Elites run vs multi-seed non-MAP-Elites).
4) **Hold-out families**: train on 4 generative families and test on the 5th to test genuine distribution shift.
5) **Bounds mode diagnostics**: show failure cases; show which constituent bound terms cause violations; optionally restrict to graph classes where validity is known.

## Overall assessment
This is a strong systems paper with a coherent integration of LLM-driven search, diversity mechanisms, and safety-aware execution, and it is well-written. The biggest gap is the conceptual framing around “discovering invariants”: since formulas are built from a fixed feature dictionary, the work is best understood as discovering **interpretable compositions/approximations** (and sometimes empirical bounds) of graph properties. Tightening this framing and adding a few targeted ablations/sensitivity analyses would substantially strengthen the paper’s scientific claims.