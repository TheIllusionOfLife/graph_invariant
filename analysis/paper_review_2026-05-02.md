# Pre-Submission Peer Review of NeurIPS 2026 Harmony Paper

**Reviewer**: internal multi-perspective audit
**Date**: 2026-05-02
**Submission target**: NeurIPS 2026 main track, Concept-and-Feasibility tag
**Submission deadline**: 2026-05-06 AOE (full paper + supplementary)
**Method**: 8 specialised subagent reviewers (3 adversarial substance + 5 hygiene), synthesised here

## Executive summary

**Overall verdict**: the paper is **mechanically clean but rhetorically over-extended**. Internal consistency is strong (zero numerical or qualitative drift across sections). The P0 issues are entirely about **substance, framing, and disclosure** — not data integrity. With the P0 fixes applied, this is a defensible Concept-and-Feasibility submission. Without them, an adversarial reviewer can credibly attack the contribution as "an under-validated empirical paper with careful wording".

The most damaging single finding is **not** a flagged risk but a discovery from the appendix audit: Table 5 (factor decomposition) shows that the Full pipeline (LLM + Harmony + MAP-Elites) **underperforms Harmony-only** (random proposer + MAP-Elites) on Astronomy (0.000 vs 0.054) and Materials (0.001 vs 0.029). The intro and conclusion credit the Harmony metric as a primary contribution, but the evidence isolates MAP-Elites archiving as the actual driver and shows the LLM + Harmony combination can be neutral or harmful. This needs honest reframing.

**Counts**: 6 P0, 17 P1, 3 P2 *(after Codex meta-review recalibration; original counts were 8 P0 / 12 P1 / 3 P2 — see "Process notes")*.

## P0 — must fix before submission

### P0-1. Bait-and-switch on contribution attribution
**Where**: `paper/sections/introduction.tex:42-58`, `paper/sections/conclusion.tex:9-15`, `paper/sections/abstract.tex:5-13`
**Convergent flag**: substance C&F-Fit + Novelty + hygiene Logic+Math (3 independent agents)

The abstract and intro market-test the Harmony metric as a primary contribution. The factor decomposition (appendix Table 5, lines 470-556) shows:
- **Harmony-only** (random proposer + MAP-Elites): Astronomy 0.054, Physics 0.032, Materials 0.029, Wikidata Physics 0.002, Wikidata Materials 0.019
- **Full** (LLM + Harmony + MAP-Elites): Astronomy 0.000, Physics 0.028, Materials 0.001, Wikidata Physics ≈ 0.002, Wikidata Materials 0.015

The Full pipeline is **neutral or worse than Harmony-only on 4 of 5 domains** (Physics is the only domain where Full ≈ Harmony-only). The intro contribution 4 (line 56-58) acknowledges "MAP-Elites is the dominant driver" but the surrounding contributions (1-3) still position the metric and LLM as primary.

**Suggested change**: rewrite intro contributions list to lead with "(1) demonstration that quality-diversity archiving is feasible for KG completion under typed-relation constraints; (2) factor-decomposition methodology that attributes structural gain to the archive vs the metric vs the proposer; (3) the Harmony metric and proposal schema as designed-in components whose marginal empirical contribution is small but whose semantic contribution (falsifiable claims, justifications) is qualitative". The current ordering markets exactly the components that the evidence demotes.

**Severity**: P0 because it is the single attack a hostile reviewer is most likely to use, and the evidence to fix it is already in the appendix — this is a framing edit, not a re-experiment.

### P0-2. Calibration gate transfer failure undisclosed
**Where**: `paper/sections/results.tex:120`, `paper/sections/abstract.tex` (silent), `paper/sections/discussion.tex` (silent)
**Convergent flag**: substance Null-Result + hygiene Empirical + hygiene Honesty (3 independent agents)

The pre-registered calibration gate (`analysis/calibration_gate.md`) requires "Harmony ≥ 10% over frequency, 95% CI lower bound > frequency mean, on ≥ 2 domains". The gate **passed** on the two calibration domains (linear_algebra +31%, periodic_table +65%). The five discovery domains in Table 1 show no such margin — Hits@10 is within 0.02 of DistMult on all three "directional improvement" domains and worse than DistMult on two.

The paper mentions the gate exactly once, in `results.tex:120`: "The calibration gate passed on both calibration domains (31--65% over frequency)". It never says **the gate did not transfer to the discovery domains**.

**Suggested change**: add one explicit paragraph in `results.tex` immediately after line 124 stating "The calibration gate (Appendix~\ref{app:calibration-gate}) was pre-registered before live runs against the two calibration domains. It passed there but did not transfer to the discovery domains, where Harmony shows only directional, non-significant differences against DistMult (\S\ref{sec:results}, Table~\ref{tab:metrics}). We treat the calibration-vs-discovery gap as a regime-finding rather than a refutation: the gate's pass on hand-curated dense KGs and its failure on sparser discovery KGs is itself diagnostic of the method's regime of applicability." Mirror in `abstract.tex` with a single sentence.

**Severity**: P0. For a Concept-and-Feasibility paper, hiding a registered-hypothesis failure is the worst possible honesty issue — it converts a defensible "directional null" into "we cherry-picked our reporting".

### P0-NEW. LLM-guidance non-identifiability *(Codex meta-review addition)*
**Where**: `paper/sections/appendix.tex:470-556` (factor decomposition Table 5), `paper/sections/introduction.tex:42-58` (contributions list)
**Flag**: Codex meta-review

This goes deeper than P0-1 (framing). If random proposer + MAP-Elites achieves comparable or better gains than the Full pipeline (LLM + Harmony + MAP-Elites) on 4 of 5 domains, the **claim "LLM-guided theory discovery" may be wrong as a contribution claim**, not just mis-framed. P0-1 fixes the attribution narrative; this finding asks whether the LLM proposer's contribution is identifiable at all.

The honest contribution that survives the data is: "(1) Quality-diversity archiving for KG mutations is feasible; (2) the Harmony score provides interpretable selection pressure; (3) **the LLM proposer's contribution is qualitative (semantic structure, falsifiable claims) not quantitative (Hits@10 gain)**." The current paper title — "Harmony-Driven Theory Discovery in Knowledge Graphs via LLM-Guided Island Search" — name-checks LLM-guided as the methodology, when the evidence supports calling it semantic-overlay instead.

**Suggested change**: in `discussion.tex` after the limitations paragraph, add an "Identifiability of the LLM contribution" sub-paragraph: "The factor decomposition shows that random proposers in MAP-Elites achieve gains comparable to the LLM-guided variant on four of five domains. We interpret this as evidence that the LLM's quantitative contribution to link prediction is not statistically identifiable in our experimental design; its identifiable contribution is qualitative (the proposal schema's claim/justification/falsifiability fields, which random proposers cannot fill). This is consistent with the C&F framing: feasibility of LLM-as-proposer is demonstrated by valid_rate convergence (Section X) but not the magnitude of any link-prediction advantage." The paper title may also need to soften "LLM-Guided" to "LLM-Augmented" or similar — flag for user judgment.

**Severity**: P0 because it is the deeper version of P0-1 and a strong area chair will push to it. The fix can coexist with P0-1's reframing.

### P0-NEW2. No-QD ablation confounds grid vs archive *(elevated from P1-8 by Codex)*
**Where**: `paper/sections/appendix.tex:475-478` (No-QD = greedy single-bin archive)
**Flag**: hygiene Empirical (originally P1) + Codex meta-review elevation

The No-QD ablation simultaneously removes (a) the 5×5 MAP-Elites grid and (b) the diversity-based archive. The paper attributes "MAP-Elites is the dominant driver" (the headline claim from P0-1's reframing) to this ablation, but the design cannot isolate whether the grid or the archive bookkeeping carries the credit.

This was originally graded P1 because it is a methodological gap. Codex elevated it to P0 because it is the *direct empirical foundation* of P0-1's reframing — if "MAP-Elites is dominant" is going to become a primary claim, the ablation behind it must support a clean attribution.

**Suggested change**: either (a) add a fourth ablation condition (greedy-with-grid) and re-run on at least one domain to disambiguate, or (b) qualify the "MAP-Elites is dominant" claim everywhere it appears: "the MAP-Elites archive (combining grid structure with diversity bookkeeping) is the dominant driver; isolating their individual contributions is left for future work". Option (b) is the 4-day-deadline-realistic fix.

**Severity**: P0 because it is the empirical foundation of the now-primary contribution claim.

### P0-3 [DEMOTED to P1-13]. No multiple-testing correction declared
**Where**: `paper/sections/appendix.tex:181-203` (statistical methods), `paper/sections/results.tex:11-12` (p-values)
**Flag**: hygiene Empirical

The paper runs 5 discovery domains × 4 KGE baselines = **20 hypothesis tests** on Hits@10 (and another 20 on MRR). No Bonferroni, BH-FDR, or other family-wise error control is mentioned. Reported p-values (0.38–1.00) are uncorrected.

While the paper's main framing is "none reach significance", the absence of an explicit correction strategy is a methodological smell that an adversarial reviewer will flag as "you didn't even know you needed to correct".

**Suggested change**: add one paragraph to the statistical methods appendix: "We report uncorrected p-values to permit downstream meta-analysis. Under family-wise error control across the 20 Hits@10 tests (5 domains × 4 baselines), the Bonferroni-corrected α=0.05 threshold becomes 0.0025; under BH-FDR the q=0.05 threshold becomes ≈0.025 at the smallest observed p. None of our reported p-values clear either threshold, consistent with our reported lack of significance." Reference this paragraph from `results.tex:11`.

**Severity**: P0 because adding the paragraph costs nothing and pre-empts a reviewer attack that would otherwise have substance.

### P0-4. "Complementary value" claim needs effect-size grounding
**Where**: `paper/sections/abstract.tex:19-20`, `paper/sections/discussion.tex:61-62`
**Convergent flag**: hygiene Empirical + hygiene Honesty (2 agents)

The abstract states "Harmony adds complementary value to frequency-based approaches in denser, multi-type KGs". The largest effect supporting this is δ_C = +0.12 (Wikidata Materials). The paper itself (`appendix.tex:196-197`) establishes the convention that **|δ_C| < 0.147 = "negligible"** per Cliff (1993). So the claim "complementary value" rests on a *negligible* effect by the paper's own threshold.

This is the kind of claim a reviewer can dismantle with the paper's own appendix.

**Suggested change**: change `abstract.tex:19-20` to "On Wikidata Materials, Harmony shows the largest directional improvement over DistMult-alone (Hits@10 0.34 vs 0.32; δ_C = 0.12, in the 'negligible' range per Cliff 1993). We interpret this as evidence of regime-applicability, not effect strength." Mirror in `discussion.tex:61-62`.

**Severity**: P0 because the paper currently advertises a negligible effect as a contribution.

### P0-5. LLM-as-judge circularity not acknowledged
**Where**: `paper/sections/results.tex:114-117`, `paper/sections/discussion.tex:23-31`
**Convergent flag**: substance Null-Result + hygiene Honesty (2 agents)

The paper reports "Gemini 2.5 Pro rates top-5 proposals at 4.5/5 falsifiability, 4.6/5 clarity" prominently in results. It defers human expert evaluation to future work. A second LLM judging proposals from a first LLM (gpt-oss:20b) is methodologically circular — **independence of evaluator and generator is a foundational assumption that this design violates**.

The discussion section's "LLM dependence and safety" paragraph (lines 23-31) addresses single-model risk, but not judge-independence.

**Suggested change**: in `discussion.tex` after line 31, add: "A methodological caveat on our automated rubric scores (Table~\ref{tab:rubric-results}): we use Gemini 2.5 Pro as a judge for proposals generated by gpt-oss:20b. While the two models are from different developers and architectures, both are large autoregressive language models and may share systematic biases (e.g., preferring fluent, well-structured proposals over substantively novel ones). The reported rubric scores should therefore be read as inter-LLM agreement, not as a substitute for human expert evaluation, which we defer to future work." Reference this from `results.tex:114`.

**Severity**: P0 because the paper currently presents LLM-judged scores as quality evidence without the independence caveat. A single-sentence acknowledgment converts the issue from "missing limitation" to "honestly disclosed scope".

### P0-6 [DEMOTED to P1-14]. KG-size limitation not acknowledged
**Where**: `paper/sections/discussion.tex:48-70`, `paper/sections/appendix.tex:8-26` (Table 2 dataset sizes)
**Convergent flag**: hygiene Empirical + hygiene Honesty (2 agents)

Five of seven domains have ≤ 153 entities (hand-curated: 41–153 entities, 39–326 edges). The two Wikidata domains have 252-253 entities. **All seven domains are toy KGs** by production-KG standards (DBpedia, full Wikidata: millions of entities).

The discussion mentions "small KGs" only in the context of frequency dominance (line 59), never as a limitation of the experimental design itself.

**Suggested change**: add to the limitations paragraph in `discussion.tex` (after line 64): "Our seven evaluation domains span 41–253 entities and 39–814 edges. This range covers early-stage scientific KG construction but does not address the regime of production-scale KGs (10^5–10^7 entities, e.g., DBpedia, full Wikidata). Whether the directional gains we observe on Wikidata Materials would scale to its full 100M-entity superset is an open question; we expect frequency baselines to dominate further at scale and Harmony's relative position to depend on edge-type-vocabulary richness, which we do not characterise here."

**Severity**: P0 because "you only show toy KGs" is a routine reviewer attack, and pre-empting it with an explicit acknowledgment is essentially free.

### P0-7 [DEMOTED to P1-15, but kept "strong P1"]. Per-domain failure analysis is shallow on 4 of 5 domains
**Where**: `paper/sections/discussion.tex:17-21` (Physics is the only domain with mechanism analysis)
**Flag**: substance Null-Result

The discussion provides a plausible failure-mode hypothesis for Physics ("LLM proposes hub-entity edges → redundant"). For Astronomy, Materials, Wikidata Physics, and Wikidata Materials, the discussion provides no mechanism — only "regime characterisation" hand-waves.

For a Concept-and-Feasibility paper whose contribution rests on null/directional results being *informative*, shallow failure analysis is a contribution-killer.

**Suggested change**: add 2-4 sentences per domain in `discussion.tex` proposing a specific failure mechanism, even if speculative. The substance subagent suggested templates like:
- Astronomy: 7-type vocabulary too coarse for fine-grained relations (photometric vs spectroscopic parallax)
- Materials: only 8 test edges per seed (53 total × 20% mask) → noise floor exceeds effect
- Wikidata Physics: DistMult performance ceiling already at 0.25 → small headroom
- Wikidata Materials: directional improvement appears at the regime where vocabulary richness matches the metric design (this is the only "Harmony works as designed" data point)

These do not need to be definitively proven — they need to be *named*, so a follow-up researcher knows what to test.

**Severity**: P0 because the absence of mechanism analysis converts the paper from "informative null" to "inconclusive null".

### P0-8 [DEMOTED to P1-16]. Eq. 2 boundedness rests on undeclared assumption
**Where**: `paper/sections/method.tex:47-51` (Compressibility), `paper/sections/introduction.tex:42-44` (boundedness claim)
**Flag**: hygiene Logic+Math

Eq. 2 hardcodes `H(p)/log₂7` as the entropy term, and the paper claims "Harmony score is bounded in [0,1]" (intro contribution 1). For KGs that use fewer than 7 edge types (e.g., periodic_table uses 2 types), the entropy bound is `log₂2 < log₂7`, so `H(p)/log₂7 < 1` and Compressibility stays in [0,1] — but **only because unused edge types contribute zero entropy, not because the formula is correct in principle**. The bound holds "by accident".

If a future user of this metric extends the EdgeType enum past 7, the formula breaks silently (entropy can exceed `log₂7`).

**Suggested change**: in `method.tex` immediately after Eq. 2, add a footnote: "We assume `H(p) ≤ log₂7` because EdgeType has cardinality 7 (Section~\ref{sec:typed-kg}). For extensibility, an implementation should use `log₂|EdgeType|` rather than `log₂7` as the denominator." This makes the assumption explicit and the implementation reusable.

**Severity**: P0 because the boundedness claim is a contribution-list item; an undeclared assumption underneath a contribution claim is fixable in one line and unfixable later.

## P1 — should fix

### P1-1. Bib entry imprecision: DistMult
**Where**: `paper/references.bib:127-133`
**Flag**: hygiene Citations

Entry has booktitle "ICLR 2015" but DOI points to arXiv:1412.6575 (preprint Dec 2014). NeurIPS-style canonical citation prefers the published venue. Either drop the arXiv DOI in favour of an ICLR 2015 URL or leave the arXiv DOI but flag that ICLR is primary. Minor — won't be flagged by reviewers in the way P0s will, but worth a 30-second fix.

### P1-2. Missing prior work — KG-BERT
**Where**: `paper/sections/related_work.tex:3-10`
**Flag**: hygiene Citations + substance Novelty

Yao et al. 2019 (arXiv:1909.03193) uses BERT for KG completion — directly comparable to DistMult's role as Harmony's generativity scorer. Should be cited as an LLM-based alternative to embedding methods in the KG-completion paragraph.

**Suggested addition**: one sentence in related_work.tex line 8: "LLM-based scorers such as KG-BERT \citep{yao2019kgbert} offer a contemporary alternative to embedding-based approaches; we use DistMult as Harmony's generativity component for its lower compute cost and well-understood ranking behaviour."

### P1-3. Missing prior work — CMA-ME
**Where**: `paper/sections/related_work.tex:22-26`
**Flag**: substance Novelty

Fontaine et al. 2020 (GECCO) shows that adaptive proposal mechanisms substantially outperform vanilla MAP-Elites. The paper's "we adopt MAP-Elites" framing in line 24 is exactly the position CMA-ME would attack.

**Suggested addition**: one sentence after line 26: "Adaptive QD methods such as CMA-ME \citep{fontaine2020cmame} improve sample efficiency over vanilla MAP-Elites; we adopt the simpler grid for interpretability of the (simplicity, gain) descriptor and leave adaptive variants for future work."

### P1-4. Missing prior work — AlphaProof
**Where**: `paper/sections/related_work.tex:12-20`
**Flag**: substance Novelty + hygiene Citations

Recent (2024-2025) flagship LLM-for-discovery work with formal verification. Relevant as a contrast — Harmony uses heuristic scoring, AlphaProof uses formal proof checking.

**Suggested addition**: one sentence in the automated-discovery paragraph contrasting Harmony's heuristic Harmony-score against AlphaProof's formal verification.

### P1-5. Triangle agreement (Eq. 3) ambiguous
**Where**: `paper/sections/method.tex:59` ("r_ac ∈ {r_ab, r_bc}")
**Flag**: hygiene Logic+Math

The notation is ambiguous: does `r_ac` need to *equal* one of the path types, or *be consistent* with them? Add a one-sentence clarification or a worked example.

### P1-6. DistMult hyperparameter scatter
**Where**: `paper/sections/method.tex:80` (cites d=50, K=10), `paper/sections/appendix.tex:300-307` (margin=1.0, lr=0.01, neg_samples=5, mask_ratio=0.20)
**Flag**: hygiene Logic+Math

Reproducibility requires reading the appendix table. Either consolidate into method.tex or add a forward reference: "(full hyperparameters in Appendix~\ref{app:hyperparams})".

### P1-7. Custom KGE baseline implementations not validated
**Where**: `paper/sections/experiments.tex` (TransE, RotatE, ComplEx implemented in-house)
**Flag**: hygiene Empirical

Custom implementations introduce risk of subtle algorithmic drift from published numbers. Either (a) cite a sanity-check against PyKEEN/published numbers in the appendix, or (b) acknowledge in limitations: "We implemented all four KGE baselines from scratch for self-containment; minor algorithmic differences from canonical implementations (PyKEEN) cannot be excluded".

### P1-8. "No-QD" ablation confounds two mechanisms
**Where**: `paper/sections/appendix.tex:475-478` (No-QD = greedy single-bin archive)
**Flag**: hygiene Empirical

Removing both the grid AND the diversity-based archive bookkeeping in the same condition means the paper cannot cleanly attribute "MAP-Elites is dominant" to either the grid or the archive mechanism. Either run a third ablation (greedy *with* grid) or qualify the claim: "we attribute structural gain to the archive-plus-grid combination; isolating their individual contributions is left for future work".

### P1-9. Stagnation recovery not isolated in any ablation
**Where**: `paper/sections/method.tex:118-174` (architecture), `paper/sections/appendix.tex` (no ablation row for stagnation off/on)
**Flag**: hygiene Logic+Math

The paper introduces a stagnation-triggered constrained-prompting mechanism but never ablates it. If a reviewer asks "does stagnation recovery actually help?", the paper has no answer. Either add a one-paragraph appendix note acknowledging this gap or run the ablation.

### P1-10. Backtesting "zero exact match" interpretation under-defended
**Where**: `paper/sections/appendix.tex:587-606`, `paper/sections/results.tex` (citation of zero-match result)
**Flag**: hygiene Logic+Math

Zero exact matches against held-out edges is framed positively ("proposals are theory-level, not memorised"). A skeptical reviewer can flip this: "your proposals are arbitrary and have no overlap with held-out structure". Add one sentence acknowledging both interpretations and pointing to the soft-match table as the discriminator.

### P1-11. Compute resources phrasing ambiguous
**Where**: `paper/sections/appendix.tex:329-339`
**Flag**: hygiene Compliance

"~10 min/domain, <2 CPU-hours total" is ambiguous: per-seed or aggregate? With 10 seeds × 5 domains × 20 generations, the math doesn't obviously close. Clarify: is the 10-min figure the total for all 10 seeds of one domain, or per-seed?

### P1-NEW1. Metric construct validity (4-component linear composite) *(Codex meta-review addition)*
**Where**: `paper/sections/method.tex:35-51` (Eq. 1 + 2), `paper/sections/appendix.tex` (calibration weight grid)
**Flag**: Codex meta-review

The Harmony score is a linear combination of four sub-scores (Compress, Cohere, Symm, Gener) with default weights α=β=γ=δ=0.25. The review never asked the foundational construct-validity question: are these four components theoretically commensurable under linear addition? Why not a multiplicative or geometric-mean composition? How were the equal-weight defaults selected, and is the calibration-weight grid (`appendix.tex:42-46`, α∈{0.3,0.5,0.7}, β∈{0.1,0.3}, γ=δ=0.25) the result of a sensitivity analysis or post-hoc adjustment to make the gate pass?

A reviewer asking "why this composite?" deserves a one-paragraph answer.

**Suggested change**: in `method.tex` after the weight statement (line 38), add: "We use a linear composition of the four components for interpretability — the contribution of each axis is identifiable in any given Harmony score. Multiplicative compositions (geometric means) penalise low values disproportionately and would conflate sub-axis weakness with overall metric failure. The equal-weight default reflects the absence of theoretical priors over the four desiderata; the calibration grid in Appendix~\ref{app:calibration-gate} explores this neighbourhood and confirms the gate passes across all six configurations." This converts an implicit design choice into a stated rationale.

**Severity**: P1 because the metric-design choice is not a P0 contribution-killer, but a strong reviewer will ask and a one-line answer pre-empts the question.

### P1-NEW2. Baseline budget fairness *(Codex meta-review addition)*
**Where**: `paper/sections/experiments.tex` (baseline setup), `paper/sections/appendix.tex:300-307` (hyperparameter table)
**Flag**: Codex meta-review

The 4 KGE baselines (TransE, DistMult, RotatE, ComplEx), the random-proposer ablation, and the Full Harmony pipeline are compared on Hits@10. But the paper does not document whether they receive comparable compute budgets, candidate counts, or tuning effort. If Harmony's pipeline runs 20 generations × 8 proposals/gen = 160 proposal-scoring rounds per seed, but the KGE baselines train once with 100 epochs and call it done, the comparison's "fairness" is implicit and can be attacked.

This is partially adjacent to the existing P1-7 (custom KGE implementation reproducibility), but Codex flagged it as a *separately-attackable axis*: even with cleanly implemented baselines, a budget mismatch undermines the comparison.

**Suggested change**: add one paragraph to `appendix.tex` near the hyperparameter table: "**Compute-budget parity.** The KGE baselines (TransE, DistMult, RotatE, ComplEx) train for 100 epochs on the full edge set, which on the largest domain (Wikidata Materials, 814 edges) consumes ≈ X seconds of CPU time. The Harmony pipeline runs Y generations × Z proposals/generation × W seconds/proposal-scoring ≈ V seconds. The Harmony pipeline thus consumes approximately R× the compute budget of the KGE baselines. A reviewer assessing the link-prediction comparison should treat this as a budget-asymmetric comparison favouring Harmony; the paper's directional-but-non-significant results should be read against this caveat." Fill in actual numbers from instrumentation.

**Severity**: P1 because budget unfairness is a standard reviewer attack but the paper's hedged framing already softens its impact.

### P1-12. NeurIPS Checklist Item 2 (Limitations) drift
**Where**: `paper/sections/checklist.tex:23`
**Flag**: hygiene Compliance

Checklist Justification cites "coarse vocabulary, metric-domain mismatch, evaluation circularity, lack of significance". The body discussion (after applying P0-5 and P0-6) will additionally cover LLM-judge circularity and KG-size limits. Update checklist to match.

## P2 — optional polish

### P2-1. Value function (Eq. 6) unbounded below
**Where**: `paper/sections/method.tex:89-91`

`V(Δ) = H(after) − H(before) − λ·Cost(Δ)` with λ=0.1 has no lower bound (cost can dominate). The paper does not run a sensitivity analysis on λ. Worth a one-line acknowledgment but not blocking.

### P2-2. Symmetry edge cases relegated to appendix
**Where**: `paper/sections/method.tex` (Eq. 4), `paper/sections/appendix.tex:695`

Zero-outdegree entities and single-entity types are handled in appendix only. A one-line forward reference in method.tex would prevent reproducibility confusion.

### P2-3. KG within-regime scaling absent
**Where**: experimental design overall

No 50→100→200→500-entity scaling curve within a single domain. P2 because the paper acknowledges its toy-KG scope; P0-6 already adds the explicit limitation.

## What the review did NOT find (positive surprises)

- **Internal consistency is clean**: zero numerical drift across abstract / intro / results / conclusion / appendix. Numbers like Hits@10 0.34/0.32, MRR 0.12/0.11, |δ_C| ≤ 0.12, "three of five domains" all agree across every site they appear (consistency reviewer report, Tables 1-3).
- **All cross-references resolve**: 12 critical `\ref{}` labels checked, no broken `??` in PDF.
- **Bib entries are factually correct**: TransE, RotatE, ComplEx, FunSearch, MAP-Elites all cite the right paper at the right venue/year. Only DistMult has the imprecise venue/DOI mix (P1-1).
- **Anonymisation is clean**: no self-citation, dataset entry properly anonymised, no author-name leak across `paper/`.
- **NeurIPS checklist 16/16 answered**: Justifications mostly track body claims; only Item 2 needs the post-fix update (P1-12).
- **LLM declaration in appendix is complete**: 3 LLMs, 10 seeds listed verbatim, sampling parameters per island stated.
- **Broader impacts paragraph**: positive + negative + mitigation all present.

## Recommended fix sequence (if user opts to apply) — **REVISED post Codex meta-review**

Codex's meta-review correctly flagged that the original "framing first" sequencing is operationally risky for a 4-day deadline: disclosure paragraphs add text and affect page density, so doing them after framing edits forces re-trimming. The corrected sequence is:

**Phase 1 — Disclosures and limitations** (recompile + page-count check after this phase). Land all the "add a paragraph" fixes first so the remaining page budget is known when framing edits are written:
- P0-2 calibration-gate transfer paragraph (results + abstract sentence)
- P0-5 LLM-judge circularity caveat (discussion)
- P1-13 (was P0-3) multiple-testing correction paragraph (appendix statistical methods)
- P1-14 (was P0-6) KG-size limitation (discussion limitations)
- P1-15 (was P0-7) per-domain failure-analysis paragraphs (discussion, 2-4 sentences per under-analysed domain)
- P1-16 (was P0-8) Eq. 2 boundedness footnote (method.tex)
- P1-NEW2 baseline budget-parity paragraph (appendix)

**Phase 2 — Framing rewrites** sized to the actual remaining page budget:
- P0-1 contribution attribution rewrite (intro contributions list, conclusion claim list, abstract opening)
- P0-NEW LLM-guidance non-identifiability sub-paragraph (discussion)
- P0-NEW2 (was P1-8) "MAP-Elites is dominant" qualification everywhere it appears
- P0-4 effect-size grounding ("complementary value" → "directional in negligible range") in abstract + discussion
- P1-NEW1 metric construct validity rationale (method.tex after weight statement)

**Phase 3 — Hygiene edits** (no page-density impact):
- P1-1 DistMult bib venue cleanup
- P1-2 KG-BERT citation
- P1-3 CMA-ME citation
- P1-4 AlphaProof citation
- P1-5 triangle-agreement clarification
- P1-6 hyperparameter consolidation forward-reference
- P1-7 custom KGE baseline implementation note
- P1-9 stagnation-recovery acknowledgment
- P1-10 backtesting interpretation footnote
- P1-11 compute-resources phrasing fix
- P1-12 checklist Item 2 sync (after Phase 1 disclosures land)

**Estimated effort**: 4-6 hours for the full P0+P1 batch; recompile and reverify page count between phases.

## Out of scope for this review

- **No experiments were re-run.** All findings are on the manuscript-as-is.
- **PDF was not recompiled**; suggested edits will require recompilation and a page-count re-verification.
- **No fixes were applied.** This document is review-only. The user will decide whether to apply fixes in a follow-up PR.

## Process notes

Eight subagents ran in two parallel waves (3 substance + 5 hygiene). Each was prompted with explicit "stress-test, do not be charitable" framing. Cross-validation by convergent independent finding: P0-1 flagged by 3 agents; P0-2 by 3 agents; P0-4, P0-5, P0-6 by 2 agents each. Single-agent P0 findings (P0-3, P0-7, P0-8) were elevated because they identify reviewer-attack vectors with high impact and low fix cost.

Codex was consulted before plan-mode exit to validate the perspective structure; it identified the missing centerpiece question ("would this still matter if directional results stayed weak?") which became the organising principle of the review.

### Codex meta-review of this review document

After the synthesis was complete, Codex was given the entire review document and asked to act as a senior NeurIPS area chair meta-reviewing the review itself. Three substantive corrections were applied:

1. **Convergent-finding bias acknowledgement**: all 8 subagents got the same adversarial "centerpiece question" framing. Their agreement is **shared-prompt convergence**, not independent reviewer convergence. The original review treated convergence as confidence, which inflated 4 of the 8 P0s. The convergence counts in this document should be read as "multiple angles surfaced this", not "three independent reviewers independently found this".

2. **Severity recalibration**: P0-3 (multiple-testing), P0-6 (KG-size), P0-7 (per-domain failure analysis depth), and P0-8 (Eq. 2 boundedness) were demoted to P1 because they conflate "cheap to fix and reviewer might object" with "submission blocker". P1-8 (No-QD ablation confound) was elevated to P0 because it is the empirical foundation of P0-1's reframed claim "MAP-Elites is dominant". The original "convergent-finding bias" was a major contributor to the over-grading.

3. **Missing findings added**: Codex flagged three substantive issues the 8 subagents did not surface, all attributable to shared-prompt blind spots: (i) LLM-guidance *non-identifiability* — a deeper version of P0-1 that asks whether the LLM contribution is identifiable at all, not just attributed correctly; (ii) metric *construct validity* — why a linear composite of 4 components, why these weights; (iii) baseline *budget fairness* — comparable compute / candidate counts across Harmony vs KGE baselines.

4. **Fix sequence inverted**: original "framing first" was operationally risky; corrected sequence puts disclosures first so framing edits can be sized to actual remaining page budget.

The bottom line from Codex: "apply with these specific changes" — directionally sound, but recalibrate severity and add the three missing findings. All four corrections applied above.
