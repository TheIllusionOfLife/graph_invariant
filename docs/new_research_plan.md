# NeurIPS Research Plan: Harmony-Driven Theory Discovery

> **Working title**: *Harmony-Driven Theory Discovery: Calibrating “Perfect” Knowledge Systems in Mathematics and the Periodic Table to Predict Missing Theories in Physics/Astronomy*

This document is a **detailed, execution-ready research plan** for a NeurIPS paper. It follows the recommended strategy:

1. **Define metrics for a “perfect” knowledge system** (harmony).
2. **Calibrate the metrics** on *near-perfect* systems:
   - **Linear algebra (morphisms-first mathematics)**
   - **The periodic table (structured empirical regularities)**
3. **Apply the metrics** to *messier* systems (physics/astronomy) to **predict missing theories** as **node/edge additions** or **constraints**.

---

## 1. Research thesis

### 1.1 Hypothesis
Knowledge systems that humans regard as “beautiful” or “complete” share measurable structural properties:

- **High compressibility** (few primitives explain many facts)
- **High coherence** (few contradictions, commutative reasoning paths)
- **High symmetry / invariance** (structure preserved under transformations)
- **High generativity** (rules generate correct new consequences)

**Claim**: These properties can be formalized as a **Harmony Metric**. When applied to incomplete scientific domains, the metric induces **high-confidence proposals** for missing components (concepts, relations, or constraints) that reduce distortion.

### 1.2 NeurIPS framing
- **Problem**: Scientific discovery is hard because our knowledge graphs/ontologies are incomplete and inconsistent.
- **Approach**: Learn a *domain-agnostic* notion of “harmony” from near-perfect systems and use it as a discovery prior.
- **Outcome**: A system that proposes hypotheses with measurable “harmony gain” and concrete validation protocols.

---

## 2. Formalism: Knowledge system as typed structure

We represent a domain as a **typed relational structure**.

### 2.1 Objects
- **Entities**: concepts, objects, variables, phenomena, laws, operators.
- **Statements**: definitions, lemmas, theorems, empirical regularities.
- **Transforms**: mappings between entities (e.g., linear maps, changes of basis), generalizations, specializations.

### 2.2 Edges (typed)
We start with a small set of edge types that are widely applicable:

1. **depends_on** (A requires B)
2. **derives** (A ⇒ B)
3. **equivalent_to** (A ⇔ B)
4. **maps_to** (f: A → B)
5. **explains** (A explains phenomenon P)
6. **contradicts** (A conflicts with B)
7. **generalizes** (inverse direction encodes specialization)

> **Design principle**: Prefer *morphism-first* edges (maps_to, derives) over static “is-a” taxonomies.

---

## 3. Harmony Metric (core contribution)

We define **Distortion(D)** for a domain structure **D** and **Harmony(D) = −Distortion(D)**.

### 3.1 Distortion components

#### A) Compressibility (MDL-style)
Approximate description length:

- **L_rules**: encoding length of primitives and rules
- **L_data|rules**: encoding length of observed statements given rules

**Distortion_comp(D) = L_rules + L_data|rules**

Practical proxies:
- number of primitive concepts
- minimal generating set size (via subgraph selection)
- rule/template reuse rate

#### B) Coherence / Commutativity
We measure how often alternative reasoning paths agree.

- For many triplets (A, B, C), if A→B and B→C exists, compare with direct A→C.

**Distortion_coh(D) = 1 − commutative_path_agreement_rate**

Proxies:
- cycle inconsistency counts
- contradiction edge density
- proof/path divergence entropy

#### C) Symmetry / Invariance
We measure whether the structure is stable under natural transformations.

- node relabelings / synonym merges
- change-of-basis–like transforms
- automorphism group size proxies

**Distortion_sym(D) = penalty for broken invariance**

Proxies:
- embedding isometry under transformations
- invariants explaining clusters (e.g., periodicity in chemistry)

#### D) Generativity / Predictive gain
We measure whether the system can predict withheld facts.

**Distortion_gen(D) = 1 − predictive_score(D)**

Predictive tasks:
- missing edge prediction
- missing attribute prediction
- constraint satisfaction / bound satisfaction

### 3.2 Composite objective

**Distortion(D) = α·Distortion_comp + β·Distortion_coh + γ·Distortion_sym + δ·Distortion_gen**

We score a proposal Δ (adding nodes/edges/constraints) by:

**Value(Δ) = Distortion(D) − Distortion(D ⊕ Δ) − λ·Cost(Δ)**

Where Cost(Δ) penalizes overly complex additions.

---

## 4. Calibration domains

### 4.1 Linear algebra (math calibration)

**Why linear algebra**:
- morphism-first (linear maps)
- rich invariance (change of basis)
- abundant commutative diagrams (composition)
- strong compression via universal constructions (rank-nullity, diagonalization patterns)

**Data sources** (initial):
- textbook structure (chapters/definitions/theorems)
- Lean/mathlib subset (optional)
- curated micro-ontology built by us (MVP)

**Tasks**:
1. **Edge recovery**: mask “derives/depends_on” edges and recover.
2. **Commutativity auditing**: detect non-commuting reasoning paths.
3. **Latent node proposal**: propose missing intermediate lemmas/concepts to restore modularity.

### 4.2 Periodic table (empirical calibration)

**Why periodic table**:
- strong symmetry/periodicity (groups/periods)
- historical “missing element” discoveries
- structured exceptions (transition metals) test robustness

**Entities**:
- elements
- properties (atomic radius, electronegativity, oxidation states)
- relations (same group, periodic trend)

**Tasks**:
1. **Missing property prediction** (mask values).
2. **Missing node prediction** (simulate pre-discovery missing elements).
3. **Constraint discovery**: derive monotonic/periodic constraints that reduce distortion.

---

## 5. Discovery domains (target)

We need a domain where:
- the knowledge system is incomplete
- candidates can be proposed as structured additions
- evaluation is feasible without decades of experiments

### Candidate A) Astronomy: exoplanet demographics + formation theory
- Entities: planet properties, host star properties, formation mechanisms
- Missing edges: “explains” links between mechanisms and observed distributions
- Evaluation: predictive likelihood on held-out surveys

### Candidate B) Physics: effective theories / scaling laws in messy regimes
- Entities: regimes, observables, approximations
- Additions: missing scaling relations / constraints
- Evaluation: out-of-sample prediction + constraint satisfaction

### Candidate C) Materials: phase diagrams and symmetry groups
- Entities: materials, phases, symmetries, transitions
- Additions: missing phase boundaries or invariance constraints
- Evaluation: consistency with known phase data

---

## 6. Hypothesis generation engine

### 6.1 Representation
We support three proposal types:

1. **Edge addition**: connect existing nodes with typed relation.
2. **Node addition**: introduce a latent concept (intermediate lemma/mechanism).
3. **Constraint addition**: inequalities, monotonicities, conservation-like laws.

### 6.2 Proposal model (LLM as editor)
LLM proposes Δ based on:
- local substructure where distortion is high
- supporting snippets/definitions
- the harmony principles

**Output schema** (structured):
- proposal_type
- nodes/edges/constraints
- justification
- minimal test plan
- falsification condition

### 6.3 Search and selection
We combine:
- **diversity search** (MAP-Elites style) along axes like simplicity × novelty
- **self-correction** loop to fix invalid proposals
- **counterexample-guided refinement** for constraints

---

## 7. Evaluation plan

### 7.1 Calibration metrics
We must show the harmony metric correlates with “completeness” in calibration domains.

Experiments:
1. **Ablation**: each distortion component removed.
2. **Mask-and-recover**: edge/property/node masking.
3. **Generalization**: apply learned weights to another math subdomain.

### 7.2 Discovery validity metrics
We evaluate proposed hypotheses by:

- **Harmony gain**: Δ reduces distortion significantly.
- **Predictive improvement**: better held-out prediction.
- **Robustness**: remains good under perturbations / resampling.
- **Novelty**: not reducible to existing named relations (via retrieval + correlation checks).

### 7.3 Human-auditability
Provide:
- compact hypothesis descriptions
- minimal supporting evidence
- counterexamples if any

---

## 8. Minimal viable prototype (MVP)

### 8.1 MVP scope (2–3 weeks)
1. Build a **small linear algebra ontology** (100–300 nodes)
2. Build a **periodic table dataset** with properties
3. Implement 2–3 harmony components (comp, coh, gen)
4. Run mask-and-recover benchmarks
5. Produce a ranked list of proposed missing edges/nodes/constraints

### 8.2 Deliverables
- code + dataset
- evaluation notebooks
- qualitative examples of “good” hypotheses

---

## 9. NeurIPS paper outline

1. Introduction: harmony as a discovery prior
2. Related work: KG completion, theorem proving, symbolic regression, AI4Science
3. Method: harmony metric + proposal engine
4. Calibration: linear algebra + periodic table
5. Discovery domain: (choose one) + results
6. Ablations + robustness
7. Discussion: limits, bias, scientific validity

---

## 10. Questions for you (to finalize the plan)

Please answer these so I can update this document precisely:

1. **Target discovery domain**: Which should we commit to for NeurIPS?
   - A) Astronomy (exoplanets)
   - B) Physics (effective theories / scaling laws)
   - C) Materials (phase diagrams)
   - D) Other (your idea)

2. **Calibration math subdomain**: Confirm the starting point.
   - A) Linear algebra (recommended)
   - B) Group theory
   - C) Category theory basics

3. **Representation preference**:
   - Keep a typed knowledge graph throughout
   - Or shift early to a more algebraic/axiomatic representation (Lean-like)

4. **Compute constraints**: Are we targeting local models only, or can we use API LLMs for proposal generation?

5. **Desired output type** in the discovery domain:
   - edges (missing relations)
   - nodes (latent concepts)
   - constraints (inequalities/scaling laws)

6. **Success criterion**: What would you consider a “win” for the paper?
   - one high-confidence novel hypothesis with strong empirical support
   - many medium-confidence hypotheses ranked well
   - strong calibration + transfer, even if discovery hypotheses are tentative

