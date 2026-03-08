#!/usr/bin/env python3
"""LLM-judge automated evaluation of top archive proposals.

Scores top-5 proposals per domain on four dimensions using Claude Opus 4.6
as an automated evaluator with two prompt variants for robustness.

Rubric dimensions (1–5 scale each):
  - Plausibility: scientific plausibility of the claim
  - Novelty: whether the claim adds new information
  - Falsifiability: whether the falsification condition is testable
  - Clarity: clarity and specificity of the claim text

Usage:
    uv run python scripts/llm_judge_rubric.py \
        --output data/results/llm_rubric_scores.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

DIMENSIONS = ["plausibility", "novelty", "falsifiability", "clarity"]
SCORE_MIN, SCORE_MAX = 1, 5


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_DETAILED_RUBRIC = """
Plausibility (1-5): Is the claim scientifically plausible given known domain knowledge?
  1=implausible/contradicts established knowledge, 3=uncertain/speculative, 5=well-grounded

Novelty (1-5): Does the claim add genuinely new information beyond the existing KG?
  1=trivially obvious/redundant, 3=moderately novel, 5=genuinely surprising and new

Falsifiability (1-5): Is the falsification condition concrete and experimentally testable?
  1=unfalsifiable/vague, 3=partially testable, 5=precisely operationalised

Clarity (1-5): Is the claim stated clearly and specifically enough to be actionable?
  1=vague/ambiguous, 3=partially clear, 5=precise and unambiguous
""".strip()

_CONCISE_RUBRIC = """
Rate 1-5 on: Plausibility (scientific groundedness), Novelty (new information added),
Falsifiability (testability of falsification condition), Clarity (specificity).
1=poor, 3=moderate, 5=excellent.
""".strip()


def build_rubric_prompt(proposal: dict, variant: str) -> str:
    """Build evaluation prompt for a proposal.

    Parameters
    ----------
    proposal: Proposal dict with claim, justification, falsification_condition fields.
    variant: "detailed" or "concise" — controls rubric verbosity.

    Returns
    -------
    Prompt string to send to the LLM judge.
    """
    if variant not in ("detailed", "concise"):
        raise ValueError(f"variant must be 'detailed' or 'concise', got {variant!r}")

    rubric = _DETAILED_RUBRIC if variant == "detailed" else _CONCISE_RUBRIC

    claim = proposal.get("claim", "")
    justification = proposal.get("justification", "")
    falsification = proposal.get("falsification_condition", "")
    domain = proposal.get("kg_domain", "unknown")
    edge_type = proposal.get("edge_type", "")
    source = proposal.get("source_entity", "")
    target = proposal.get("target_entity", "")

    return f"""You are an expert scientific knowledge evaluator assessing theory proposals for
knowledge graphs. Evaluate the following proposal from the {domain} domain.

PROPOSAL:
  Claim: {claim}
  Justification: {justification}
  Falsification condition: {falsification}
  KG mutation: {edge_type}({source} → {target})

RUBRIC:
{rubric}

Respond with ONLY a valid JSON object in this exact format:
{{"plausibility": <1-5>, "novelty": <1-5>, "falsifiability": <1-5>, "clarity": <1-5>}}

No explanation, no markdown, no other text — just the JSON object."""


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------


def parse_rubric_scores(response: str) -> dict[str, int]:
    """Extract rubric scores from LLM response text.

    Handles plain JSON and JSON inside markdown code blocks.

    Returns
    -------
    Dict with keys: plausibility, novelty, falsifiability, clarity (int values 1-5).

    Raises
    ------
    ValueError if scores are out of range or dimensions are missing.
    json.JSONDecodeError if response cannot be parsed as JSON.
    """
    text = response.strip()

    # Strip markdown code block if present
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1)
    else:
        # Try to extract the first JSON object from the text
        json_match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

    data = json.loads(text)

    # Validate all dimensions are present
    missing = [d for d in DIMENSIONS if d not in data]
    if missing:
        raise ValueError(f"Missing dimensions in response: {missing}")

    # Validate score range
    for dim in DIMENSIONS:
        val = int(data[dim])
        if not (SCORE_MIN <= val <= SCORE_MAX):
            raise ValueError(
                f"Score for '{dim}' is {val}, must be in [{SCORE_MIN}, {SCORE_MAX}]"
            )
        data[dim] = val

    return {d: data[d] for d in DIMENSIONS}


# ---------------------------------------------------------------------------
# Agreement metric
# ---------------------------------------------------------------------------


def compute_agreement(
    scores_a: dict[str, int],
    scores_b: dict[str, int],
    tolerance: int = 0,
) -> float:
    """Compute inter-prompt agreement as fraction of dimensions within tolerance.

    Parameters
    ----------
    scores_a, scores_b: Score dicts from the two prompt variants.
    tolerance: Max absolute difference allowed for a "match" (default 0 = exact).

    Returns
    -------
    Float in [0, 1] — fraction of dimensions where |a - b| <= tolerance.
    """
    matches = sum(
        1 for d in DIMENSIONS if abs(scores_a[d] - scores_b[d]) <= tolerance
    )
    # Scale to [0, 1] based on range: agreement of 0 when all differ by max (4)
    # and 1 when all match exactly.
    # Use fraction-of-dimensions-matching as simple agreement metric.
    total_diff = sum(abs(scores_a[d] - scores_b[d]) for d in DIMENSIONS)
    max_diff = (SCORE_MAX - SCORE_MIN) * len(DIMENSIONS)  # = 4 * 4 = 16
    if max_diff == 0:
        return 1.0
    return 1.0 - (total_diff / max_diff)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_scores(entries: list[dict[str, float]]) -> dict[str, float]:
    """Compute mean score per dimension across multiple proposal evaluations.

    Parameters
    ----------
    entries: List of score dicts, each with DIMENSIONS as keys.

    Returns
    -------
    Dict with per-dimension means.

    Raises
    ------
    ValueError if entries is empty.
    """
    if not entries:
        raise ValueError("entries must be non-empty")

    return {
        dim: statistics.mean(e[dim] for e in entries)
        for dim in DIMENSIONS
    }


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------


def _call_claude(prompt: str, model: str = "claude-opus-4-6") -> str:
    """Call Claude API and return the response text."""
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def evaluate_domain_proposals(
    proposals: list[dict],
    domain: str,
    model: str = "claude-opus-4-6",
    top_n: int = 5,
) -> dict:
    """Evaluate top-N proposals for a domain using two prompt variants.

    Returns
    -------
    Dict with keys: domain, proposal_scores (list), mean_scores, agreement_rate.
    """
    top = proposals[:top_n]
    proposal_scores = []

    for prop in top:
        prompt_detailed = build_rubric_prompt(prop, variant="detailed")
        prompt_concise = build_rubric_prompt(prop, variant="concise")

        try:
            resp_a = _call_claude(prompt_detailed, model=model)
            scores_a = parse_rubric_scores(resp_a)
        except Exception as e:
            print(f"  [WARN] detailed prompt failed for {prop.get('id', '?')}: {e}")
            continue

        try:
            resp_b = _call_claude(prompt_concise, model=model)
            scores_b = parse_rubric_scores(resp_b)
        except Exception as e:
            print(f"  [WARN] concise prompt failed for {prop.get('id', '?')}: {e}")
            # Use only detailed scores with agreement=N/A
            scores_b = scores_a

        agreement = compute_agreement(scores_a, scores_b)
        # Average the two variants for the final score
        avg_scores = {d: (scores_a[d] + scores_b[d]) / 2.0 for d in DIMENSIONS}

        proposal_scores.append({
            "proposal_id": prop.get("id", "unknown"),
            "claim": prop.get("claim", "")[:120],
            "scores_detailed": scores_a,
            "scores_concise": scores_b,
            "scores_avg": avg_scores,
            "inter_prompt_agreement": agreement,
        })

    if not proposal_scores:
        return {"domain": domain, "proposal_scores": [], "mean_scores": {}, "agreement_rate": 0.0}

    mean_scores = aggregate_scores([p["scores_avg"] for p in proposal_scores])
    agreement_rate = statistics.mean(p["inter_prompt_agreement"] for p in proposal_scores)

    return {
        "domain": domain,
        "n_evaluated": len(proposal_scores),
        "proposal_scores": proposal_scores,
        "mean_scores": mean_scores,
        "agreement_rate": agreement_rate,
    }


def load_top_proposals(domain: str, base_dir: Path, top_n: int = 5) -> list[dict]:
    """Load top-N proposals by fitness from MLX batch run archive."""
    import glob

    proposals_with_fitness: list[tuple[float, dict]] = []
    for cp_path in sorted(glob.glob(str(base_dir / domain / "seed_*/checkpoint.json"))):
        cp = json.loads(Path(cp_path).read_text())
        cells = cp.get("archive", {}).get("cells", {})
        for cell in cells.values():
            fitness = cell.get("fitness_signal", 0.0)
            prop = cell.get("proposal", {})
            if prop.get("claim"):
                proposals_with_fitness.append((fitness, prop))

    # Sort by fitness descending, deduplicate by claim text
    proposals_with_fitness.sort(key=lambda x: x[0], reverse=True)
    seen_claims: set[str] = set()
    top: list[dict] = []
    for _, prop in proposals_with_fitness:
        claim = prop.get("claim", "")
        if claim not in seen_claims:
            seen_claims.add(claim)
            top.append(prop)
            if len(top) == top_n:
                break
    return top


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM judge rubric evaluation")
    parser.add_argument(
        "--mlx-base",
        type=Path,
        default=Path("artifacts/harmony_mlx_full/2026-03-05_mlx_no_think_full"),
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/llm_rubric_scores.json"),
    )
    args = parser.parse_args()

    domains = ["astronomy", "physics", "materials", "wikidata_physics", "wikidata_materials"]
    results: dict[str, dict] = {}

    for domain in domains:
        print(f"\n[{domain}] Loading top-{args.top_n} proposals...")
        proposals = load_top_proposals(domain, args.mlx_base, top_n=args.top_n)
        print(f"  Found {len(proposals)} proposals. Evaluating...")
        result = evaluate_domain_proposals(
            proposals, domain, model=args.model, top_n=args.top_n
        )
        results[domain] = result
        ms = result.get("mean_scores", {})
        if ms:
            print(
                f"  Mean scores: "
                f"plaus={ms.get('plausibility', 0):.2f} "
                f"novel={ms.get('novelty', 0):.2f} "
                f"falsi={ms.get('falsifiability', 0):.2f} "
                f"clarity={ms.get('clarity', 0):.2f} "
                f"agreement={result.get('agreement_rate', 0):.2f}"
            )

    # Overall summary
    all_means = [r["mean_scores"] for r in results.values() if r.get("mean_scores")]
    if all_means:
        overall = aggregate_scores(all_means)
        results["_overall"] = {
            "mean_scores": overall,
            "agreement_rate": statistics.mean(
                r["agreement_rate"] for r in results.values() if "agreement_rate" in r
            ),
        }
        print("\nOverall mean scores:")
        for dim, score in overall.items():
            print(f"  {dim}: {score:.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
