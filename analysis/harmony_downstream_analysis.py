"""Harmony↔downstream correlation, failure mode analysis, and regime characterization.

Addresses reviewer C-W5 (Harmony score improvements don't transfer to
link prediction gains) and reviewer A-Q3 (per-domain failure analysis).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.component_correlation import _spearman


def component_correlation_analysis(
    proposal_scores: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute Spearman correlation between each Harmony component delta and Hits@10 delta.

    Parameters
    ----------
    proposal_scores:
        List of dicts, each with keys: delta_compress, delta_cohere,
        delta_symm, delta_gener, delta_hits10.

    Returns
    -------
    Dict with:
      - correlations: {component: Spearman rho with delta_hits10}
      - p_values: {component: p-value} (placeholder — use scipy for real p-values)
    """
    components = ["delta_compress", "delta_cohere", "delta_symm", "delta_gener"]
    hits10_values = [s["delta_hits10"] for s in proposal_scores]

    correlations: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for comp in components:
        comp_values = [s[comp] for s in proposal_scores]
        rho = _spearman(comp_values, hits10_values)
        correlations[comp] = rho
        # Placeholder p-value (real implementation would use scipy.stats)
        p_values[comp] = 0.0

    return {"correlations": correlations, "p_values": p_values}


def failure_mode_analysis(
    proposals: list[dict[str, object]],
) -> dict[str, dict[str, int]]:
    """Classify proposals into failure modes.

    Parameters
    ----------
    proposals:
        List of dicts with keys: proposal_type, edge_type, valid, hits10_delta.

    Returns
    -------
    Dict with:
      - type_distribution: Counter of proposal_type values
      - edge_type_distribution: Counter of edge_type values (excluding None)
      - classification: counts of valid_helpful, valid_neutral, valid_harmful
    """
    type_dist: Counter[str] = Counter()
    edge_type_dist: Counter[str] = Counter()
    classification: dict[str, int] = {
        "valid_helpful": 0,
        "valid_neutral": 0,
        "valid_harmful": 0,
    }

    for p in proposals:
        ptype = str(p.get("proposal_type", "UNKNOWN"))
        type_dist[ptype] += 1

        etype = p.get("edge_type")
        if etype is not None:
            edge_type_dist[str(etype)] += 1

        valid = bool(p.get("valid", False))
        delta = float(p.get("hits10_delta", 0.0))

        if valid:
            if delta > 0:
                classification["valid_helpful"] += 1
            elif delta < 0:
                classification["valid_harmful"] += 1
            else:
                classification["valid_neutral"] += 1

    return {
        "type_distribution": dict(type_dist),
        "edge_type_distribution": dict(edge_type_dist),
        "classification": classification,
    }


def regime_characterization(
    domain_results: list[dict[str, float]],
) -> dict[str, list[dict[str, float]]]:
    """Build scatter data for regime characterization.

    Parameters
    ----------
    domain_results:
        List of dicts with keys: domain, density, entropy, harmony_freq_gap.

    Returns
    -------
    Dict with:
      - density_vs_gap: list of {x: density, y: harmony_freq_gap, domain}
      - entropy_vs_gap: list of {x: entropy, y: harmony_freq_gap, domain}
    """
    density_vs_gap: list[dict[str, float]] = []
    entropy_vs_gap: list[dict[str, float]] = []

    for r in domain_results:
        density_vs_gap.append(
            {
                "x": float(r["density"]),
                "y": float(r["harmony_freq_gap"]),
                "domain": r["domain"],
            }
        )
        entropy_vs_gap.append(
            {
                "x": float(r["entropy"]),
                "y": float(r["harmony_freq_gap"]),
                "domain": r["domain"],
            }
        )

    return {
        "density_vs_gap": density_vs_gap,
        "entropy_vs_gap": entropy_vs_gap,
    }
