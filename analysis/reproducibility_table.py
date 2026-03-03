"""Unified reproducibility table for all KG domains.

Generates a summary table with: entity count, edge count, entity types,
edge types, source (hand-built vs Wikidata), and construction method.

Addresses reviewer B-W1 (dataset definition inconsistency) and reviewer C
(Table 4 missing Wikidata stats).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harmony.types import KnowledgeGraph

_DOMAIN_BUILDERS: dict[str, dict[str, str]] = {
    "linear_algebra": {
        "builder": "harmony.datasets.linear_algebra.build_linear_algebra_kg",
        "source": "hand-built",
    },
    "periodic_table": {
        "builder": "harmony.datasets.periodic_table.build_periodic_table_kg",
        "source": "hand-built",
    },
    "astronomy": {
        "builder": "harmony.datasets.astronomy.build_astronomy_kg",
        "source": "hand-built",
    },
    "physics": {
        "builder": "harmony.datasets.physics.build_physics_kg",
        "source": "hand-built",
    },
    "materials": {
        "builder": "harmony.datasets.materials.build_materials_kg",
        "source": "hand-built",
    },
    "wikidata_physics": {
        "builder": "harmony.datasets.wikidata_physics.build_wikidata_physics_kg",
        "source": "Wikidata",
    },
    "wikidata_materials": {
        "builder": "harmony.datasets.wikidata_materials.build_wikidata_materials_kg",
        "source": "Wikidata",
    },
}


def _load_kg(builder_path: str) -> KnowledgeGraph:
    module_path, func_name = builder_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)()


def generate_reproducibility_table() -> pd.DataFrame:
    """Generate a unified reproducibility table for all 7 KG domains.

    Returns
    -------
    DataFrame with columns: domain, entities, edges, entity_types,
    edge_types, source.
    """
    rows: list[dict[str, object]] = []

    for domain, info in _DOMAIN_BUILDERS.items():
        kg = _load_kg(info["builder"])

        entity_types = {e.entity_type for e in kg.entities.values()}
        edge_types = {e.edge_type.name for e in kg.edges}

        rows.append(
            {
                "domain": domain,
                "entities": kg.num_entities,
                "edges": kg.num_edges,
                "entity_types": len(entity_types),
                "edge_types": len(edge_types),
                "source": info["source"],
            }
        )

    return pd.DataFrame(rows)
