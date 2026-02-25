"""Discovery dataset: effective theories and scaling laws knowledge graph.

Encodes physical regimes, theories, observables, scaling laws, and
approximations. Missing scaling-relation "explains" edges between
theories/regimes and observables are the discovery targets.
"""
from __future__ import annotations

from ..types import EdgeType, Entity, KnowledgeGraph, TypedEdge

_ENTITIES: list[tuple[str, str, dict]] = [
    # Core theories
    ("newtonian_mechanics",     "theory", {"domain": "classical", "year": 1687}),
    ("quantum_mechanics",       "theory", {"domain": "quantum", "year": 1925}),
    ("special_relativity",      "theory", {"domain": "relativistic", "year": 1905}),
    ("general_relativity",      "theory", {"domain": "gravitational", "year": 1915}),
    ("thermodynamics",          "theory", {"domain": "thermal", "year": 1850}),
    ("statistical_mechanics",   "theory", {"domain": "statistical", "year": 1870}),
    ("quantum_field_theory",    "theory", {"domain": "quantum_relativistic", "year": 1948}),
    ("condensed_matter_theory", "theory", {"domain": "many_body", "year": 1950}),
    # Physical regimes
    ("classical_regime",        "regime", {"condition": "ℏ→0 or large quantum numbers"}),
    ("quantum_regime",          "regime", {"condition": "ℏ effects dominate"}),
    ("relativistic_regime",     "regime", {"condition": "v ~ c"}),
    ("non_relativistic_regime", "regime", {"condition": "v << c"}),
    ("high_temperature_regime", "regime", {"condition": "kT >> energy gaps"}),
    ("low_temperature_regime",  "regime", {"condition": "kT << energy gaps"}),
    ("high_energy_regime",      "regime", {"condition": "E >> rest mass energy"}),
    ("gravitational_regime",    "regime", {"condition": "GM/rc² ~ 1"}),
    # Observables
    ("energy",                  "observable", {"dimension": "ML²T⁻²"}),
    ("momentum",                "observable", {"dimension": "MLT⁻¹"}),
    ("force",                   "observable", {"dimension": "MLT⁻²"}),
    ("temperature",             "observable", {"dimension": "Θ"}),
    ("pressure",                "observable", {"dimension": "ML⁻¹T⁻²"}),
    ("entropy",                 "observable", {"dimension": "ML²T⁻²Θ⁻¹"}),
    ("heat_capacity",           "observable", {"dimension": "ML²T⁻²Θ⁻¹"}),
    ("electrical_conductivity", "observable", {"dimension": "M⁻¹L⁻³T³A²"}),
    ("magnetic_moment",         "observable", {"dimension": "L²A"}),
    # Scaling laws / equations
    ("coulombs_law",            "scaling_law", {"formula": "F = kq₁q₂/r²"}),
    ("ohms_law",                "scaling_law", {"formula": "V = IR"}),
    ("planck_radiation_law",    "scaling_law", {"formula": "B(ν,T) = 2hν³/c² × 1/(e^(hν/kT)-1)"}),
    ("boltzmann_distribution",  "scaling_law", {"formula": "P(E) ∝ e^(-E/kT)"}),
    ("fourier_law",             "scaling_law", {"formula": "q = -k∇T"}),
    ("navier_stokes",           "scaling_law", {"formula": "ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u"}),
    ("schrodinger_equation",    "scaling_law", {"formula": "iℏ∂ψ/∂t = Ĥψ"}),
    ("einstein_field_equations","scaling_law", {"formula": "Gμν = 8πGTμν/c⁴"}),
    # Approximations
    ("harmonic_approximation", "approximation", {"description": "Expand potential to 2nd order"}),
    ("born_oppenheimer_approximation", "approximation", {"description": "Separate nuclei from e-"}),
    ("mean_field_approximation", "approximation", {"description": "Single-particle mean field"}),
    ("perturbation_theory", "approximation", {"description": "Expand in small parameter"}),
    ("saddle_point_approximation", "approximation", {"description": "Stationary phase integral"}),
    # Unifying concepts
    ("symmetry", "concept", {"description": "Invariance under transformation"}),
    ("conservation_law", "concept", {"description": "Noether: symmetry → conservation"}),
    ("phase_transition", "concept", {"description": "Qualitative change in collective state"}),
    ("spontaneous_symmetry_breaking", "concept", {"description": "Ground state breaks symmetry"}),
    ("universality_class", "concept", {"description": "Systems sharing critical exponents"}),
]

_EDGES: list[tuple[str, str, EdgeType]] = [
    # Theory hierarchy (classical limits / generalisations)
    ("quantum_mechanics",     "newtonian_mechanics",   EdgeType.GENERALIZES),
    ("special_relativity",    "newtonian_mechanics",   EdgeType.GENERALIZES),
    ("general_relativity",    "special_relativity",    EdgeType.GENERALIZES),
    ("quantum_field_theory",  "quantum_mechanics",     EdgeType.GENERALIZES),
    ("quantum_field_theory",  "special_relativity",    EdgeType.GENERALIZES),
    ("statistical_mechanics", "thermodynamics",        EdgeType.DERIVES),
    ("condensed_matter_theory","statistical_mechanics", EdgeType.DEPENDS_ON),
    # Theory → regime correspondences
    ("classical_regime",      "newtonian_mechanics",   EdgeType.DEPENDS_ON),
    ("quantum_regime",        "quantum_mechanics",     EdgeType.DEPENDS_ON),
    ("relativistic_regime",   "special_relativity",    EdgeType.DEPENDS_ON),
    ("gravitational_regime",  "general_relativity",    EdgeType.DEPENDS_ON),
    ("high_energy_regime",    "quantum_field_theory",  EdgeType.DEPENDS_ON),
    ("low_temperature_regime","quantum_mechanics",     EdgeType.DEPENDS_ON),
    ("high_temperature_regime","classical_regime",     EdgeType.EQUIVALENT_TO),
    ("non_relativistic_regime","classical_regime",     EdgeType.GENERALIZES),
    # Scaling laws derive from theories
    ("schrodinger_equation",     "quantum_mechanics",    EdgeType.DERIVES),
    ("einstein_field_equations", "general_relativity",   EdgeType.DERIVES),
    ("boltzmann_distribution",   "statistical_mechanics",EdgeType.DERIVES),
    ("planck_radiation_law",     "boltzmann_distribution",EdgeType.DERIVES),
    ("coulombs_law",             "force",               EdgeType.EXPLAINS),
    ("ohms_law",                 "electrical_conductivity", EdgeType.EXPLAINS),
    ("fourier_law",              "temperature",         EdgeType.DEPENDS_ON),
    ("fourier_law",              "heat_capacity",       EdgeType.EXPLAINS),
    ("navier_stokes",            "pressure",            EdgeType.DEPENDS_ON),
    # Approximations depend on parent theories
    ("harmonic_approximation",         "quantum_mechanics",  EdgeType.DEPENDS_ON),
    ("born_oppenheimer_approximation", "quantum_mechanics",  EdgeType.DEPENDS_ON),
    ("born_oppenheimer_approximation", "condensed_matter_theory", EdgeType.EXPLAINS),
    ("mean_field_approximation",       "statistical_mechanics", EdgeType.DEPENDS_ON),
    ("mean_field_approximation",       "condensed_matter_theory",EdgeType.DERIVES),
    ("perturbation_theory",            "quantum_mechanics",  EdgeType.DEPENDS_ON),
    ("saddle_point_approximation",     "statistical_mechanics",EdgeType.DEPENDS_ON),
    # Unifying concepts
    ("symmetry",                       "conservation_law",          EdgeType.EXPLAINS),
    ("spontaneous_symmetry_breaking",  "symmetry",                  EdgeType.DEPENDS_ON),
    ("spontaneous_symmetry_breaking",  "phase_transition",          EdgeType.EXPLAINS),
    ("universality_class",             "phase_transition",          EdgeType.DEPENDS_ON),
    ("universality_class",             "statistical_mechanics",     EdgeType.DERIVES),
    ("phase_transition",               "thermodynamics",            EdgeType.EXPLAINS),
    # Observable dependencies
    ("entropy",      "statistical_mechanics", EdgeType.DEPENDS_ON),
    ("entropy",      "thermodynamics",        EdgeType.DEPENDS_ON),
    ("heat_capacity","thermodynamics",        EdgeType.DEPENDS_ON),
    ("magnetic_moment","quantum_mechanics",   EdgeType.DEPENDS_ON),
    ("energy",       "newtonian_mechanics",   EdgeType.DEPENDS_ON),
    ("momentum",     "newtonian_mechanics",   EdgeType.DEPENDS_ON),
    ("force",        "newtonian_mechanics",   EdgeType.DEPENDS_ON),
    # Schrodinger → energy / momentum
    ("schrodinger_equation","energy",   EdgeType.DEPENDS_ON),
    ("schrodinger_equation","momentum", EdgeType.DEPENDS_ON),
]


def build_physics_kg() -> KnowledgeGraph:
    """Build the effective theories / scaling laws discovery knowledge graph."""
    kg = KnowledgeGraph(domain="physics")
    for eid, etype, props in _ENTITIES:
        kg.add_entity(Entity(id=eid, entity_type=etype, properties=props))
    for src, tgt, etype in _EDGES:
        kg.add_edge(TypedEdge(source=src, target=tgt, edge_type=etype))
    return kg
