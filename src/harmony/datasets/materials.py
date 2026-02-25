"""Discovery dataset: phase diagrams and materials knowledge graph.

Encodes materials, phases, crystal structures, physical properties,
phase transitions, and applications. Missing phase-boundary "explains"
edges (which condition triggers which transition) are the discovery targets.
"""

from __future__ import annotations

from ..types import EdgeType, Entity, KnowledgeGraph, TypedEdge

_ENTITIES: list[tuple[str, str, dict]] = [
    # Materials
    ("iron", "material", {"symbol": "Fe", "category": "pure_metal"}),
    ("steel", "material", {"composition": "Fe-C alloy", "category": "alloy"}),
    ("aluminum", "material", {"symbol": "Al", "category": "pure_metal"}),
    ("copper", "material", {"symbol": "Cu", "category": "pure_metal"}),
    ("silicon", "material", {"symbol": "Si", "category": "semiconductor"}),
    ("germanium", "material", {"symbol": "Ge", "category": "semiconductor"}),
    ("gallium_arsenide", "material", {"formula": "GaAs", "category": "compound_sc"}),
    ("niobium_titanium", "material", {"formula": "NbTi", "category": "superconductor"}),
    ("barium_titanate", "material", {"formula": "BaTiO3", "category": "ferroelectric"}),
    ("yttrium_barium_copper_oxide", "material", {"formula": "YBa2Cu3O7", "category": "high_tc_sc"}),
    ("manganese_oxide", "material", {"formula": "MnO", "category": "antiferromagnet"}),
    # Phases
    ("ferromagnetic_phase", "phase", {"description": "Spontaneous magnetization below Tc"}),
    ("paramagnetic_phase", "phase", {"description": "Random spin orientation above Tc"}),
    ("antiferromagnetic_phase", "phase", {"description": "Alternating spin alignment below TN"}),
    ("ferroelectric_phase", "phase", {"description": "Spontaneous electric polarization"}),
    ("paraelectric_phase", "phase", {"description": "No spontaneous polarization above Tc"}),
    ("austenite_phase", "phase", {"crystal": "FCC", "description": "High-T stable Fe"}),
    ("martensite_phase", "phase", {"crystal": "BCT", "description": "Metastable hard Fe"}),
    ("superconducting_phase", "phase", {"description": "Zero electrical resistance below Tc"}),
    ("normal_conducting_phase", "phase", {"description": "Finite resistance above Tc"}),
    ("metallic_phase", "phase", {"description": "High electrical conductivity"}),
    ("semiconducting_phase", "phase", {"description": "Valence/conduction band gap"}),
    ("amorphous_phase", "phase", {"description": "No long-range crystalline order"}),
    # Crystal structures
    ("bcc_structure", "crystal_structure", {"coordination_number": 8, "name": "BCC"}),
    ("fcc_structure", "crystal_structure", {"coordination_number": 12, "name": "FCC"}),
    ("bct_structure", "crystal_structure", {"name": "body-centered tetragonal"}),
    ("diamond_cubic", "crystal_structure", {"coordination_number": 4, "name": "diamond"}),
    ("perovskite_structure", "crystal_structure", {"name": "ABO3 perovskite"}),
    # Physical properties / conditions
    ("temperature", "property", {"unit": "K"}),
    ("pressure", "property", {"unit": "GPa"}),
    ("magnetic_field", "property", {"unit": "T"}),
    ("electrical_conductivity", "property", {"unit": "S/m"}),
    ("thermal_conductivity", "property", {"unit": "W/m·K"}),
    ("hardness", "property", {"unit": "HV (Vickers)"}),
    ("yield_strength", "property", {"unit": "MPa"}),
    ("band_gap", "property", {"unit": "eV"}),
    ("curie_temperature", "property", {"unit": "K", "symbol": "Tc"}),
    ("neel_temperature", "property", {"unit": "K", "symbol": "TN"}),
    ("critical_temperature", "property", {"unit": "K", "symbol": "Tc (SC)"}),
    # Phase transitions
    ("magnetic_phase_transition", "transition", {"drives": "FM→PM at Tc"}),
    ("martensitic_transformation", "transition", {"drives": "austenite→martensite"}),
    ("ferroelectric_transition", "transition", {"drives": "FE→PE at Tc"}),
    ("superconducting_transition", "transition", {"drives": "normal→SC at Tc"}),
    ("metal_insulator_transition", "transition", {"drives": "metallic→insulating"}),
    # Applications
    ("transformer_core", "application", {"requires": "low hysteresis, high permeability"}),
    ("structural_component", "application", {"requires": "high strength, ductility"}),
    ("semiconductor_device", "application", {"requires": "controlled band gap"}),
    ("capacitor_device", "application", {"requires": "high dielectric constant"}),
    ("superconducting_magnet", "application", {"requires": "zero resistance coil"}),
]

_EDGES: list[tuple[str, str, EdgeType]] = [
    # Material → crystal structure
    ("iron", "bcc_structure", EdgeType.MAPS_TO),
    ("aluminum", "fcc_structure", EdgeType.MAPS_TO),
    ("copper", "fcc_structure", EdgeType.MAPS_TO),
    ("silicon", "diamond_cubic", EdgeType.MAPS_TO),
    ("germanium", "diamond_cubic", EdgeType.MAPS_TO),
    ("austenite_phase", "fcc_structure", EdgeType.MAPS_TO),
    ("martensite_phase", "bct_structure", EdgeType.MAPS_TO),
    ("barium_titanate", "perovskite_structure", EdgeType.MAPS_TO),
    # Material → phase
    ("iron", "ferromagnetic_phase", EdgeType.MAPS_TO),
    ("manganese_oxide", "antiferromagnetic_phase", EdgeType.MAPS_TO),
    ("niobium_titanium", "superconducting_phase", EdgeType.MAPS_TO),
    ("yttrium_barium_copper_oxide", "superconducting_phase", EdgeType.MAPS_TO),
    ("barium_titanate", "ferroelectric_phase", EdgeType.MAPS_TO),
    ("silicon", "semiconducting_phase", EdgeType.MAPS_TO),
    ("germanium", "semiconducting_phase", EdgeType.MAPS_TO),
    ("gallium_arsenide", "semiconducting_phase", EdgeType.MAPS_TO),
    ("copper", "metallic_phase", EdgeType.MAPS_TO),
    ("iron", "metallic_phase", EdgeType.MAPS_TO),
    ("aluminum", "metallic_phase", EdgeType.MAPS_TO),
    # Steel derives from iron (composition/processing)
    ("steel", "iron", EdgeType.DERIVES),
    ("steel", "austenite_phase", EdgeType.DEPENDS_ON),
    ("steel", "martensite_phase", EdgeType.DEPENDS_ON),
    # Phase transition logic
    ("magnetic_phase_transition", "ferromagnetic_phase", EdgeType.EXPLAINS),
    ("magnetic_phase_transition", "paramagnetic_phase", EdgeType.EXPLAINS),
    ("magnetic_phase_transition", "curie_temperature", EdgeType.DEPENDS_ON),
    ("magnetic_phase_transition", "magnetic_field", EdgeType.DEPENDS_ON),
    ("martensitic_transformation", "austenite_phase", EdgeType.DERIVES),
    ("martensitic_transformation", "martensite_phase", EdgeType.DERIVES),
    ("martensitic_transformation", "temperature", EdgeType.DEPENDS_ON),
    ("ferroelectric_transition", "ferroelectric_phase", EdgeType.EXPLAINS),
    ("ferroelectric_transition", "paraelectric_phase", EdgeType.EXPLAINS),
    ("ferroelectric_transition", "curie_temperature", EdgeType.DEPENDS_ON),
    ("superconducting_transition", "superconducting_phase", EdgeType.EXPLAINS),
    ("superconducting_transition", "normal_conducting_phase", EdgeType.EXPLAINS),
    ("superconducting_transition", "critical_temperature", EdgeType.DEPENDS_ON),
    ("metal_insulator_transition", "pressure", EdgeType.DEPENDS_ON),
    ("metal_insulator_transition", "temperature", EdgeType.DEPENDS_ON),
    ("metal_insulator_transition", "metallic_phase", EdgeType.CONTRADICTS),
    # Phase → property
    ("semiconducting_phase", "band_gap", EdgeType.DEPENDS_ON),
    ("superconducting_phase", "critical_temperature", EdgeType.DEPENDS_ON),
    ("ferromagnetic_phase", "curie_temperature", EdgeType.DEPENDS_ON),
    ("antiferromagnetic_phase", "neel_temperature", EdgeType.DEPENDS_ON),
    ("metallic_phase", "electrical_conductivity", EdgeType.EXPLAINS),
    # Material → property
    ("silicon", "band_gap", EdgeType.DEPENDS_ON),
    ("germanium", "band_gap", EdgeType.DEPENDS_ON),
    ("gallium_arsenide", "band_gap", EdgeType.DEPENDS_ON),
    ("iron", "curie_temperature", EdgeType.DEPENDS_ON),
    # Applications
    ("transformer_core", "ferromagnetic_phase", EdgeType.DEPENDS_ON),
    ("structural_component", "yield_strength", EdgeType.DEPENDS_ON),
    ("structural_component", "hardness", EdgeType.DEPENDS_ON),
    ("semiconductor_device", "semiconducting_phase", EdgeType.DEPENDS_ON),
    ("capacitor_device", "ferroelectric_phase", EdgeType.DEPENDS_ON),
    ("superconducting_magnet", "superconducting_phase", EdgeType.DEPENDS_ON),
]


def build_materials_kg() -> KnowledgeGraph:
    """Build the phase diagrams / materials discovery knowledge graph."""
    kg = KnowledgeGraph(domain="materials")
    for eid, etype, props in _ENTITIES:
        kg.add_entity(Entity(id=eid, entity_type=etype, properties=props))
    for src, tgt, etype in _EDGES:
        kg.add_edge(TypedEdge(source=src, target=tgt, edge_type=etype))
    return kg
