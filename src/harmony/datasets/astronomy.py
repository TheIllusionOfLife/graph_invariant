"""Discovery dataset: exoplanet demographics knowledge graph.

Encodes planet types, stellar types, formation mechanisms, observed
phenomena, and physical properties. Several "explains" edges between
formation mechanisms and observed distributions are intentionally
sparse — these are the targets the proposal engine should discover.
"""
from __future__ import annotations

from ..types import EdgeType, Entity, KnowledgeGraph, TypedEdge

_ENTITIES: list[tuple[str, str, dict]] = [
    # Planet types
    ("hot_jupiter",       "planet_type", {"typical_period_days": 3.0,  "radius_earth": 11.0}),
    ("warm_jupiter",      "planet_type", {"typical_period_days": 30.0, "radius_earth": 11.0}),
    ("cold_giant",        "planet_type", {"typical_period_days": 4000, "radius_earth": 11.0}),
    ("sub_neptune",       "planet_type", {"typical_period_days": 20.0, "radius_earth": 2.5}),
    ("mini_neptune",      "planet_type", {"typical_period_days": 20.0, "radius_earth": 2.0}),
    ("super_earth",       "planet_type", {"typical_period_days": 15.0, "radius_earth": 1.5}),
    ("earth_analog",      "planet_type", {"typical_period_days": 365,  "radius_earth": 1.0}),
    ("ocean_world",       "planet_type", {"water_mass_fraction": 0.5}),
    ("lava_world",        "planet_type", {"equilibrium_temp_k": 2500}),
    ("rogue_planet",      "planet_type", {"bound_to_star": False}),
    # Stellar types
    ("solar_type_star",   "stellar_type", {"spectral_class": "G", "mass_solar": 1.0}),
    ("k_dwarf",           "stellar_type", {"spectral_class": "K", "mass_solar": 0.7}),
    ("red_dwarf",         "stellar_type", {"spectral_class": "M", "mass_solar": 0.3}),
    ("hot_a_star",        "stellar_type", {"spectral_class": "A", "mass_solar": 2.0}),
    ("subgiant",          "stellar_type", {"evolved": True}),
    ("giant_star",        "stellar_type", {"evolved": True, "radius_solar": 10}),
    # Formation mechanisms
    ("core_accretion",    "mechanism", {"description": "Rocky core grows then accretes gas"}),
    ("gravitational_instability", "mechanism", {"description": "Disk fragments → giant planets"}),
    ("disk_migration",    "mechanism", {"description": "Inward migration via disk torques"}),
    ("photoevaporation",  "mechanism", {"description": "XUV radiation strips atmospheres"}),
    ("giant_impacts",     "mechanism", {"description": "Late-stage embryo collisions"}),
    ("pebble_accretion",  "mechanism", {"description": "cm pebbles drift and are accreted"}),
    # Observed phenomena / patterns
    ("radius_gap",        "phenomenon", {"location_earth_radii": 1.8}),
    ("hot_desert",        "phenomenon", {"description": "No hot Neptunes at short periods"}),
    ("occurrence_rate",   "phenomenon", {"description": "Planet rate per stellar type"}),
    ("mass_radius_relation", "phenomenon", {"description": "Empirical M-R power law"}),
    ("stellar_metallicity_correlation", "phenomenon", {"description": "Metal-rich host effect"}),
    ("period_distribution", "phenomenon", {"description": "Distribution of orbital periods"}),
    ("planet_multiplicity", "phenomenon", {"description": "Number of planets per system"}),
    # Physical properties / observables
    ("orbital_period",       "property", {"unit": "days"}),
    ("semi_major_axis",      "property", {"unit": "AU"}),
    ("planet_radius",        "property", {"unit": "R_earth"}),
    ("planet_mass",          "property", {"unit": "M_earth"}),
    ("stellar_mass",         "property", {"unit": "M_sun"}),
    ("stellar_metallicity",  "property", {"unit": "dex [Fe/H]"}),
    ("equilibrium_temperature", "property", {"unit": "K"}),
    ("insolation_flux",      "property", {"unit": "S_earth"}),
    ("eccentricity",         "property", {"range": [0, 1]}),
    # Surveys
    ("kepler_survey",     "survey", {"n_planets": 4000, "method": "transit"}),
    ("tess_survey",       "survey", {"method": "transit"}),
    ("radial_velocity_survey", "survey", {"method": "radial_velocity"}),
]

_EDGES: list[tuple[str, str, EdgeType]] = [
    # Formation mechanism → planet type (core of the science)
    ("core_accretion",   "super_earth",    EdgeType.EXPLAINS),
    ("core_accretion",   "earth_analog",   EdgeType.EXPLAINS),
    ("core_accretion",   "sub_neptune",    EdgeType.EXPLAINS),
    ("pebble_accretion", "cold_giant",     EdgeType.EXPLAINS),
    ("gravitational_instability", "cold_giant",  EdgeType.EXPLAINS),
    ("gravitational_instability", "rogue_planet", EdgeType.EXPLAINS),
    ("disk_migration",   "hot_jupiter",    EdgeType.EXPLAINS),
    ("photoevaporation", "radius_gap",     EdgeType.EXPLAINS),
    ("giant_impacts",    "earth_analog",   EdgeType.EXPLAINS),
    # pebble_accretion generalises core_accretion (it's a faster variant of the same paradigm)
    ("core_accretion",   "pebble_accretion", EdgeType.GENERALIZES),
    # Stellar evolution
    ("subgiant",         "giant_star",     EdgeType.DERIVES),
    ("solar_type_star",  "subgiant",       EdgeType.DERIVES),
    # Planet type dependencies on mechanisms / phenomena
    ("hot_jupiter",      "disk_migration", EdgeType.DEPENDS_ON),
    ("mini_neptune",     "photoevaporation", EdgeType.DEPENDS_ON),
    ("radius_gap",       "super_earth",    EdgeType.DEPENDS_ON),
    ("radius_gap",       "sub_neptune",    EdgeType.DEPENDS_ON),
    ("hot_desert",       "hot_jupiter",    EdgeType.DEPENDS_ON),
    ("hot_desert",       "photoevaporation", EdgeType.DEPENDS_ON),
    # Phenomenon dependencies on properties
    ("mass_radius_relation",  "planet_radius", EdgeType.DEPENDS_ON),
    ("mass_radius_relation",  "planet_mass",   EdgeType.DEPENDS_ON),
    ("period_distribution",   "orbital_period", EdgeType.DEPENDS_ON),
    ("period_distribution",   "disk_migration", EdgeType.DEPENDS_ON),
    ("occurrence_rate",       "stellar_metallicity", EdgeType.DEPENDS_ON),
    ("stellar_metallicity_correlation", "stellar_metallicity", EdgeType.DEPENDS_ON),
    ("stellar_metallicity_correlation", "core_accretion", EdgeType.EXPLAINS),
    ("planet_multiplicity",   "occurrence_rate", EdgeType.DEPENDS_ON),
    # Physical property relationships
    ("equilibrium_temperature", "orbital_period",  EdgeType.DEPENDS_ON),
    ("equilibrium_temperature", "stellar_mass",    EdgeType.DEPENDS_ON),
    ("equilibrium_temperature", "insolation_flux", EdgeType.EQUIVALENT_TO),
    ("ocean_world",    "equilibrium_temperature", EdgeType.DEPENDS_ON),
    ("lava_world",     "equilibrium_temperature", EdgeType.DEPENDS_ON),
    ("earth_analog",   "insolation_flux",         EdgeType.DEPENDS_ON),
    # Survey → phenomena
    ("kepler_survey",         "radius_gap",      EdgeType.EXPLAINS),
    ("kepler_survey",         "occurrence_rate", EdgeType.MAPS_TO),
    ("tess_survey",           "occurrence_rate", EdgeType.MAPS_TO),
    ("radial_velocity_survey","stellar_metallicity_correlation", EdgeType.MAPS_TO),
    ("radial_velocity_survey","mass_radius_relation",           EdgeType.MAPS_TO),
    # Property chains
    ("orbital_period",  "semi_major_axis",  EdgeType.EQUIVALENT_TO),
    ("semi_major_axis", "equilibrium_temperature", EdgeType.EXPLAINS),
]


def build_astronomy_kg() -> KnowledgeGraph:
    """Build the exoplanet demographics discovery knowledge graph."""
    kg = KnowledgeGraph(domain="astronomy")
    for eid, etype, props in _ENTITIES:
        kg.add_entity(Entity(id=eid, entity_type=etype, properties=props))
    for src, tgt, etype in _EDGES:
        kg.add_edge(TypedEdge(source=src, target=tgt, edge_type=etype))
    return kg
