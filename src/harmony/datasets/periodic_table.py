"""Calibration dataset: periodic table knowledge graph.

118 elements as entities with structural edges encoding:
  - element MAPS_TO period
  - element MAPS_TO group (where applicable)
  - category GENERALIZES element

Used for calibration because the structure is maximally "harmonious"
(Mendeleev's great insight: periodic law + group/period organisation).
"""
from __future__ import annotations

from ..types import EdgeType, Entity, KnowledgeGraph, TypedEdge

# (atomic_number, symbol, name, period, group_or_None, category)
# group=None for lanthanides (Z=58-71) and actinides (Z=90-103)
_ELEMENTS_DATA: list[tuple[int, str, str, int, int | None, str]] = [
    (1,   "H",  "Hydrogen",      1, 1,    "nonmetal"),
    (2,   "He", "Helium",        1, 18,   "noble_gas"),
    (3,   "Li", "Lithium",       2, 1,    "alkali_metal"),
    (4,   "Be", "Beryllium",     2, 2,    "alkaline_earth"),
    (5,   "B",  "Boron",         2, 13,   "metalloid"),
    (6,   "C",  "Carbon",        2, 14,   "nonmetal"),
    (7,   "N",  "Nitrogen",      2, 15,   "nonmetal"),
    (8,   "O",  "Oxygen",        2, 16,   "nonmetal"),
    (9,   "F",  "Fluorine",      2, 17,   "halogen"),
    (10,  "Ne", "Neon",          2, 18,   "noble_gas"),
    (11,  "Na", "Sodium",        3, 1,    "alkali_metal"),
    (12,  "Mg", "Magnesium",     3, 2,    "alkaline_earth"),
    (13,  "Al", "Aluminum",      3, 13,   "post_transition_metal"),
    (14,  "Si", "Silicon",       3, 14,   "metalloid"),
    (15,  "P",  "Phosphorus",    3, 15,   "nonmetal"),
    (16,  "S",  "Sulfur",        3, 16,   "nonmetal"),
    (17,  "Cl", "Chlorine",      3, 17,   "halogen"),
    (18,  "Ar", "Argon",         3, 18,   "noble_gas"),
    (19,  "K",  "Potassium",     4, 1,    "alkali_metal"),
    (20,  "Ca", "Calcium",       4, 2,    "alkaline_earth"),
    (21,  "Sc", "Scandium",      4, 3,    "transition_metal"),
    (22,  "Ti", "Titanium",      4, 4,    "transition_metal"),
    (23,  "V",  "Vanadium",      4, 5,    "transition_metal"),
    (24,  "Cr", "Chromium",      4, 6,    "transition_metal"),
    (25,  "Mn", "Manganese",     4, 7,    "transition_metal"),
    (26,  "Fe", "Iron",          4, 8,    "transition_metal"),
    (27,  "Co", "Cobalt",        4, 9,    "transition_metal"),
    (28,  "Ni", "Nickel",        4, 10,   "transition_metal"),
    (29,  "Cu", "Copper",        4, 11,   "transition_metal"),
    (30,  "Zn", "Zinc",          4, 12,   "transition_metal"),
    (31,  "Ga", "Gallium",       4, 13,   "post_transition_metal"),
    (32,  "Ge", "Germanium",     4, 14,   "metalloid"),
    (33,  "As", "Arsenic",       4, 15,   "metalloid"),
    (34,  "Se", "Selenium",      4, 16,   "nonmetal"),
    (35,  "Br", "Bromine",       4, 17,   "halogen"),
    (36,  "Kr", "Krypton",       4, 18,   "noble_gas"),
    (37,  "Rb", "Rubidium",      5, 1,    "alkali_metal"),
    (38,  "Sr", "Strontium",     5, 2,    "alkaline_earth"),
    (39,  "Y",  "Yttrium",       5, 3,    "transition_metal"),
    (40,  "Zr", "Zirconium",     5, 4,    "transition_metal"),
    (41,  "Nb", "Niobium",       5, 5,    "transition_metal"),
    (42,  "Mo", "Molybdenum",    5, 6,    "transition_metal"),
    (43,  "Tc", "Technetium",    5, 7,    "transition_metal"),
    (44,  "Ru", "Ruthenium",     5, 8,    "transition_metal"),
    (45,  "Rh", "Rhodium",       5, 9,    "transition_metal"),
    (46,  "Pd", "Palladium",     5, 10,   "transition_metal"),
    (47,  "Ag", "Silver",        5, 11,   "transition_metal"),
    (48,  "Cd", "Cadmium",       5, 12,   "transition_metal"),
    (49,  "In", "Indium",        5, 13,   "post_transition_metal"),
    (50,  "Sn", "Tin",           5, 14,   "post_transition_metal"),
    (51,  "Sb", "Antimony",      5, 15,   "metalloid"),
    (52,  "Te", "Tellurium",     5, 16,   "metalloid"),
    (53,  "I",  "Iodine",        5, 17,   "halogen"),
    (54,  "Xe", "Xenon",         5, 18,   "noble_gas"),
    (55,  "Cs", "Cesium",        6, 1,    "alkali_metal"),
    (56,  "Ba", "Barium",        6, 2,    "alkaline_earth"),
    (57,  "La", "Lanthanum",     6, 3,    "lanthanide"),
    (58,  "Ce", "Cerium",        6, None, "lanthanide"),
    (59,  "Pr", "Praseodymium",  6, None, "lanthanide"),
    (60,  "Nd", "Neodymium",     6, None, "lanthanide"),
    (61,  "Pm", "Promethium",    6, None, "lanthanide"),
    (62,  "Sm", "Samarium",      6, None, "lanthanide"),
    (63,  "Eu", "Europium",      6, None, "lanthanide"),
    (64,  "Gd", "Gadolinium",    6, None, "lanthanide"),
    (65,  "Tb", "Terbium",       6, None, "lanthanide"),
    (66,  "Dy", "Dysprosium",    6, None, "lanthanide"),
    (67,  "Ho", "Holmium",       6, None, "lanthanide"),
    (68,  "Er", "Erbium",        6, None, "lanthanide"),
    (69,  "Tm", "Thulium",       6, None, "lanthanide"),
    (70,  "Yb", "Ytterbium",     6, None, "lanthanide"),
    (71,  "Lu", "Lutetium",      6, None, "lanthanide"),
    (72,  "Hf", "Hafnium",       6, 4,    "transition_metal"),
    (73,  "Ta", "Tantalum",      6, 5,    "transition_metal"),
    (74,  "W",  "Tungsten",      6, 6,    "transition_metal"),
    (75,  "Re", "Rhenium",       6, 7,    "transition_metal"),
    (76,  "Os", "Osmium",        6, 8,    "transition_metal"),
    (77,  "Ir", "Iridium",       6, 9,    "transition_metal"),
    (78,  "Pt", "Platinum",      6, 10,   "transition_metal"),
    (79,  "Au", "Gold",          6, 11,   "transition_metal"),
    (80,  "Hg", "Mercury",       6, 12,   "transition_metal"),
    (81,  "Tl", "Thallium",      6, 13,   "post_transition_metal"),
    (82,  "Pb", "Lead",          6, 14,   "post_transition_metal"),
    (83,  "Bi", "Bismuth",       6, 15,   "post_transition_metal"),
    (84,  "Po", "Polonium",      6, 16,   "metalloid"),
    (85,  "At", "Astatine",      6, 17,   "halogen"),
    (86,  "Rn", "Radon",         6, 18,   "noble_gas"),
    (87,  "Fr", "Francium",      7, 1,    "alkali_metal"),
    (88,  "Ra", "Radium",        7, 2,    "alkaline_earth"),
    (89,  "Ac", "Actinium",      7, 3,    "actinide"),
    (90,  "Th", "Thorium",       7, None, "actinide"),
    (91,  "Pa", "Protactinium",  7, None, "actinide"),
    (92,  "U",  "Uranium",       7, None, "actinide"),
    (93,  "Np", "Neptunium",     7, None, "actinide"),
    (94,  "Pu", "Plutonium",     7, None, "actinide"),
    (95,  "Am", "Americium",     7, None, "actinide"),
    (96,  "Cm", "Curium",        7, None, "actinide"),
    (97,  "Bk", "Berkelium",     7, None, "actinide"),
    (98,  "Cf", "Californium",   7, None, "actinide"),
    (99,  "Es", "Einsteinium",   7, None, "actinide"),
    (100, "Fm", "Fermium",       7, None, "actinide"),
    (101, "Md", "Mendelevium",   7, None, "actinide"),
    (102, "No", "Nobelium",      7, None, "actinide"),
    (103, "Lr", "Lawrencium",    7, None, "actinide"),
    (104, "Rf", "Rutherfordium", 7, 4,    "transition_metal"),
    (105, "Db", "Dubnium",       7, 5,    "transition_metal"),
    (106, "Sg", "Seaborgium",    7, 6,    "transition_metal"),
    (107, "Bh", "Bohrium",       7, 7,    "transition_metal"),
    (108, "Hs", "Hassium",       7, 8,    "transition_metal"),
    (109, "Mt", "Meitnerium",    7, 9,    "transition_metal"),
    (110, "Ds", "Darmstadtium",  7, 10,   "transition_metal"),
    (111, "Rg", "Roentgenium",   7, 11,   "transition_metal"),
    (112, "Cn", "Copernicium",   7, 12,   "transition_metal"),
    (113, "Nh", "Nihonium",      7, 13,   "post_transition_metal"),
    (114, "Fl", "Flerovium",     7, 14,   "post_transition_metal"),
    (115, "Mc", "Moscovium",     7, 15,   "post_transition_metal"),
    (116, "Lv", "Livermorium",   7, 16,   "post_transition_metal"),
    (117, "Ts", "Tennessine",    7, 17,   "halogen"),
    (118, "Og", "Oganesson",     7, 18,   "noble_gas"),
]

_CATEGORIES = [
    "alkali_metal",
    "alkaline_earth",
    "transition_metal",
    "post_transition_metal",
    "metalloid",
    "nonmetal",
    "halogen",
    "noble_gas",
    "lanthanide",
    "actinide",
]


def _element_id(symbol: str) -> str:
    return f"elem_{symbol.lower()}"


def _period_id(period: int) -> str:
    return f"period_{period}"


def _group_id(group: int) -> str:
    return f"group_{group}"


def _category_id(cat: str) -> str:
    return f"cat_{cat}"


def build_periodic_table_kg() -> KnowledgeGraph:
    """Build the periodic table calibration knowledge graph."""
    kg = KnowledgeGraph(domain="periodic_table")

    # Period entities (1–7)
    for p in range(1, 8):
        kg.add_entity(Entity(
            id=_period_id(p),
            entity_type="period",
            properties={"period_number": p},
        ))

    # Group entities (1–18)
    for g in range(1, 19):
        kg.add_entity(Entity(
            id=_group_id(g),
            entity_type="group",
            properties={"group_number": g},
        ))

    # Category entities
    for cat in _CATEGORIES:
        kg.add_entity(Entity(
            id=_category_id(cat),
            entity_type="category",
            properties={"name": cat},
        ))

    # Element entities + structural edges
    for z, sym, name, period, group, category in _ELEMENTS_DATA:
        eid = _element_id(sym)
        kg.add_entity(Entity(
            id=eid,
            entity_type="element",
            properties={
                "atomic_number": z,
                "symbol": sym,
                "name": name,
                "period": period,
                "group": group,
                "category": category,
            },
        ))
        # element maps_to its period
        kg.add_edge(TypedEdge(
            source=eid,
            target=_period_id(period),
            edge_type=EdgeType.MAPS_TO,
        ))
        # element maps_to its group (if defined)
        if group is not None:
            kg.add_edge(TypedEdge(
                source=eid,
                target=_group_id(group),
                edge_type=EdgeType.MAPS_TO,
            ))
        # category generalizes element
        kg.add_edge(TypedEdge(
            source=_category_id(category),
            target=eid,
            edge_type=EdgeType.GENERALIZES,
        ))

    return kg
