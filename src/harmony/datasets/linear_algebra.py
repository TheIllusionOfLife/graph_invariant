"""Calibration dataset: linear algebra micro-ontology.

~50 entities (concepts + theorems) and ~65 typed edges encoding the
dependency structure of a standard linear algebra course. Used to
calibrate the Harmony metric because the "correct" KG structure is
well-established and admits ground-truth masked-edge recovery.
"""
from __future__ import annotations

from ..types import EdgeType, Entity, KnowledgeGraph, TypedEdge

_CONCEPTS = [
    # Core algebraic structures
    ("vector_space", "concept", {"description": "Set with vector addition and scalar mult."}),
    ("field", "concept", {"description": "Algebraic structure with + and × operations"}),
    ("linear_map", "concept", {"description": "Structure-preserving map between vector spaces"}),
    ("matrix", "concept", {"description": "Rectangular array representing a linear map in bases"}),
    # Fundamental subspace concepts
    ("rank", "concept", {"description": "Dimension of the image of a linear map"}),
    ("nullity", "concept", {"description": "Dimension of the kernel of a linear map"}),
    ("nullspace", "concept", {"description": "Kernel: set of vectors mapping to zero"}),
    ("image_space", "concept", {"description": "Range/column space of a linear map"}),
    # Basis and dimension
    ("basis", "concept", {"description": "Linearly independent spanning set"}),
    ("dimension", "concept", {"description": "Cardinality of a basis"}),
    ("linear_independence", "concept", {"description": "No non-trivial combination sums to zero"}),
    ("span", "concept", {"description": "Set of all linear combinations of a set of vectors"}),
    # Eigentheory
    ("eigenvalue", "concept", {"description": "Scalar λ such that Av = λv"}),
    ("eigenvector", "concept", {"description": "Non-zero vector satisfying Av = λv"}),
    ("eigenspace", "concept", {"description": "Nullspace of (A - λI)"}),
    ("characteristic_polynomial", "concept", {"description": "det(A - λI); roots are eigenvalues"}),
    ("diagonalizability", "concept", {"description": "Whether a matrix has a full eigenbasis"}),
    # Matrix properties
    ("determinant", "concept", {"description": "Scalar encoding volume scaling of linear map"}),
    ("trace", "concept", {"description": "Sum of diagonal entries; sum of eigenvalues"}),
    ("invertible_matrix", "concept", {"description": "Square matrix with non-zero determinant"}),
    ("singular_matrix", "concept", {"description": "Square matrix with zero determinant"}),
    ("symmetric_matrix", "concept", {"description": "Matrix equal to its transpose: A = Aᵀ"}),
    ("positive_definite", "concept", {"description": "Symmetric matrix with positive eigenvals"}),
    ("normal_matrix", "concept", {"description": "Matrix commuting with its adjoint: AA* = A*A"}),
    ("adjoint", "concept", {"description": "Conjugate transpose of a matrix"}),
    # Inner product structure
    ("inner_product", "concept", {"description": "Bilinear form encoding angles and lengths"}),
    ("norm", "concept", {"description": "Length function induced by inner product"}),
    ("orthogonality", "concept", {"description": "Zero inner product between two vectors"}),
    ("orthonormal_basis", "concept", {"description": "Basis of unit-length orthogonal vectors"}),
    ("projection", "concept", {"description": "Idempotent linear map onto a subspace"}),
    # Structural concepts
    ("dual_space", "concept", {"description": "Space of all linear functionals on V"}),
    ("linear_functional", "concept", {"description": "Linear map from V to its scalar field"}),
    ("direct_sum", "concept", {"description": "Decomposition V = U ⊕ W"}),
    ("isomorphism", "concept", {"description": "Bijective linear map; structure-preserving"}),
    ("quotient_space", "concept", {"description": "V modulo a subspace W"}),
]

_THEOREMS = [
    ("rank_nullity_theorem", "theorem", {"statement": "dim(nullspace) + dim(image) = dim(domain)"}),
    ("spectral_theorem", "theorem", {"statement": "Symmetric matrix is orthogonally diag."}),
    ("cayley_hamilton_theorem", "theorem", {"statement": "Matrix satisfies its char. polynomial"}),
    ("jordan_normal_form", "theorem", {"statement": "Matrix ~ Jordan block diagonal form"}),
    ("singular_value_decomposition", "theorem", {"statement": "A = UΣVᵀ for orthogonal U, V"}),
    ("gram_schmidt_process", "theorem", {"statement": "Converts any basis to orthonormal basis"}),
    ("dimension_theorem", "theorem", {"statement": "All bases of a vector space have same dim"}),
    ("cauchy_schwarz_inequality", "theorem", {"statement": "|⟨u,v⟩| ≤ ‖u‖‖v‖"}),
    ("qr_decomposition", "theorem", {"statement": "A = QR for orthogonal Q, upper-triangular R"}),
    ("lu_decomposition", "theorem", {"statement": "Invertible A = LU for triangular L, U"}),
    ("fundamental_theorem_linear_maps", "theorem", {
        "statement": "V/ker(T) is isomorphic to im(T); equivalent to rank-nullity"
    }),
    ("rouche_capelli_theorem", "theorem", {
        "statement": "Ax=b consistent iff rank(A) = rank(A|b)"
    }),
    ("spectral_decomposition", "theorem", {"statement": "Symmetric A = QΛQᵀ for orthogonal Q"}),
    ("orthogonal_decomposition", "theorem", {"statement": "V = W ⊕ W⊥ for any subspace W"}),
    ("min_poly_divides_char_poly", "theorem", {
        "statement": "The minimal polynomial divides the characteristic polynomial"
    }),
]

# (source_id, target_id, edge_type)
_EDGES: list[tuple[str, str, EdgeType]] = [
    # --- Core structure ---
    ("vector_space", "field", EdgeType.DEPENDS_ON),
    ("linear_map", "vector_space", EdgeType.DEPENDS_ON),
    ("matrix", "linear_map", EdgeType.MAPS_TO),          # matrix represents a linear map in bases
    ("basis", "vector_space", EdgeType.DEPENDS_ON),
    ("dimension", "basis", EdgeType.DEPENDS_ON),
    ("span", "vector_space", EdgeType.DEPENDS_ON),
    ("linear_independence", "span", EdgeType.DEPENDS_ON),
    ("basis", "linear_independence", EdgeType.DEPENDS_ON),
    # --- Subspaces ---
    ("nullspace", "linear_map", EdgeType.DEPENDS_ON),
    ("image_space", "linear_map", EdgeType.DEPENDS_ON),
    ("rank", "image_space", EdgeType.DEPENDS_ON),
    ("nullity", "nullspace", EdgeType.DEPENDS_ON),
    ("quotient_space", "vector_space", EdgeType.DEPENDS_ON),
    # --- Eigentheory ---
    ("eigenvalue", "matrix", EdgeType.DEPENDS_ON),
    ("eigenvector", "eigenvalue", EdgeType.DEPENDS_ON),
    ("eigenspace", "eigenvalue", EdgeType.DEPENDS_ON),
    ("eigenspace", "nullspace", EdgeType.EQUIVALENT_TO),  # eigenspace IS nullspace of (A-λI)
    ("characteristic_polynomial", "matrix", EdgeType.DEPENDS_ON),
    ("eigenvalue", "characteristic_polynomial", EdgeType.DERIVES),
    ("diagonalizability", "eigenvalue", EdgeType.DEPENDS_ON),
    ("trace", "eigenvalue", EdgeType.EQUIVALENT_TO),      # trace = sum of eigenvalues
    # --- Matrix scalars ---
    ("determinant", "matrix", EdgeType.DEPENDS_ON),
    ("invertible_matrix", "determinant", EdgeType.DEPENDS_ON),
    ("singular_matrix", "determinant", EdgeType.DEPENDS_ON),
    ("invertible_matrix", "singular_matrix", EdgeType.CONTRADICTS),
    # --- Symmetry / adjoint ---
    ("adjoint", "matrix", EdgeType.DEPENDS_ON),
    ("adjoint", "inner_product", EdgeType.DEPENDS_ON),
    ("symmetric_matrix", "adjoint", EdgeType.DEPENDS_ON),
    ("normal_matrix", "adjoint", EdgeType.DEPENDS_ON),
    ("positive_definite", "symmetric_matrix", EdgeType.DEPENDS_ON),
    ("positive_definite", "inner_product", EdgeType.EQUIVALENT_TO),
    # --- Inner product structure ---
    ("inner_product", "vector_space", EdgeType.DEPENDS_ON),
    ("norm", "inner_product", EdgeType.DERIVES),
    ("orthogonality", "inner_product", EdgeType.DEPENDS_ON),
    ("orthonormal_basis", "orthogonality", EdgeType.DEPENDS_ON),
    ("orthonormal_basis", "basis", EdgeType.GENERALIZES),
    ("projection", "orthonormal_basis", EdgeType.DEPENDS_ON),
    ("cauchy_schwarz_inequality", "inner_product", EdgeType.DEPENDS_ON),
    ("cauchy_schwarz_inequality", "norm", EdgeType.DERIVES),
    # --- Dual space ---
    ("dual_space", "vector_space", EdgeType.DEPENDS_ON),
    ("linear_functional", "dual_space", EdgeType.DEPENDS_ON),
    ("linear_functional", "linear_map", EdgeType.GENERALIZES),
    # --- Structural ---
    ("direct_sum", "vector_space", EdgeType.DEPENDS_ON),
    ("isomorphism", "linear_map", EdgeType.DEPENDS_ON),
    ("isomorphism", "invertible_matrix", EdgeType.EQUIVALENT_TO),
    # --- Theorems: derivation chains ---
    ("rank_nullity_theorem", "rank", EdgeType.EXPLAINS),
    ("rank_nullity_theorem", "nullity", EdgeType.EXPLAINS),
    ("rank_nullity_theorem", "dimension", EdgeType.DEPENDS_ON),
    ("fundamental_theorem_linear_maps", "rank_nullity_theorem", EdgeType.EQUIVALENT_TO),
    ("fundamental_theorem_linear_maps", "isomorphism", EdgeType.DERIVES),
    ("dimension_theorem", "basis", EdgeType.DERIVES),
    ("dimension_theorem", "dimension", EdgeType.EXPLAINS),
    ("spectral_theorem", "symmetric_matrix", EdgeType.DEPENDS_ON),
    ("spectral_theorem", "eigenvalue", EdgeType.EXPLAINS),
    ("spectral_theorem", "diagonalizability", EdgeType.EXPLAINS),
    ("spectral_decomposition", "spectral_theorem", EdgeType.DERIVES),
    ("spectral_decomposition", "orthonormal_basis", EdgeType.DEPENDS_ON),
    ("cayley_hamilton_theorem", "characteristic_polynomial", EdgeType.DEPENDS_ON),
    ("min_poly_divides_char_poly", "characteristic_polynomial", EdgeType.DEPENDS_ON),
    ("min_poly_divides_char_poly", "cayley_hamilton_theorem", EdgeType.DERIVES),
    ("jordan_normal_form", "eigenvalue", EdgeType.DEPENDS_ON),
    ("jordan_normal_form", "characteristic_polynomial", EdgeType.DEPENDS_ON),
    ("gram_schmidt_process", "inner_product", EdgeType.DEPENDS_ON),
    ("gram_schmidt_process", "orthonormal_basis", EdgeType.DERIVES),
    ("qr_decomposition", "gram_schmidt_process", EdgeType.DERIVES),
    ("qr_decomposition", "matrix", EdgeType.EXPLAINS),
    ("lu_decomposition", "invertible_matrix", EdgeType.DEPENDS_ON),
    ("lu_decomposition", "matrix", EdgeType.EXPLAINS),
    ("singular_value_decomposition", "matrix", EdgeType.EXPLAINS),
    ("singular_value_decomposition", "rank", EdgeType.EXPLAINS),
    ("singular_value_decomposition", "orthonormal_basis", EdgeType.DEPENDS_ON),
    ("orthogonal_decomposition", "projection", EdgeType.DERIVES),
    ("orthogonal_decomposition", "direct_sum", EdgeType.EQUIVALENT_TO),
    ("rouche_capelli_theorem", "rank", EdgeType.DEPENDS_ON),
    ("rouche_capelli_theorem", "dimension_theorem", EdgeType.DERIVES),
]


def build_linear_algebra_kg() -> KnowledgeGraph:
    """Build the linear algebra calibration knowledge graph."""
    kg = KnowledgeGraph(domain="linear_algebra")

    for eid, etype, props in _CONCEPTS + _THEOREMS:
        kg.add_entity(Entity(id=eid, entity_type=etype, properties=props))

    for src, tgt, etype in _EDGES:
        kg.add_edge(TypedEdge(source=src, target=tgt, edge_type=etype))

    return kg
