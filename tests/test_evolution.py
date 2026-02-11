from graph_invariant.evolution import migrate_ring_top1
from graph_invariant.types import Candidate, CheckpointState


def test_migrate_ring_top1_moves_best_candidate_when_better():
    state = CheckpointState(
        generation=10,
        islands={
            0: [Candidate(id="a", code="def new_invariant(G):\n    return 1", val_score=0.9)],
            1: [Candidate(id="b", code="def new_invariant(G):\n    return 1", val_score=0.2)],
            2: [Candidate(id="c", code="def new_invariant(G):\n    return 1", val_score=0.1)],
            3: [Candidate(id="d", code="def new_invariant(G):\n    return 1", val_score=0.3)],
        },
        rng_seed=1,
    )
    next_state = migrate_ring_top1(state)
    assert any(c.id == "a" for c in next_state.islands[1])
