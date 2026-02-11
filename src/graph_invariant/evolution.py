from dataclasses import replace

from .types import Candidate, CheckpointState


def _best_candidate(candidates: list[Candidate]) -> Candidate:
    return max(candidates, key=lambda c: c.val_score)


def _replace_if_better(target_island: list[Candidate], incoming: Candidate) -> list[Candidate]:
    if not target_island:
        return [incoming]
    worst_idx = min(range(len(target_island)), key=lambda i: target_island[i].val_score)
    if incoming.val_score > target_island[worst_idx].val_score:
        updated = list(target_island)
        updated[worst_idx] = incoming
        return updated
    return list(target_island)


def migrate_ring_top1(state: CheckpointState) -> CheckpointState:
    new_islands = {k: list(v) for k, v in state.islands.items()}
    order = sorted(new_islands.keys())
    for island_id in order:
        src_candidates = state.islands.get(island_id, [])
        if not src_candidates:
            continue
        incoming = _best_candidate(src_candidates)
        next_idx = (order.index(island_id) + 1) % len(order)
        dst_id = order[next_idx]
        new_islands[dst_id] = _replace_if_better(new_islands.get(dst_id, []), incoming)
    return replace(state, islands=new_islands)
