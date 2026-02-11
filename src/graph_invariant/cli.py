import argparse
from pathlib import Path

import networkx as nx

from .config import Phase1Config
from .data import generate_phase1_datasets
from .logging_io import append_jsonl, load_checkpoint, save_checkpoint
from .sandbox import evaluate_candidate_on_graphs
from .scoring import compute_metrics, compute_simplicity_score, compute_total_score
from .types import Candidate, CheckpointState


def _target_values(graphs: list[nx.Graph], target_name: str) -> list[float]:
    if target_name == "average_shortest_path_length":
        return [nx.average_shortest_path_length(g) for g in graphs]
    if target_name == "diameter":
        return [float(nx.diameter(g)) for g in graphs]
    raise ValueError(f"unsupported target: {target_name}")


def run_phase1(cfg: Phase1Config, resume: str | None = None) -> int:
    datasets = generate_phase1_datasets(cfg)
    artifacts_dir = Path(cfg.artifacts_dir)
    log_path = artifacts_dir / "logs" / "events.jsonl"
    ckpt_path = artifacts_dir / "checkpoints" / "phase1.json"

    if resume:
        state = load_checkpoint(resume)
    else:
        state = CheckpointState(generation=0, islands={i: [] for i in range(4)}, rng_seed=cfg.seed)

    candidate_code = (
        "def new_invariant(G):\n"
        "    n = G.number_of_nodes()\n"
        "    m = G.number_of_edges()\n"
        "    return (n + 1.0) / (m + 1.0)"
    )
    y_true = _target_values(datasets.val, cfg.target_name)
    y_pred_raw = evaluate_candidate_on_graphs(
        candidate_code, datasets.val, timeout_sec=cfg.timeout_sec, memory_mb=cfg.memory_mb
    )
    pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred_raw, strict=False) if yp is not None]
    if not pairs:
        raise RuntimeError("candidate produced no valid values")
    y_t, y_p = zip(*pairs, strict=False)
    metrics = compute_metrics(list(y_t), list(y_p))
    simplicity = compute_simplicity_score(candidate_code)
    total = compute_total_score(abs(metrics.rho_spearman), simplicity, novelty_bonus=1.0)
    candidate = Candidate(
        id="seed_candidate",
        code=candidate_code,
        island_id=0,
        generation=state.generation,
        val_score=total,
        simplicity_score=simplicity,
        novelty_bonus=1.0,
    )
    state.islands[0] = [candidate]
    append_jsonl(
        "candidate_evaluated",
        {
            "candidate_id": candidate.id,
            "spearman": metrics.rho_spearman,
            "pearson": metrics.r_pearson,
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "total_score": total,
        },
        log_path,
    )
    save_checkpoint(state, ckpt_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Graph invariant discovery CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    phase1 = sub.add_parser("phase1")
    phase1.add_argument("--config", type=str, default=None)
    phase1.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    cfg = Phase1Config.from_json(args.config) if args.config else Phase1Config()

    if args.command == "phase1":
        return run_phase1(cfg, resume=args.resume)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
