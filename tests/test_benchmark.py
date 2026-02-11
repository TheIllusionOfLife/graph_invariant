import json
from pathlib import Path

from graph_invariant.benchmark import run_benchmark
from graph_invariant.config import Phase1Config


def test_run_benchmark_aggregates_seed_runs(monkeypatch, tmp_path):
    calls: list[tuple[int, str]] = []

    def fake_run_phase1(cfg: Phase1Config, resume: str | None = None) -> int:
        del resume
        calls.append((cfg.seed, cfg.artifacts_dir))
        root = Path(cfg.artifacts_dir)
        root.mkdir(parents=True, exist_ok=True)
        success = cfg.seed % 2 == 0
        (root / "phase1_summary.json").write_text(
            json.dumps({"success": success, "val_metrics": {"spearman": 0.9 if success else 0.8}}),
            encoding="utf-8",
        )
        (root / "baselines_summary.json").write_text(
            json.dumps({"schema_version": 1, "pysr_baseline": {"status": "ok"}}),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("graph_invariant.benchmark.run_phase1", fake_run_phase1)
    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        benchmark_seeds=(11, 22, 33),
        run_baselines=True,
        max_generations=0,
    )
    status = run_benchmark(cfg)
    assert status == 0
    assert [seed for seed, _ in calls] == [11, 22, 33]

    benchmark_roots = sorted((tmp_path / "artifacts").glob("benchmark_*"))
    assert len(benchmark_roots) == 1
    payload = json.loads((benchmark_roots[0] / "benchmark_summary.json").read_text("utf-8"))
    assert payload["total_runs"] == 3
    assert payload["success_count"] == 1


def test_run_benchmark_returns_failure_when_any_seed_run_fails(monkeypatch, tmp_path):
    def fake_run_phase1(cfg: Phase1Config, resume: str | None = None) -> int:
        del cfg, resume
        return 1

    monkeypatch.setattr("graph_invariant.benchmark.run_phase1", fake_run_phase1)
    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        benchmark_seeds=(11, 22),
        run_baselines=False,
        max_generations=0,
    )
    status = run_benchmark(cfg)
    assert status == 1

    benchmark_roots = sorted((tmp_path / "artifacts").glob("benchmark_*"))
    payload = json.loads((benchmark_roots[0] / "benchmark_summary.json").read_text("utf-8"))
    assert payload["failed_runs"] == 2
