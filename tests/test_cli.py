import json

import pytest

from graph_invariant import cli


def test_main_benchmark_command_invokes_runner(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    def fake_run_benchmark(cfg):  # noqa: ANN001
        calls["artifacts_dir"] = cfg.artifacts_dir
        return 0

    config_path = tmp_path / "benchmark.json"
    config_path.write_text(json.dumps({"artifacts_dir": "artifacts_bench"}), encoding="utf-8")
    monkeypatch.setattr("graph_invariant.benchmark.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        "sys.argv",
        ["graph_invariant", "benchmark", "--config", str(config_path)],
    )

    assert cli.main() == 0
    assert calls["artifacts_dir"] == "artifacts_bench"
