import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def matrix_module():
    module_path = Path(__file__).resolve().parent.parent / "scripts" / "run_neurips_matrix.py"
    spec = importlib.util.spec_from_file_location("run_neurips_matrix", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_neurips_matrix module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_one_enables_map_elites_for_dual(monkeypatch, tmp_path, matrix_module):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps(
            {
                "target_name": "average_shortest_path_length",
                "num_train_graphs": 2,
                "num_val_graphs": 2,
                "num_test_graphs": 2,
                "max_generations": 0,
                "population_size": 1,
                "artifacts_dir": "ignored",
                "run_baselines": False,
                "persist_prompt_and_response_logs": False,
                "enable_dual_map_elites": True,
                "enable_map_elites": False,
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def _fake_run_phase1(cfg):
        captured["enable_map_elites"] = cfg.enable_map_elites
        run_root = Path(cfg.artifacts_dir)
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "phase1_summary.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "fitness_mode": "correlation",
                    "val_metrics": {"spearman": 0.9},
                    "test_metrics": {"spearman": 0.8},
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(matrix_module, "run_phase1", _fake_run_phase1)

    run = matrix_module._run_one(config_path=config_path, seed=11, output_root=tmp_path / "out")
    assert captured["enable_map_elites"] is True
    assert run["status"] == 0
    assert run["val_spearman"] == 0.9
    assert run["test_spearman"] == 0.8


def test_main_writes_matrix_summary(monkeypatch, tmp_path, matrix_module):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps(
            {
                "target_name": "average_shortest_path_length",
                "num_train_graphs": 2,
                "num_val_graphs": 2,
                "num_test_graphs": 2,
                "max_generations": 0,
                "population_size": 1,
                "artifacts_dir": "ignored",
                "run_baselines": False,
                "persist_prompt_and_response_logs": False,
            }
        ),
        encoding="utf-8",
    )

    def _fake_run_phase1(cfg):
        run_root = Path(cfg.artifacts_dir)
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "phase1_summary.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "fitness_mode": "correlation",
                    "val_metrics": {"spearman": 0.5 + 0.01 * cfg.seed},
                    "test_metrics": {"spearman": 0.4 + 0.01 * cfg.seed},
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(matrix_module, "run_phase1", _fake_run_phase1)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_neurips_matrix.py",
            "--configs",
            str(config_path),
            "--seeds",
            "1",
            "2",
            "--output-root",
            str(tmp_path / "matrix"),
        ],
    )

    matrix_module.main()

    summary_path = tmp_path / "matrix" / "matrix_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["total_runs"] == 2
    assert "cfg" in payload["experiments"]
    assert payload["experiments"]["cfg"]["summary"]["successful_runs"] == 2
