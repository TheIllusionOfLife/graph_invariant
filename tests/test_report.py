import json
from pathlib import Path

import pytest

from graph_invariant import cli


def test_main_report_command_writes_markdown(monkeypatch, tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "phase1_summary.json").write_text(
        json.dumps({"success": True, "best_candidate_id": "c1"}),
        encoding="utf-8",
    )
    (artifacts_dir / "baselines_summary.json").write_text(
        json.dumps({"stat_baselines": {"linear_regression": {"status": "ok"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["graph_invariant", "report", "--artifacts", str(artifacts_dir)],
    )

    assert cli.main() == 0
    assert "Report written to" in capsys.readouterr().out
    assert (artifacts_dir / "report.md").exists()


def test_main_report_command_handles_schema_v2_summary(monkeypatch, tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "phase1_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "success": True,
                "best_candidate_id": "c1",
                "stop_reason": "early_stop",
                "val_metrics": {"spearman": 0.9},
                "test_metrics": {"spearman": 0.85},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["graph_invariant", "report", "--artifacts", str(artifacts_dir)],
    )

    assert cli.main() == 0
    report = (artifacts_dir / "report.md").read_text(encoding="utf-8")
    assert "Validation Spearman: 0.9" in report
    assert "Test Spearman: 0.85" in report


def test_main_report_command_tolerates_malformed_json(monkeypatch, tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "phase1_summary.json").write_text("{bad json", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["graph_invariant", "report", "--artifacts", str(artifacts_dir)],
    )

    assert cli.main() == 0
    report = (artifacts_dir / "report.md").read_text(encoding="utf-8")
    assert "Phase 1 Report" in report


def test_report_includes_ood_validation(monkeypatch, tmp_path):
    from graph_invariant.cli import write_report

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "phase1_summary.json").write_text(
        json.dumps({"success": True, "best_candidate_id": "c1"}),
        encoding="utf-8",
    )
    (artifacts_dir / "ood_validation.json").write_text(
        json.dumps(
            {
                "large_random": {"spearman": 0.85, "valid_count": 90, "total_count": 100},
                "extreme_params": {"spearman": 0.72, "valid_count": 45, "total_count": 50},
                "special_topology": {"spearman": 0.91, "valid_count": 8, "total_count": 8},
            }
        ),
        encoding="utf-8",
    )
    report_path = write_report(str(artifacts_dir))
    report = report_path.read_text(encoding="utf-8")
    assert "## OOD Validation" in report
    assert "large_random" in report
    assert "spearman=0.8500" in report


def test_report_includes_map_elites_coverage(monkeypatch, tmp_path):
    from graph_invariant.cli import write_report

    artifacts_dir = tmp_path / "artifacts"
    logs_dir = artifacts_dir / "logs"
    logs_dir.mkdir(parents=True)
    (artifacts_dir / "phase1_summary.json").write_text(
        json.dumps({"success": True, "best_candidate_id": "c1"}),
        encoding="utf-8",
    )
    events = [
        json.dumps(
            {
                "event_type": "generation_summary",
                "payload": {"map_elites_stats_primary": {"coverage": 3}},
            }
        ),
        json.dumps(
            {
                "event_type": "generation_summary",
                "payload": {"map_elites_stats_primary": {"coverage": 7}},
            }
        ),
    ]
    (logs_dir / "events.jsonl").write_text("\n".join(events) + "\n", encoding="utf-8")
    report_path = write_report(str(artifacts_dir))
    report = report_path.read_text(encoding="utf-8")
    assert "## MAP-Elites Archive" in report
    assert "Final coverage: 7 cells" in report
    assert "3 -> 7" in report
