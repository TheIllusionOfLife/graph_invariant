import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(scope="module")
def figures_module():
    module_path = Path(__file__).resolve().parent.parent / "analysis" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location("generate_figures", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load generate_figures module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_baseline_comparison_collects_from_later_experiment(
    figures_module, tmp_path, monkeypatch
):
    captured = {}
    original_subplots = plt.subplots

    def _wrapped_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(figures_module.plt, "subplots", _wrapped_subplots)

    data = {
        "exp_without_baselines": {
            "val_spearman": 0.5,
            "test_spearman": 0.4,
            "baselines": {},
            "ood": {},
            "convergence": {},
        },
        "exp_with_baselines": {
            "val_spearman": 0.6,
            "test_spearman": 0.55,
            "baselines": {
                "stat_baselines": {
                    "linear_regression": {
                        "status": "ok",
                        "val_metrics": {"spearman": 0.7},
                        "test_metrics": {"spearman": 0.68},
                    }
                },
                "pysr_baseline": {
                    "status": "ok",
                    "val_metrics": {"spearman": 0.8},
                    "test_metrics": {"spearman": 0.79},
                },
            },
            "ood": {},
            "convergence": {},
        },
    }
    out = tmp_path / "baseline.pdf"
    figures_module.plot_baseline_comparison(data, out)
    assert out.exists()

    xtick_labels = [tick.get_text() for tick in captured["ax"].get_xticklabels()]
    assert "Linear Regression" in xtick_labels
    assert "PySR" in xtick_labels


def test_main_handles_malformed_figure_data(figures_module, tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "analysis_results"
    data_dir.mkdir()
    (data_dir / "figure_data.json").write_text("{not-json}", encoding="utf-8")
    out_dir = tmp_path / "figures"

    monkeypatch.setattr(
        "sys.argv",
        ["generate_figures.py", "--data", str(data_dir), "--output", str(out_dir)],
    )

    figures_module.main()
    output = capsys.readouterr().out
    assert "failed to read" in output
    assert not any(out_dir.glob("*.pdf"))


def test_plot_benchmark_boxplot_uses_runs_data(figures_module, tmp_path):
    data = {
        "benchmark/benchmark_20260215T230550Z": {
            "runs": [
                {"seed": 1, "val_spearman": 0.9, "test_spearman": 0.91},
                {"seed": 2, "val_spearman": 0.88, "test_spearman": 0.89},
            ]
        }
    }
    out = tmp_path / "benchmark_boxplot.pdf"
    figures_module.plot_benchmark_boxplot(data, out)
    assert out.exists()
