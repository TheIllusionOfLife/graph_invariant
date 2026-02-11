from __future__ import annotations

import logging
from importlib import import_module
from types import ModuleType
from typing import Any

import networkx as nx
import numpy as np

from ..scoring import compute_metrics
from .features import features_from_graphs

LOGGER = logging.getLogger(__name__)


def _load_pysr_module() -> ModuleType | None:
    try:
        return import_module("pysr")
    except ImportError:
        return None


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    metrics = compute_metrics(y_true.tolist(), y_pred.tolist())
    return {
        "spearman": metrics.rho_spearman,
        "pearson": metrics.r_pearson,
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "valid_count": metrics.valid_count,
    }


def _make_regressor(
    pysr_module: ModuleType,
    niterations: int,
    populations: int,
    timeout_in_seconds: float | None,
):
    regressor_cls = getattr(pysr_module, "PySRRegressor", None)
    if regressor_cls is None:
        regressor_cls = getattr(pysr_module, "SymbolicRegressor", None)
    if regressor_cls is None:
        return None
    kwargs: dict[str, Any] = {
        "niterations": niterations,
        "populations": populations,
        "procs": 0,
        "random_state": 42,
    }
    if timeout_in_seconds is not None:
        kwargs["timeout_in_seconds"] = timeout_in_seconds
    return regressor_cls(**kwargs)


def run_pysr_baseline(
    train_graphs: list[nx.Graph],
    val_graphs: list[nx.Graph],
    test_graphs: list[nx.Graph],
    y_train: list[float],
    y_val: list[float],
    y_test: list[float],
    niterations: int = 30,
    populations: int = 8,
    timeout_in_seconds: float | None = 60.0,
) -> dict[str, object]:
    pysr_module = _load_pysr_module()
    if pysr_module is None:
        return {"status": "skipped", "reason": "pysr not installed"}

    x_train = features_from_graphs(train_graphs)
    x_val = features_from_graphs(val_graphs)
    x_test = features_from_graphs(test_graphs)
    y_train_np = np.asarray(y_train, dtype=float)
    y_val_np = np.asarray(y_val, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)

    if x_train.shape[0] == 0 or y_train_np.size == 0:
        return {"status": "skipped", "reason": "empty training data"}
    arrays = (x_train, y_train_np, x_val, y_val_np, x_test, y_test_np)
    if any(np.isnan(arr).any() for arr in arrays):
        return {"status": "skipped", "reason": "nan in features/targets"}

    model = _make_regressor(
        pysr_module=pysr_module,
        niterations=niterations,
        populations=populations,
        timeout_in_seconds=timeout_in_seconds,
    )
    if model is None:
        return {"status": "skipped", "reason": "pysr regressor class unavailable"}

    try:
        model.fit(x_train, y_train_np)
        return {
            "status": "ok",
            "model": "pysr",
            "val_metrics": _metrics_dict(y_val_np, np.asarray(model.predict(x_val), dtype=float)),
            "test_metrics": _metrics_dict(
                y_test_np, np.asarray(model.predict(x_test), dtype=float)
            ),
        }
    except Exception as exc:
        LOGGER.exception("pysr baseline execution failed")
        return {
            "status": "error",
            "reason": f"pysr execution failed: {type(exc).__name__}",
            "error_type": type(exc).__name__,
        }
