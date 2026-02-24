from __future__ import annotations

import logging
from importlib import import_module
from types import ModuleType
from typing import Any

import networkx as nx
import numpy as np

from ..scoring import compute_metrics
from .features import FEATURE_ORDER, features_from_graphs

LOGGER = logging.getLogger(__name__)

_LEAKABLE_FEATURES = frozenset(FEATURE_ORDER)


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


def _numeric_vector(values: Any, expected_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != expected_size:
        raise ValueError(f"prediction size mismatch: expected {expected_size}, got {arr.size}")
    return arr


def _make_regressor(
    pysr_module: ModuleType,
    niterations: int,
    populations: int,
    procs: int,
    timeout_in_seconds: float | None,
) -> Any | None:
    regressor_cls = getattr(pysr_module, "PySRRegressor", None)
    if regressor_cls is None:
        regressor_cls = getattr(pysr_module, "SymbolicRegressor", None)
    if regressor_cls is None:
        return None
    kwargs: dict[str, Any] = {
        "niterations": niterations,
        "populations": populations,
        "procs": procs,  # Sequential by default to avoid nested pool contention.
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
    procs: int = 0,
    timeout_in_seconds: float | None = 60.0,
    target_name: str | None = None,
    enable_spectral_feature_pack: bool = True,
) -> dict[str, object]:
    pysr_module = _load_pysr_module()
    if pysr_module is None:
        return {"status": "skipped", "reason": "pysr not installed"}

    exclude = (target_name,) if target_name and target_name in _LEAKABLE_FEATURES else None
    x_train = features_from_graphs(
        train_graphs,
        exclude_features=exclude,
        enable_spectral_feature_pack=enable_spectral_feature_pack,
    )
    x_val = features_from_graphs(
        val_graphs,
        exclude_features=exclude,
        enable_spectral_feature_pack=enable_spectral_feature_pack,
    )
    x_test = features_from_graphs(
        test_graphs,
        exclude_features=exclude,
        enable_spectral_feature_pack=enable_spectral_feature_pack,
    )
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
        procs=procs,
        timeout_in_seconds=timeout_in_seconds,
    )
    if model is None:
        return {"status": "skipped", "reason": "pysr regressor class unavailable"}

    try:
        model.fit(x_train, y_train_np)
        val_pred = _numeric_vector(model.predict(x_val), expected_size=y_val_np.size)
        test_pred = _numeric_vector(model.predict(x_test), expected_size=y_test_np.size)
        return {
            "status": "ok",
            "model": "pysr",
            "val_metrics": _metrics_dict(y_val_np, val_pred),
            "test_metrics": _metrics_dict(y_test_np, test_pred),
        }
    except (RuntimeError, ValueError, TypeError) as exc:
        LOGGER.exception("pysr baseline execution failed")
        return {
            "status": "error",
            "reason": f"pysr execution failed: {type(exc).__name__}",
            "error_type": type(exc).__name__,
        }
    except (ArithmeticError, OverflowError) as exc:
        LOGGER.exception("pysr baseline execution failed")
        return {
            "status": "error",
            "reason": f"pysr execution failed: {type(exc).__name__}",
            "error_type": type(exc).__name__,
        }
