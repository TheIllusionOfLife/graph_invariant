from __future__ import annotations

import networkx as nx
import numpy as np

from ..known_invariants import compute_known_invariant_values
from ..scoring import compute_metrics

_FEATURE_ORDER = (
    "density",
    "clustering_coefficient",
    "degree_assortativity",
    "transitivity",
    "average_degree",
    "max_degree",
    "spectral_radius",
    "diameter",
    "algebraic_connectivity",
)


def _features_from_graphs(graphs: list[nx.Graph]) -> np.ndarray:
    values = compute_known_invariant_values(graphs)
    cols = [values[name] for name in _FEATURE_ORDER]
    return np.asarray(list(zip(*cols, strict=True)), dtype=float)


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    metrics = compute_metrics(y_true.tolist(), y_pred.tolist())
    return {
        "spearman": metrics.rho_spearman,
        "pearson": metrics.r_pearson,
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "valid_count": metrics.valid_count,
    }


def _run_linear_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, object]:
    x_aug = np.c_[np.ones(len(x_train)), x_train]
    coef, *_ = np.linalg.lstsq(x_aug, y_train, rcond=None)

    def predict(x: np.ndarray) -> np.ndarray:
        x_eval = np.c_[np.ones(len(x)), x]
        return x_eval @ coef

    return {
        "status": "ok",
        "val_metrics": _metrics_dict(y_val, predict(x_val)),
        "test_metrics": _metrics_dict(y_test, predict(x_test)),
    }


def _run_random_forest_optional(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, object]:
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        return {"status": "skipped", "reason": "scikit-learn not installed"}

    model = RandomForestRegressor(n_estimators=128, random_state=42)
    model.fit(x_train, y_train)
    return {
        "status": "ok",
        "val_metrics": _metrics_dict(y_val, model.predict(x_val)),
        "test_metrics": _metrics_dict(y_test, model.predict(x_test)),
    }


def run_stat_baselines(
    train_graphs: list[nx.Graph],
    val_graphs: list[nx.Graph],
    test_graphs: list[nx.Graph],
    y_train: list[float],
    y_val: list[float],
    y_test: list[float],
) -> dict[str, object]:
    x_train = _features_from_graphs(train_graphs)
    x_val = _features_from_graphs(val_graphs)
    x_test = _features_from_graphs(test_graphs)
    y_train_np = np.asarray(y_train, dtype=float)
    y_val_np = np.asarray(y_val, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)

    return {
        "linear_regression": _run_linear_regression(
            x_train,
            y_train_np,
            x_val,
            y_val_np,
            x_test,
            y_test_np,
        ),
        "random_forest": _run_random_forest_optional(
            x_train,
            y_train_np,
            x_val,
            y_val_np,
            x_test,
            y_test_np,
        ),
    }
