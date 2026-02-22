from __future__ import annotations

import networkx as nx
import numpy as np

from ..scoring import compute_metrics
from .features import FEATURE_ORDER, features_from_dict, features_from_graphs

# Features that can be targets — must be excluded from baseline inputs when used as targets.
_LEAKABLE_FEATURES = frozenset(FEATURE_ORDER)

# Backwards-compatible alias retained for existing tests and imports.
_FEATURE_ORDER = FEATURE_ORDER


def _features_from_graphs(
    graphs: list[nx.Graph],
    exclude_features: tuple[str, ...] | None = None,
    enable_spectral_feature_pack: bool = True,
) -> np.ndarray:
    return features_from_graphs(
        graphs,
        exclude_features=exclude_features,
        enable_spectral_feature_pack=enable_spectral_feature_pack,
    )


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
    if x_train.shape[0] == 0 or y_train.size == 0:
        return {"status": "skipped", "reason": "empty training data"}

    # Fit intercept without building augmented matrices on every predict call.
    x_mean = np.mean(x_train, axis=0)
    y_mean = float(np.mean(y_train))
    x_centered = x_train - x_mean
    y_centered = y_train - y_mean
    coef, *_ = np.linalg.lstsq(x_centered, y_centered, rcond=None)
    intercept = y_mean - float(np.dot(x_mean, coef))

    def predict(x: np.ndarray) -> np.ndarray:
        return (x @ coef) + intercept

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
    if x_train.shape[0] == 0 or y_train.size == 0:
        return {"status": "skipped", "reason": "empty training data"}

    arrays = (x_train, y_train, x_val, y_val, x_test, y_test)
    if any(np.isnan(arr).any() for arr in arrays):
        return {"status": "skipped", "reason": "nan in features/targets"}

    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
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
    target_name: str | None = None,
    enable_spectral_feature_pack: bool = True,
    known_invariants_train: dict[str, list[float]] | None = None,
    known_invariants_val: dict[str, list[float]] | None = None,
    known_invariants_test: dict[str, list[float]] | None = None,
) -> dict[str, object]:
    exclude = (target_name,) if target_name and target_name in _LEAKABLE_FEATURES else None
    if known_invariants_train is not None:
        x_train = features_from_dict(
            known_invariants_train,
            len(train_graphs),
            exclude,
            enable_spectral_feature_pack=enable_spectral_feature_pack,
        )
    else:
        x_train = _features_from_graphs(
            train_graphs,
            exclude,
            enable_spectral_feature_pack=enable_spectral_feature_pack,
        )

    if known_invariants_val is not None:
        x_val = features_from_dict(
            known_invariants_val,
            len(val_graphs),
            exclude,
            enable_spectral_feature_pack=enable_spectral_feature_pack,
        )
    else:
        x_val = _features_from_graphs(
            val_graphs,
            exclude,
            enable_spectral_feature_pack=enable_spectral_feature_pack,
        )

    if known_invariants_test is not None:
        x_test = features_from_dict(
            known_invariants_test,
            len(test_graphs),
            exclude,
            enable_spectral_feature_pack=enable_spectral_feature_pack,
        )
    else:
        x_test = _features_from_graphs(
            test_graphs,
            exclude,
            enable_spectral_feature_pack=enable_spectral_feature_pack,
        )
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
