"""Classical baselines on engineered features (roadmap v0.2, spec §9.1).

Baselines are first-class here: the transformer is only interesting if it beats *strong* simple
models on the events that need temporal/cross-channel reasoning, and loses gracefully elsewhere.
Each classifier is an sklearn ``Pipeline`` with an inline ``StandardScaler`` so feature scaling is
fit on the training fold only (no leakage), regardless of how the data was generated.

The IsolationForest is deliberately *not* a 10-way classifier — it is framed as normal-vs-rest
anomaly detection to make the point that supervised event classification and anomaly detection are
related but distinct tasks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def make_logreg() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            # multinomial is the default in modern sklearn; the old multi_class arg was removed.
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def make_random_forest() -> Pipeline:
    # Trees are scale-invariant, but the StandardScaler keeps the pipeline interface uniform.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                # n_jobs=1: parallel trees + the torch/xgboost OpenMP runtimes deadlock when
                # both libraries run in one process on macOS (see tests/conftest.py).
                RandomForestClassifier(
                    n_estimators=300, class_weight="balanced", n_jobs=1, random_state=0
                ),
            ),
        ]
    )


def make_xgboost() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method="hist",
                    n_jobs=1,
                    random_state=0,
                ),
            ),
        ]
    )


# Name -> factory. The transformer (v0.3) must clear the best of these.
CLASSICAL_REGISTRY: dict[str, Callable[[], Pipeline]] = {
    "logreg": make_logreg,
    "random_forest": make_random_forest,
    "xgboost": make_xgboost,
}


def make_isolation_forest() -> IsolationForest:
    """Unsupervised anomaly detector (normal-vs-rest framing; see module docstring)."""
    return IsolationForest(n_estimators=200, contamination="auto", random_state=0, n_jobs=1)


def feature_importance(
    model: Pipeline, feature_names: list[str], top_k: int = 15
) -> list[tuple[str, float]]:
    """Return the top-``k`` ``(name, importance)`` pairs for tree models (RF/XGB).

    Falls back to absolute logistic-regression coefficient magnitude (averaged over classes).
    Returns an empty list if no importance signal is available.
    """
    clf: Any = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        importances = np.abs(np.asarray(clf.coef_, dtype=float)).mean(axis=0)
    else:
        return []
    order = np.argsort(importances)[::-1][:top_k]
    return [(feature_names[i], float(importances[i])) for i in order]
