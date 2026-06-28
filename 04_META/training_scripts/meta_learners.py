"""
Unified window-level meta learners (4 tree probs -> raw meta score).

Each learner exposes fit(train_frame) and predict_proba(X)[:, 1] for sklearn compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier

from range_trust_meta import RangeTrustMeta, TREE_COLUMNS as RT_TREE_COLUMNS

DEFAULT_FEATURE_COLUMNS = ("xgb", "lightgbm", "random_forest", "extra_trees")


def _frame_to_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return frame[list(feature_columns)].to_numpy(dtype=np.float32)


class IdentityCalibrator:
    """Pass-through calibrator for method='none'."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=np.float32).reshape(-1), 0.0, 1.0)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = self.predict(x)
        return np.column_stack([1.0 - p, p])


@dataclass
class MetaLearnerSpec:
    name: str
    description: str
    builder: Callable[[], Any]


class SklearnMetaWrapper:
    def __init__(self, model: Any, *, kind: str, feature_columns: list[str]) -> None:
        self.model = model
        self.kind = kind
        self.feature_columns = feature_columns

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)

    def fit(self, frame: pd.DataFrame, label_column: str = "label") -> SklearnMetaWrapper:
        x = _frame_to_matrix(frame, self.feature_columns)
        y = frame[label_column].astype(int).to_numpy()
        self.model.fit(x, y)
        return self


class RangeTrustWrapper:
    def __init__(
        self,
        *,
        pooling_mode: str = "weighted",
        trust_mode: str = "rate",
        disagreement: bool = False,
        n_bins: int = 20,
        feature_columns: list[str] | None = None,
    ) -> None:
        self.pooling_mode = pooling_mode
        self.trust_mode = trust_mode
        self.disagreement = disagreement
        self.n_bins = n_bins
        self.feature_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
        self._meta = RangeTrustMeta(n_bins=n_bins, pooling_mode=pooling_mode)  # type: ignore[arg-type]
        self._lift_base = 0.5

    def fit(self, frame: pd.DataFrame, label_column: str = "label") -> RangeTrustWrapper:
        self._meta.fit(frame, label_column=label_column)
        self._lift_base = max(float(frame[label_column].mean()), 1e-6)
        return self

    def _trust_values(self, tree: str, prob: float) -> float:
        trust = self._meta.trust_for(tree, prob)
        if self.trust_mode == "lift":
            return float(trust / self._lift_base)
        return float(trust)

    def _row_score(self, tree_probs: dict[str, float]) -> float:
        entries = []
        for tree in RT_TREE_COLUMNS:
            risk = float(tree_probs[tree])
            trust = self._trust_values(tree, risk)
            entries.append({"tree": tree, "risk": risk, "trust": trust})

        if self.disagreement:
            for idx, e in enumerate(entries):
                others = [o["risk"] for j, o in enumerate(entries) if j != idx]
                mean_dist = float(np.mean(np.abs(e["risk"] - np.asarray(others)))) if others else 1.0
                e["weight"] = e["trust"] / (1e-6 + mean_dist)
        else:
            for e in entries:
                e["weight"] = e["trust"]

        if self.pooling_mode == "rank_decay":
            ranked = sorted(entries, key=lambda r: r["weight"], reverse=True)
            divisors = (1.0, 10.0, 100.0, 1000.0)
            return float(sum(r["risk"] / divisors[i] for i, r in enumerate(ranked)))

        weights = np.array([e["weight"] for e in entries], dtype=np.float64)
        risks = np.array([e["risk"] for e in entries], dtype=np.float64)
        wsum = float(weights.sum())
        if wsum <= 0:
            return float(risks.mean())
        return float(np.dot(weights / wsum, risks))

    def _trust_array(self, tree: str, probs: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(self._meta.bin_edges, np.clip(probs, 0, 1), side="right") - 1, 0, self._meta.n_bins - 1)
        trust = self._meta.trust_profiles[tree][idx]
        if self.trust_mode == "lift":
            return (trust / self._lift_base).astype(np.float32)
        return trust.astype(np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        col_map = {name: i for i, name in enumerate(self.feature_columns)}
        risks = np.column_stack([x[:, col_map[t]] for t in RT_TREE_COLUMNS]).astype(np.float64)
        trusts = np.column_stack([self._trust_array(t, risks[:, i]) for i, t in enumerate(RT_TREE_COLUMNS)])

        if self.disagreement:
            scores = np.zeros(x.shape[0], dtype=np.float32)
            for i in range(x.shape[0]):
                tree_probs = {tree: float(risks[i, j]) for j, tree in enumerate(RT_TREE_COLUMNS)}
                scores[i] = self._row_score(tree_probs)
            p = np.clip(scores, 0.0, 1.0)
            return np.column_stack([1.0 - p, p])

        if self.pooling_mode == "rank_decay":
            scores = np.zeros(x.shape[0], dtype=np.float64)
            divisors = np.array([1.0, 10.0, 100.0, 1000.0])
            for i in range(x.shape[0]):
                order = np.argsort(-trusts[i])
                scores[i] = sum(risks[i, order[j]] / divisors[j] for j in range(4))
            p = np.clip(scores, 0.0, 1.0).astype(np.float32)
            return np.column_stack([1.0 - p, p])

        weights = trusts
        wsum = weights.sum(axis=1, keepdims=True)
        wsum = np.where(wsum <= 0, 1.0, wsum)
        pooled = (weights * risks).sum(axis=1) / wsum[:, 0]
        p = np.clip(pooled, 0.0, 1.0).astype(np.float32)
        return np.column_stack([1.0 - p, p])


class ClosestPairMeta:
    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self.feature_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
        self._fitted = True

    def fit(self, frame: pd.DataFrame, label_column: str = "label") -> ClosestPairMeta:
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = np.zeros(x.shape[0], dtype=np.float32)
        for i in range(x.shape[0]):
            vals = x[i]
            best_dist = float("inf")
            best_pair = (0, 1)
            for a in range(len(vals)):
                for b in range(a + 1, len(vals)):
                    dist = abs(float(vals[a]) - float(vals[b]))
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (a, b)
            scores[i] = float((vals[best_pair[0]] + vals[best_pair[1]]) / 2.0)
        p = np.clip(scores, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


class ReductionMeta:
    def __init__(self, op: str, feature_columns: list[str] | None = None) -> None:
        self.op = op
        self.feature_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
        self._fitted = True

    def fit(self, frame: pd.DataFrame, label_column: str = "label") -> ReductionMeta:
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.op == "mean":
            p = x.mean(axis=1)
        elif self.op == "median":
            p = np.median(x, axis=1)
        elif self.op == "max":
            p = x.max(axis=1)
        else:
            raise ValueError(self.op)
        p = np.clip(p.astype(np.float32), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def build_logistic(c: float, *, solver: str = "liblinear") -> LogisticRegression:
    return LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver=solver,
    )


def build_polynomial_logistic(c: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_meta_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        scale_pos_weight=2.0,
        tree_method="hist",
        device="cpu",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def meta_learner_catalog(feature_columns: list[str]) -> list[tuple[str, str, Any]]:
    """Return (name, description, unfitted_model)."""
    return [
        (
            "logistic_c10",
            "Logistic C=10 balanced (legacy baseline)",
            SklearnMetaWrapper(build_logistic(10.0, solver="lbfgs"), kind="logistic", feature_columns=feature_columns),
        ),
        (
            "logistic_c005",
            "Logistic C=0.05 liblinear (current production)",
            SklearnMetaWrapper(build_logistic(0.05), kind="logistic", feature_columns=feature_columns),
        ),
        (
            "polynomial_logistic",
            "Degree-2 interaction logistic C=1",
            SklearnMetaWrapper(build_polynomial_logistic(1.0), kind="polynomial", feature_columns=feature_columns),
        ),
        (
            "meta_xgboost",
            "Shallow XGBoost on 4 tree probs",
            SklearnMetaWrapper(build_meta_xgb(), kind="xgboost", feature_columns=feature_columns),
        ),
        ("tree_mean", "Unweighted mean of 4 trees", ReductionMeta("mean", feature_columns)),
        ("tree_median", "Median of 4 trees", ReductionMeta("median", feature_columns)),
        ("tree_max", "Max of 4 trees", ReductionMeta("max", feature_columns)),
        ("closest_pair", "Average of closest tree pair by score distance", ClosestPairMeta(feature_columns)),
        (
            "range_trust_weighted",
            "Bin trust weighted pool",
            RangeTrustWrapper(pooling_mode="weighted", trust_mode="rate", feature_columns=feature_columns),
        ),
        (
            "range_trust_lift",
            "Bin lift-weighted pool",
            RangeTrustWrapper(pooling_mode="weighted", trust_mode="lift", feature_columns=feature_columns),
        ),
        (
            "range_trust_disagreement",
            "Trust / disagreement weighted pool",
            RangeTrustWrapper(
                pooling_mode="weighted",
                trust_mode="rate",
                disagreement=True,
                feature_columns=feature_columns,
            ),
        ),
        (
            "range_trust_rank_decay",
            "Bin trust rank-decay pool",
            RangeTrustWrapper(pooling_mode="rank_decay", trust_mode="rate", feature_columns=feature_columns),
        ),
    ]


def _unwrap_for_save(model: Any) -> tuple[Any, list[str] | None]:
    """Persist inner sklearn/xgb estimators so web runtime does not need meta_learners."""
    feature_columns = getattr(model, "feature_columns", None)
    if isinstance(model, SklearnMetaWrapper):
        return model.model, list(model.feature_columns)
    return model, list(feature_columns) if feature_columns else None


def save_meta_model(model: Any, path: Path, *, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    inner, feature_columns = _unwrap_for_save(model)
    payload: dict[str, Any] = {"meta_name": name, "model": inner}
    if feature_columns:
        payload["feature_columns"] = feature_columns
    joblib.dump(payload, path)


def load_meta_model(path: Path) -> tuple[str, Any]:
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return str(payload.get("meta_name", "unknown")), payload["model"]
    return "legacy", payload
