"""Shared helpers for window-stack risk aggregation (max-pool + uplift)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SCRIPTS_ROOT = Path(__file__).resolve().parent
AGGREGATOR_ROOT = SCRIPTS_ROOT.parent
PROJECT_ROOT = AGGREGATOR_ROOT.parent
META_SCRIPTS_ROOT = PROJECT_ROOT / "04_META" / "training_scripts"
TREE_SCRIPTS_ROOT = PROJECT_ROOT / "03_TREE" / "training_scripts"
DEFAULT_CONFIG_PATH = AGGREGATOR_ROOT / "aggregator_config.yaml"

_SCORE_DIR = PROJECT_ROOT / "05_SCORE"
if str(_SCORE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCORE_DIR))
from deployment_tiers import (  # noqa: E402
    DEFAULT_REVIEW_THRESHOLD,
    classify_function_deployment_tier,
)

AGREEMENT_STATUSES = (
    "agree_positive",
    "agree_negative",
    "review_suggested",
    "diffuse_risk",
)

DEPLOYMENT_TIERS = (
    "vuln",
    "needs_review",
    "safe",
    "confirmed",
    "investigate",
    "soft_review",
)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def import_meta_common():
    if str(META_SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(META_SCRIPTS_ROOT))
    import meta_common  # noqa: PLC0415

    return meta_common


def import_train_trees():
    spec = importlib.util.spec_from_file_location("train_trees", TREE_SCRIPTS_ROOT / "train_trees.py")
    train_trees = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(train_trees)
    return train_trees


def resolve_window_embedding_path(embeddings_root: str, split: str, pool: str) -> Path:
    return resolve_path(f"{embeddings_root}/{pool}/{split}/{split}_window_embeddings.parquet")


def apply_score_calibrator(calibrator_bundle: dict[str, Any], raw_scores: np.ndarray) -> np.ndarray:
    method = str(calibrator_bundle.get("method", "isotonic")).lower()
    calibrator = calibrator_bundle["calibrator"]
    if method == "none":
        if hasattr(calibrator, "predict"):
            return np.clip(calibrator.predict(raw_scores), 0.0, 1.0).astype(np.float32)
        return np.clip(np.asarray(raw_scores, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if method == "isotonic":
        return np.clip(calibrator.predict(raw_scores), 0.0, 1.0).astype(np.float32)
    return calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1].astype(np.float32)


def resolve_meta_model(meta_payload: Any) -> Any:
    if isinstance(meta_payload, dict) and "model" in meta_payload:
        return meta_payload["model"]
    return meta_payload


def predict_meta_raw(meta_model: Any, base_matrix: np.ndarray) -> np.ndarray:
    model = resolve_meta_model(meta_model)
    return model.predict_proba(base_matrix)[:, 1].astype(np.float32)


def function_meta_from_window_frame(
    window_frame: pd.DataFrame,
    *,
    function_group_column: str = "function_group_id",
    label_column: str = "label",
) -> pd.DataFrame:
    """One row per function with its label, derived from window rows."""
    return (
        window_frame.groupby(function_group_column, sort=False)[label_column]
        .max()
        .reset_index()
    )


def prob_max_contributing_indices(
    window_probs: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> list[int]:
    """Indices of windows tied at the peak window probability (risk max-pool)."""
    if window_probs.size == 0:
        return []
    peak = float(np.max(window_probs))
    return [int(i) for i, prob in enumerate(window_probs) if abs(float(prob) - peak) <= tolerance]


def classify_agreement(
    *,
    function_flagged: bool,
    window_probs: np.ndarray,
    window_threshold: float,
) -> tuple[str, list[int]]:
    flagged_indices = [int(i) for i, prob in enumerate(window_probs) if float(prob) >= window_threshold]
    any_window_flagged = bool(flagged_indices)

    if function_flagged and any_window_flagged:
        return "agree_positive", flagged_indices
    if not function_flagged and not any_window_flagged:
        return "agree_negative", flagged_indices
    if not function_flagged and any_window_flagged:
        return "review_suggested", flagged_indices
    return "diffuse_risk", flagged_indices


def classify_precision_tier(
    *,
    function_flagged: bool,
    window_probs: np.ndarray,
    window_threshold_triage: float,
    window_threshold_confirmed: float,
) -> tuple[str, list[int], list[int], bool, bool]:
    """Precision-first deployment tiers (dual window thresholds)."""
    confirmed_indices = [
        int(i) for i, prob in enumerate(window_probs) if float(prob) >= window_threshold_confirmed
    ]
    triage_indices = [
        int(i) for i, prob in enumerate(window_probs) if float(prob) >= window_threshold_triage
    ]
    has_confirmed = bool(confirmed_indices)

    if function_flagged and has_confirmed:
        tier = "confirmed"
    elif has_confirmed and not function_flagged:
        tier = "investigate"
    elif function_flagged and not has_confirmed:
        tier = "soft_review"
    else:
        tier = "safe"

    user_facing_vuln = tier in ("confirmed", "investigate")
    whole_function_vuln = tier == "confirmed"
    return tier, confirmed_indices, triage_indices, user_facing_vuln, whole_function_vuln


def build_function_records(
    *,
    function_frame: pd.DataFrame,
    function_group_column: str,
    label_column: str,
    calibrated_scores: np.ndarray,
    function_threshold: float,
    window_frame: pd.DataFrame,
    window_threshold: float,
    tolerance: float,
    precision_cfg: dict[str, Any] | None = None,
    scoring_mode: str = "window_stack_aggregate",
    function_review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    tiered_deployment: bool = True,
) -> list[dict[str, Any]]:
    window_by_function = {
        group_id: group.sort_values("window_index").reset_index(drop=True)
        for group_id, group in window_frame.groupby(function_group_column, sort=False)
    }

    records: list[dict[str, Any]] = []
    for row_index, row in function_frame.iterrows():
        group_id = str(row[function_group_column])
        label = int(row[label_column])
        function_score = float(calibrated_scores[row_index])
        tier = classify_function_deployment_tier(
            function_score,
            vuln_threshold=function_threshold,
            review_threshold=function_review_threshold,
        )
        function_flagged = bool(tier["function_flagged"]) if tiered_deployment else function_score >= function_threshold

        group_windows = window_by_function.get(group_id)
        if group_windows is None or group_windows.empty:
            if tiered_deployment:
                status = str(tier["agreement_status"])
                _, flagged_indices = classify_agreement(
                    function_flagged=function_flagged,
                    window_probs=np.array([], dtype=np.float32),
                    window_threshold=window_threshold,
                )
            else:
                status, flagged_indices = classify_agreement(
                    function_flagged=function_flagged,
                    window_probs=np.array([], dtype=np.float32),
                    window_threshold=window_threshold,
                )
            record = {
                "function_group_id": group_id,
                "label": label,
                "function_score_calibrated": function_score,
                "function_flagged": function_flagged,
                "agreement_status": status,
                "window_count": 0,
                "flagged_window_indices": flagged_indices,
                "flagged_window_ids": [],
                "flagged_windows": [],
                "contributing_window_indices": [],
                "contributing_window_ids": [],
                "contributing_windows": [],
                "max_window_prob": None,
            }
            if tiered_deployment:
                record.update(
                    {
                        "deployment_tier": tier["deployment_tier"],
                        "function_needs_review": tier["function_needs_review"],
                        "user_facing_vuln": tier["user_facing_vuln"],
                        "whole_function_vuln": tier["whole_function_vuln"],
                    }
                )
            elif precision_cfg:
                tier, confirmed_idx, triage_idx, user_facing, whole_fn = classify_precision_tier(
                    function_flagged=function_flagged,
                    window_probs=np.array([], dtype=np.float32),
                    window_threshold_triage=float(precision_cfg["window_threshold_triage"]),
                    window_threshold_confirmed=float(precision_cfg["window_threshold_confirmed"]),
                )
                record.update(
                    {
                        "deployment_tier": tier,
                        "user_facing_vuln": user_facing,
                        "whole_function_vuln": whole_fn,
                        "confirmed_window_indices": confirmed_idx,
                        "triage_window_indices": triage_idx,
                    }
                )
            records.append(record)
            continue

        window_probs = group_windows["window_prob"].to_numpy(dtype=np.float32)
        window_ids = group_windows["window_id"].astype(str).tolist()
        window_indices = group_windows["window_index"].astype(int).tolist()

        _, flagged_indices = classify_agreement(
            function_flagged=function_score >= function_threshold,
            window_probs=window_probs,
            window_threshold=window_threshold,
        )
        if tiered_deployment:
            status = str(tier["agreement_status"])
            function_flagged = bool(tier["function_flagged"])
        else:
            status, flagged_indices = classify_agreement(
                function_flagged=function_flagged,
                window_probs=window_probs,
                window_threshold=window_threshold,
            )
        contributing_indices = prob_max_contributing_indices(window_probs, tolerance=tolerance)
        flagged_ids = [window_ids[i] for i in flagged_indices]
        contributing_ids = [window_ids[i] for i in contributing_indices]

        flagged_windows = [
            {
                "window_index": window_indices[i],
                "window_id": window_ids[i],
                "window_prob": float(window_probs[i]),
            }
            for i in flagged_indices
        ]
        contributing_windows = [
            {
                "window_index": window_indices[i],
                "window_id": window_ids[i],
                "window_prob": float(window_probs[i]),
            }
            for i in contributing_indices
        ]

        record = {
            "function_group_id": group_id,
            "label": label,
            "function_score_calibrated": function_score,
            "function_flagged": function_flagged,
            "agreement_status": status,
            "window_count": int(len(group_windows)),
            "flagged_window_indices": [window_indices[i] for i in flagged_indices],
            "flagged_window_ids": flagged_ids,
            "flagged_windows": flagged_windows,
            "contributing_window_indices": [window_indices[i] for i in contributing_indices],
            "contributing_window_ids": contributing_ids,
            "contributing_windows": contributing_windows,
            "max_window_prob": float(window_probs.max()),
        }
        if tiered_deployment:
            record.update(
                {
                    "deployment_tier": tier["deployment_tier"],
                    "function_needs_review": tier["function_needs_review"],
                    "user_facing_vuln": tier["user_facing_vuln"],
                    "whole_function_vuln": tier["whole_function_vuln"],
                }
            )
        elif precision_cfg:
            tier, confirmed_idx, triage_idx, user_facing, whole_fn = classify_precision_tier(
                function_flagged=function_flagged,
                window_probs=window_probs,
                window_threshold_triage=float(precision_cfg["window_threshold_triage"]),
                window_threshold_confirmed=float(precision_cfg["window_threshold_confirmed"]),
            )
            confirmed_ids = [window_ids[i] for i in confirmed_idx]
            record.update(
                {
                    "deployment_tier": tier,
                    "user_facing_vuln": user_facing,
                    "whole_function_vuln": whole_fn,
                    "confirmed_window_indices": [window_indices[i] for i in confirmed_idx],
                    "confirmed_window_ids": confirmed_ids,
                    "confirmed_windows": [
                        {
                            "window_index": window_indices[i],
                            "window_id": window_ids[i],
                            "window_prob": float(window_probs[i]),
                        }
                        for i in confirmed_idx
                    ],
                    "triage_window_indices": [window_indices[i] for i in triage_idx],
                }
            )
        records.append(record)
    return records


def summarize_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    counts: dict[str, int] = {status: 0 for status in AGREEMENT_STATUSES}
    label_counts: dict[str, dict[str, int]] = {
        status: {"label_0": 0, "label_1": 0} for status in AGREEMENT_STATUSES
    }
    for record in records:
        status = str(record["agreement_status"])
        counts[status] += 1
        label_key = "label_1" if int(record["label"]) == 1 else "label_0"
        label_counts[status][label_key] += 1

    percentages = {status: (count / total if total else 0.0) for status, count in counts.items()}
    return {
        "total_functions": total,
        "counts": counts,
        "percentages": percentages,
        "by_label": label_counts,
    }


def tier_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support_pos": int((y_true == 1).sum()),
        "support_neg": int((y_true == 0).sum()),
        "flagged": int(y_pred.sum()),
    }


def summarize_precision_tiers(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    counts: dict[str, int] = {tier: 0 for tier in DEPLOYMENT_TIERS}
    label_counts: dict[str, dict[str, int]] = {
        tier: {"label_0": 0, "label_1": 0} for tier in DEPLOYMENT_TIERS
    }
    for record in records:
        tier = str(record.get("deployment_tier", "safe"))
        if tier not in counts:
            tier = "safe"
        counts[tier] += 1
        label_key = "label_1" if int(record["label"]) == 1 else "label_0"
        label_counts[tier][label_key] += 1

    y_true = np.array([int(r["label"]) for r in records], dtype=int)
    metrics: dict[str, dict[str, float]] = {}
    for key, getter in [
        ("confirmed", lambda r: bool(r.get("whole_function_vuln"))),
        ("user_facing", lambda r: bool(r.get("user_facing_vuln"))),
        ("legacy_agree_positive", lambda r: str(r.get("agreement_status")) == "agree_positive"),
    ]:
        y_pred = np.array([int(getter(r)) for r in records], dtype=int)
        metrics[key] = tier_binary_metrics(y_true, y_pred)

    return {
        "total_functions": total,
        "counts": counts,
        "percentages": {tier: (count / total if total else 0.0) for tier, count in counts.items()},
        "by_label": label_counts,
        "binary_metrics": metrics,
    }
