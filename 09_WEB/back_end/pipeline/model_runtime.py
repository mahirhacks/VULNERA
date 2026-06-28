"""Load production models once and score in-memory function windows."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import yaml

BACKEND_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = BACKEND_ROOT.parent
PROJECT_ROOT = WEB_ROOT.parent
DATA_PROCESSING_ROOT = PROJECT_ROOT / "01_Data_Processing"
PIPELINE_ROOT = DATA_PROCESSING_ROOT / "dataset_pipeline"
AGGREGATOR_SCRIPTS = PROJECT_ROOT / "06_AGGREGATOR" / "training_scripts"
META_SCRIPTS = PROJECT_ROOT / "04_META" / "training_scripts"
XAI_SCRIPTS = PROJECT_ROOT / "07_XAI" / "training_scripts"


def _ensure_meta_scripts_path() -> None:
    if str(META_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(META_SCRIPTS))


def _import_module(module_name: str, filename: str):
    path = PIPELINE_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_dataset_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "01_Data_Processing" / "dataset_config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class ModelBundle:
    base_models: dict[str, Any]
    meta_model: Any
    calibrator_bundle: dict[str, Any]
    window_model: Any | None
    function_threshold: float
    function_review_threshold: float
    window_threshold: float
    window_threshold_confirmed: float
    feature_columns: list[str]
    pool: str
    window_pool: str
    tolerance: float
    precision_cfg: dict[str, Any] | None
    scoring_mode: str = "window_stack_aggregate"
    spread_weight: float = 0.25


def _load_ml_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "02_ML_Model" / "ml_config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _aggregator_config_path() -> Path:
    return PROJECT_ROOT / "06_AGGREGATOR" / "aggregator_config.yaml"


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    aggregator_cfg_path = _aggregator_config_path()
    if not aggregator_cfg_path.exists():
        raise FileNotFoundError(
            f"Aggregator config not found: {aggregator_cfg_path}. "
            "Train the window-stack models first (see README.md)."
        )
    with aggregator_cfg_path.open(encoding="utf-8") as handle:
        aggregator_cfg = yaml.safe_load(handle)["run_aggregator"]

    scoring_mode = str(aggregator_cfg.get("scoring_mode", "window_stack_aggregate")).lower()
    meta_cfg_path = _resolve(aggregator_cfg["meta_config"])
    with meta_cfg_path.open(encoding="utf-8") as handle:
        meta_settings = yaml.safe_load(handle)["train_meta"]

    feature_columns = [str(col) for col in meta_settings["feature_columns"]]
    model_keys = {
        "xgb": "xgboost_model",
        "lightgbm": "lightgbm_model",
        "random_forest": "random_forest_model",
        "extra_trees": "extra_trees_model",
    }
    base_models: dict[str, Any] = {}
    for col in feature_columns:
        path_key = model_keys[col]
        base_models[col] = joblib.load(_resolve(meta_settings[path_key]))

    _ensure_meta_scripts_path()
    meta_model = joblib.load(_resolve(aggregator_cfg["meta_model"]))
    calibrator_bundle = joblib.load(_resolve(aggregator_cfg["score_calibrator"]))
    calibrated = json.loads(_resolve(aggregator_cfg["calibrated_deployment"]).read_text(encoding="utf-8"))
    deployment_threshold = float(calibrated["deployment_threshold_calibrated"])
    window_threshold = deployment_threshold

    precision_cfg = None
    window_threshold_confirmed = deployment_threshold
    precision_path = _resolve(
        aggregator_cfg.get("precision_deployment", "06_AGGREGATOR/results/precision_deployment.json")
    )
    if precision_path.exists():
        precision_deployment = json.loads(precision_path.read_text(encoding="utf-8"))
        window_threshold_confirmed = float(
            precision_deployment.get("window_threshold_confirmed", deployment_threshold)
        )
        precision_cfg = {
            "function_threshold_triage": deployment_threshold,
            "window_threshold_triage": deployment_threshold,
            "window_threshold_confirmed": window_threshold_confirmed,
        }

    function_threshold = deployment_threshold
    tier_thresholds = calibrated.get("tier_thresholds") or {}
    function_review_threshold = float(
        tier_thresholds.get("review", calibrated.get("deployment_threshold_review", 0.26))
    )

    return ModelBundle(
        base_models=base_models,
        meta_model=meta_model,
        calibrator_bundle=calibrator_bundle,
        window_model=None,
        function_threshold=function_threshold,
        function_review_threshold=function_review_threshold,
        window_threshold=window_threshold,
        window_threshold_confirmed=window_threshold_confirmed,
        feature_columns=feature_columns,
        pool=str(aggregator_cfg.get("pool", "max")),
        window_pool=str(aggregator_cfg.get("window_pool", aggregator_cfg.get("pool", "max"))),
        tolerance=float(aggregator_cfg.get("max_pool_tolerance", 1e-6)),
        precision_cfg=precision_cfg,
        scoring_mode=scoring_mode,
        spread_weight=float(aggregator_cfg.get("spread_weight", 0.25)),
    )


def _web_scan_settings() -> dict[str, Any]:
    try:
        from services.config_service import load_config

        return load_config().get("scan") or {}
    except Exception:
        return {}


def get_encoder_model_path() -> str:
    """Resolved GraphCodeBERT checkpoint (local dir or Hugging Face hub id)."""
    from dataset_pipeline._hf_models import resolve_pretrained_source

    ml_cfg = _load_ml_config()
    encoder_cfg = ml_cfg.get("encoder", {})
    configured = encoder_cfg.get("model_path") or encoder_cfg.get("hub_id")
    if configured:
        return resolve_pretrained_source(_resolve(configured))
    cfg = _load_dataset_config()
    return resolve_pretrained_source(_resolve(cfg["10_embedder"]["model_path"]))


def encode_scan_windows(
    codes: list[str],
    *,
    batch_size: int,
) -> tuple[np.ndarray, None, None]:
    """Encode scan windows with GraphCodeBERT CLS vectors."""
    encoder = get_encoder()
    embeddings = encoder.encode(codes, batch_size=batch_size)
    return embeddings, None, None


@lru_cache(maxsize=1)
def get_encoder():
    embedder = _import_module("embedder", "10_embedder.py")
    cfg = _load_dataset_config()
    ecfg = cfg["10_embedder"]
    model_path = get_encoder_model_path()
    scan_cfg = _web_scan_settings()
    use_cpu = bool(scan_cfg.get("use_cpu_embedder", True))
    device = "cpu" if use_cpu else None
    fp16 = bool(ecfg.get("fp16", True)) and not use_cpu
    return embedder.GraphCodeEncoder(
        model_path,
        max_length=int(ecfg.get("max_length", 512)),
        fp16=fp16,
        device=device,
    )


def release_encoder() -> None:
    """Free GraphCodeBERT from GPU/CPU memory."""
    import gc

    import torch

    if get_encoder.cache_info().currsize == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    encoder = get_encoder()
    model = getattr(encoder, "model", None)
    if model is not None:
        try:
            model.cpu()
        except Exception:
            pass
        del encoder.model
    if hasattr(encoder, "tokenizer"):
        del encoder.tokenizer
    del encoder
    get_encoder.cache_clear()
    _collect_gpu_garbage()


def _collect_gpu_garbage() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _dispose_ml_model(model: Any) -> None:
    """Best-effort teardown for joblib / xgboost / lightgbm estimators."""
    if model is None:
        return
    try:
        if hasattr(model, "set_params"):
            try:
                model.set_params(device="cpu")
            except Exception:
                pass
    except Exception:
        pass
    booster = getattr(model, "get_booster", None)
    if callable(booster):
        try:
            del model._Booster
        except Exception:
            pass


def release_model_bundle(bundle: ModelBundle | None = None) -> None:
    """Unload tree ensemble, meta learner, calibrator, and window detector."""
    if bundle is None and get_model_bundle.cache_info().currsize:
        bundle = get_model_bundle()

    if bundle is not None:
        for key in list(bundle.base_models):
            _dispose_ml_model(bundle.base_models[key])
            del bundle.base_models[key]
        bundle.base_models.clear()

        _dispose_ml_model(bundle.meta_model)
        if bundle.window_model is not None:
            _dispose_ml_model(bundle.window_model)

        calibrator_bundle = bundle.calibrator_bundle
        if isinstance(calibrator_bundle, dict):
            calibrator = calibrator_bundle.get("calibrator")
            if calibrator is not None:
                _dispose_ml_model(calibrator)
                del calibrator_bundle["calibrator"]

        bundle.meta_model = None
        bundle.calibrator_bundle = {}
        bundle.window_model = None

    get_model_bundle.cache_clear()
    _collect_gpu_garbage()


def release_gpu_for_llm() -> None:
    """Drop embedder + all ML models from memory before loading Qwen."""
    release_encoder()
    release_model_bundle()


def get_pipeline_helpers():
    if str(DATA_PROCESSING_ROOT) not in sys.path:
        sys.path.insert(0, str(DATA_PROCESSING_ROOT))
    cleaner = _import_module("cleaner", "3_cleaner.py")
    batcher = _import_module("batcher", "9_batcher.py")
    embedder = _import_module("embedder", "10_embedder.py")
    return cleaner, batcher, embedder


def apply_calibrator(bundle: ModelBundle, raw_scores: np.ndarray) -> np.ndarray:
    method = str(bundle.calibrator_bundle.get("method", "isotonic")).lower()
    calibrator = bundle.calibrator_bundle["calibrator"]
    if method == "none":
        if hasattr(calibrator, "predict"):
            return np.clip(calibrator.predict(raw_scores), 0.0, 1.0).astype(np.float32)
        return np.clip(np.asarray(raw_scores, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if method == "isotonic":
        return np.clip(calibrator.predict(raw_scores), 0.0, 1.0).astype(np.float32)
    return calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1].astype(np.float32)


def score_function_windows(
    bundle: ModelBundle,
    *,
    function_group_id: str,
    window_rows: list[dict[str, Any]],
    window_embeddings: np.ndarray,
    on_step_complete: Callable[[str], None] | None = None,
    function_name: str = "",
) -> dict[str, Any]:
    if str(AGGREGATOR_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(AGGREGATOR_SCRIPTS))
    if str(XAI_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(XAI_SCRIPTS))

    from aggregator_common import build_function_records  # noqa: PLC0415
    from window_stack_common import pool_window_probs_for_function, score_window_embeddings  # noqa: PLC0415
    from xai_common import attach_window_code, tier_status_meta  # noqa: PLC0415

    from pipeline.progress_tracker import TREE_LABELS  # noqa: PLC0415

    suffix = f" · {function_name}" if function_name else ""

    if bundle.scoring_mode != "window_stack_aggregate":
        raise RuntimeError(
            f"Unsupported scoring_mode {bundle.scoring_mode!r}; expected window_stack_aggregate"
        )

    for col, model in bundle.base_models.items():
        if hasattr(model, "set_params"):
            try:
                model.set_params(device="cpu")
            except Exception:
                pass
        if on_step_complete is not None:
            label = TREE_LABELS.get(col, col)
            on_step_complete(f"{label} scoring windows{suffix}")

    raw_scores, window_probs, base_matrix = score_window_embeddings(
        window_embeddings,
        base_models=bundle.base_models,
        feature_columns=bundle.feature_columns,
        meta_model=bundle.meta_model,
        calibrator_bundle=bundle.calibrator_bundle,
    )
    pooled = pool_window_probs_for_function(
        window_probs,
        threshold=bundle.function_threshold,
        weight=bundle.spread_weight,
    )
    calibrated_score = float(pooled["pooled_risk"])
    raw_score = float(raw_scores.max()) if raw_scores.size else 0.0
    base_probs = {
        col: float(base_matrix[:, idx].mean()) if base_matrix.size else 0.0
        for idx, col in enumerate(bundle.feature_columns)
    }
    if on_step_complete is not None:
        on_step_complete(
            f"Meta + calibration · function {calibrated_score:.1%} "
            f"(peak window {pooled['base_max_risk']:.1%}){suffix}"
        )

    window_frame = pd.DataFrame(
        {
            "function_group_id": function_group_id,
            "window_id": [str(row["id"]) for row in window_rows],
            "window_index": [int(row["window_index"]) for row in window_rows],
            "label": 0,
            "embedding": list(window_embeddings),
            "window_prob": window_probs,
        }
    )

    function_frame = pd.DataFrame(
        {
            "function_group_id": [function_group_id],
            "label": [0],
            **({col: [base_probs[col]] for col in bundle.feature_columns} if bundle.feature_columns else {}),
        }
    )

    record = build_function_records(
        function_frame=function_frame,
        function_group_column="function_group_id",
        label_column="label",
        calibrated_scores=np.array([calibrated_score], dtype=np.float32),
        function_threshold=bundle.function_threshold,
        function_review_threshold=bundle.function_review_threshold,
        tiered_deployment=True,
        window_frame=window_frame,
        window_threshold=bundle.window_threshold,
        tolerance=bundle.tolerance,
        precision_cfg=None,
        scoring_mode=bundle.scoring_mode,
    )[0]

    code_index = {
        str(row["id"]): {"code": str(row["code"]), "cwe": None, "cve": None}
        for row in window_rows
    }
    enriched = attach_window_code(record, code_index)
    enriched["status_display"] = tier_status_meta(enriched)
    if on_step_complete is not None:
        status_label = enriched["status_display"]["label"]
        on_step_complete(f"Aggregation triage · {status_label}{suffix}")

    enriched["base_probs"] = base_probs
    enriched["raw_function_score"] = raw_score
    enriched["max_window_prob"] = float(window_probs.max()) if window_probs.size else 0.0
    enriched["window_spread_uplift"] = float(pooled.get("spread_uplift", 0.0))
    enriched["base_max_window_risk"] = float(pooled.get("base_max_risk", enriched["max_window_prob"]))

    flagged_set = {int(i) for i in enriched.get("flagged_window_indices") or []}
    confirmed_set = {int(i) for i in enriched.get("confirmed_window_indices") or []}
    contributing_set = {int(i) for i in enriched.get("contributing_window_indices") or []}
    enriched["all_windows"] = [
        {
            "window_index": int(row["window_index"]),
            "window_id": str(row["window_id"]),
            "window_prob": float(row["window_prob"]),
            "flagged": int(row["window_index"]) in flagged_set,
            "confirmed": int(row["window_index"]) in confirmed_set,
            "max_pool_contributor": int(row["window_index"]) in contributing_set,
        }
        for _, row in window_frame.iterrows()
    ]
    enriched["thresholds"] = {
        "function": float(bundle.function_threshold),
        "function_review": float(bundle.function_review_threshold),
        "window": float(bundle.window_threshold),
        "window_confirmed": float(bundle.window_threshold_confirmed),
        "scoring_mode": bundle.scoring_mode,
    }
    return enriched
