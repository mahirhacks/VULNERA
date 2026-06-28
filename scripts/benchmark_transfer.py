"""
Cross-era transfer benchmark: run published pretrained detectors on VULNERA's
temporal valid/test splits (no retraining). Tune each baseline threshold on
valid (max F1), report precision / recall / F1 on valid and test.

VULNERA scores are read from aggregator artifacts when present (deployment τ).

Usage:
  python scripts/benchmark_transfer.py
  python scripts/benchmark_transfer.py --max-samples 500   # smoke
  python scripts/benchmark_transfer.py --models graphcodebert-devign
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from vulnera_config import DEFAULT_MANIFEST, PROJECT_ROOT, load_manifest

DEFAULT_SPLITS = ("valid", "test")
MAX_LENGTH = 512

BASELINES: dict[str, dict[str, str]] = {
    "graphcodebert-devign": {
        "display": "GraphCodeBERT (fine-tuned on Devign)",
        "model": "mahdin70/graphcodebert-devign-code-vulnerability-detector",
        "tokenizer": "microsoft/graphcodebert-base",
        "family": "Devign-line",
    },
    "codebert-devign": {
        "display": "CodeBERT (fine-tuned on Devign)",
        "model": "mahdin70/codebert-devign-code-vulnerability-detector",
        "tokenizer": "microsoft/codebert-base",
        "family": "Devign-line",
    },
}


@dataclass(frozen=True)
class SplitFrame:
    codes: list[str]
    labels: np.ndarray


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "support_pos": int((y_true == 1).sum()),
        "support_neg": int((y_true == 0).sum()),
        "n": int(len(y_true)),
    }


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, 0.0
    for threshold in np.arange(0.20, 0.81, 0.02):
        metrics = binary_metrics(y_true, y_prob, threshold=float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = float(threshold)
    return best_t, best_f1


def processed_split_path(manifest: dict[str, Any], split: str) -> Path:
    processed_root = PROJECT_ROOT / manifest["paths"]["processed_root"]
    return processed_root / "whole" / f"{split}.parquet"


def load_split_frame(path: Path, *, max_samples: int | None) -> SplitFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing split parquet: {path}")
    df = pd.read_parquet(path, columns=["code", "label"])
    df = df.dropna(subset=["code", "label"])
    df = df[df["code"].astype(str).str.strip() != ""]
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    labels = df["label"].astype(int).to_numpy()
    codes = df["code"].astype(str).tolist()
    return SplitFrame(codes=codes, labels=labels)


def load_deployment_threshold() -> float:
    score_cfg = PROJECT_ROOT / "05_SCORE" / "score_config.yaml"
    if score_cfg.is_file():
        with score_cfg.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        block = data.get("calibrated_deployment") or data.get("deployment") or {}
        if "deployment_threshold_calibrated" in block:
            return float(block["deployment_threshold_calibrated"])
    selected = PROJECT_ROOT / "05_SCORE" / "window_stack" / "selected" / "calibrated_deployment.json"
    if selected.is_file():
        payload = json.loads(selected.read_text(encoding="utf-8"))
        return float(payload["deployment_threshold_calibrated"])
    return 0.32


def load_vulnera_split_metrics(split: str, threshold: float) -> dict[str, Any] | None:
    artifact = PROJECT_ROOT / "06_AGGREGATOR" / "artifacts" / f"{split}_function_aggregation.parquet"
    if not artifact.is_file():
        return None
    df = pd.read_parquet(artifact, columns=["label", "function_score_calibrated"])
    metrics = binary_metrics(
        df["label"].to_numpy(dtype=int),
        df["function_score_calibrated"].to_numpy(dtype=float),
        threshold=threshold,
    )
    return {
        "id": "vulnera",
        "display": "VULNERA (window-stack + corroboration)",
        "family": "this work",
        "trained_on": "temporal 1c corpora (1999–2019 train)",
        "evaluated_on": f"VULNERA {split} split",
        "threshold_policy": f"fixed deployment τ={threshold:.2f}",
        split: metrics,
    }


def predict_probs(
    *,
    model_id: str,
    tokenizer_id: str,
    codes: list[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()

    probs: list[float] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(codes), batch_size), desc=model_id, leave=False):
            batch = codes[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.extend(float(p) for p in batch_probs)
    return np.asarray(probs, dtype=float)


def evaluate_baseline(
    baseline_id: str,
    *,
    valid: SplitFrame,
    test: SplitFrame,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    spec = BASELINES[baseline_id]
    valid_prob = predict_probs(
        model_id=spec["model"],
        tokenizer_id=spec["tokenizer"],
        codes=valid.codes,
        device=device,
        batch_size=batch_size,
    )
    test_prob = predict_probs(
        model_id=spec["model"],
        tokenizer_id=spec["tokenizer"],
        codes=test.codes,
        device=device,
        batch_size=batch_size,
    )
    threshold, valid_best_f1 = best_f1_threshold(valid.labels, valid_prob)
    return {
        "id": baseline_id,
        "display": spec["display"],
        "family": spec["family"],
        "huggingface_model": spec["model"],
        "trained_on": "Devign dataset (FFmpeg/QEMU; in-distribution for these checkpoints)",
        "evaluated_on": "VULNERA temporal valid/test (2020–2024 commit years)",
        "threshold_policy": "max F1 on VULNERA valid",
        "valid_threshold_search_best_f1": valid_best_f1,
        "valid": binary_metrics(valid.labels, valid_prob, threshold=threshold),
        "test": binary_metrics(test.labels, test_prob, threshold=threshold),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Baseline transfer benchmark",
        "",
        "Pretrained Devign-dataset classifiers evaluated on **VULNERA's temporal valid/test**",
        "without retraining. Threshold tuned on valid (max F1). VULNERA uses deployment τ.",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Device: {payload['device']}",
        f"- Valid functions: {payload['counts']['valid']:,}",
        f"- Test functions: {payload['counts']['test']:,}",
    ]
    if payload.get("max_samples"):
        lines.append(f"- Note: subsampled to {payload['max_samples']:,} rows per split (smoke run)")
    lines.extend(["", "## Test F1 (primary)", "", "| Model | Precision | Recall | F1 | Threshold |", "|-------|----------:|-------:|---:|----------:|"])

    for row in payload["models"]:
        test = row["test"]
        lines.append(
            f"| {row['display']} | {test['precision']:.3f} | {test['recall']:.3f} | "
            f"{test['f1']:.3f} | {test['threshold']:.2f} |"
        )

    lines.extend(["", "## Valid (threshold selection)", "", "| Model | Precision | Recall | F1 | Threshold |", "|-------|----------:|-------:|---:|----------:|"])
    for row in payload["models"]:
        valid = row["valid"]
        lines.append(
            f"| {row['display']} | {valid['precision']:.3f} | {valid['recall']:.3f} | "
            f"{valid['f1']:.3f} | {valid['threshold']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "- **VULNERA**: trained on chronologically older corpora; scores from offline aggregator artifacts.",
            "- **Baselines**: Hugging Face checkpoints fine-tuned on the Devign dataset; zero-shot transfer to VULNERA splits.",
            "- Compare **test F1** on the same post-2020 functions — measures forward-era generalization, not in-distribution Devign leaderboard scores.",
            "",
            "Reproduce: `python scripts/benchmark_transfer.py`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pretrained baselines on VULNERA valid/test")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(BASELINES),
        default=sorted(BASELINES),
        help="Baseline checkpoints to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None, help="Subsample each split (smoke test)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "06_AGGREGATOR" / "results" / "baseline_transfer",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    manifest = load_manifest(DEFAULT_MANIFEST)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    valid = load_split_frame(
        processed_split_path(manifest, "valid"),
        max_samples=args.max_samples,
    )
    test = load_split_frame(
        processed_split_path(manifest, "test"),
        max_samples=args.max_samples,
    )

    deployment_tau = load_deployment_threshold()
    models_out: list[dict[str, Any]] = []

    vulnera_valid = load_vulnera_split_metrics("valid", deployment_tau)
    vulnera_test = load_vulnera_split_metrics("test", deployment_tau)
    if vulnera_valid and vulnera_test:
        models_out.append(
            {
                **{k: vulnera_valid[k] for k in ("id", "display", "family", "trained_on", "evaluated_on", "threshold_policy")},
                "valid": vulnera_valid["valid"],
                "test": vulnera_test["test"],
            }
        )
        print(
            f"VULNERA (artifacts): valid F1={vulnera_valid['valid']['f1']:.4f} "
            f"test F1={vulnera_test['test']['f1']:.4f} @ tau={deployment_tau:.2f}",
            flush=True,
        )
    else:
        print("VULNERA aggregator artifacts not found — run aggregate_valid/test first.", flush=True)

    for baseline_id in args.models:
        print(f"\nEvaluating {baseline_id} on {device} ...", flush=True)
        models_out.append(
            evaluate_baseline(
                baseline_id,
                valid=valid,
                test=test,
                device=device,
                batch_size=args.batch_size,
            )
        )
        row = models_out[-1]
        print(
            f"  valid F1={row['valid']['f1']:.4f} @ t={row['valid']['threshold']:.2f} | "
            f"test F1={row['test']['f1']:.4f}",
            flush=True,
        )

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "max_samples": args.max_samples,
        "deployment_threshold_vulnera": deployment_tau,
        "counts": {"valid": len(valid.labels), "test": len(test.labels)},
        "models": models_out,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "baseline_transfer_summary.json"
    md_path = args.output_dir / "baseline_transfer_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    print(f"\nWrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
