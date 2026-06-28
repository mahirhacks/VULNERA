"""
Full test-split threshold sweep for deployed calibrated function scores.

Usage:
    python 06_AGGREGATOR/training_scripts/threshold_sweep_deployed.py
    python 06_AGGREGATOR/training_scripts/threshold_sweep_deployed.py --min 0.1 --max 0.9 --step 0.01
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent
AGGREGATOR_ROOT = SCRIPTS_ROOT.parent
PROJECT_ROOT = AGGREGATOR_ROOT.parent


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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
        "flagged": int(y_pred.sum()),
    }


def main() -> None:
    parser = ArgumentParser(description="Sweep deployed calibrated function thresholds.")
    parser.add_argument("--min", type=float, default=0.10, dest="min_t")
    parser.add_argument("--max", type=float, default=0.90, dest="max_t")
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=AGGREGATOR_ROOT / "results")
    args = parser.parse_args()

    scores = pd.read_parquet(PROJECT_ROOT / "05_SCORE/logistic/test_calibrated_scores.parquet")
    disc = pd.read_parquet(AGGREGATOR_ROOT / "artifacts/test_function_aggregation.parquet")
    deploy = json.loads((PROJECT_ROOT / "05_SCORE/logistic/calibrated_deployment.json").read_text(encoding="utf-8"))
    precision_path = AGGREGATOR_ROOT / "results/precision_deployment.json"
    precision = json.loads(precision_path.read_text(encoding="utf-8")) if precision_path.exists() else {}
    window_triage = float(precision.get("window_threshold_triage", 0.26))
    window_confirmed = float(precision.get("window_threshold_confirmed", 0.42))
    current_t = float(deploy["deployment_threshold_calibrated"])

    y = scores["label"].astype(int).to_numpy()
    cal = scores["calibrated_score"].to_numpy(dtype=np.float32)
    max_win = disc["max_window_prob"].fillna(0).to_numpy(dtype=np.float32)

    rows: list[dict] = []
    for threshold in np.arange(args.min_t, args.max_t + args.step / 2, args.step):
        t = round(float(threshold), 2)
        func = binary_metrics(y, (cal >= t).astype(int))
        union = binary_metrics(y, ((cal >= t) | (max_win >= window_triage)).astype(int))
        confirmed = binary_metrics(y, ((cal >= t) & (max_win >= window_confirmed)).astype(int))
        rows.append(
            {
                "threshold": t,
                **{f"func_{k}": v for k, v in func.items()},
                **{f"union_{k}": v for k, v in union.items()},
                **{f"confirmed_{k}": v for k, v in confirmed.items()},
            }
        )

    curve = pd.DataFrame(rows)
    best_f1 = curve.loc[curve["func_f1"].idxmax()]
    policy = curve[curve["func_recall"] >= 0.60].sort_values(
        ["func_precision", "func_f1", "threshold"], ascending=[False, False, True]
    )
    current = curve.loc[(curve["threshold"] - current_t).abs().idxmin()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.min_t:.2f}_{args.max_t:.2f}".replace(".", "p")
    csv_path = args.output_dir / f"threshold_sweep_deployed_{tag}_full_test.csv"
    report_path = args.output_dir / f"threshold_sweep_deployed_{tag}_report.txt"
    curve.to_csv(csv_path, index=False)

    lines = [
        "VULNERA deployed threshold sweep — FULL test split",
        f"Functions: {len(y):,} | Positives: {int((y == 1).sum()):,} | Negatives: {int((y == 0).sum()):,}",
        f"Sweep: {args.min_t:.2f} – {args.max_t:.2f} (step {args.step:.2f}) on calibrated function scores",
        f"Current deployment: {current_t:.2f} | Window triage: {window_triage:.2f} | Confirmed window: {window_confirmed:.2f}",
        "",
        f"CURRENT @ {current_t:.2f}",
        f"  Function:   P={current.func_precision:.4f} R={current.func_recall:.4f} F1={current.func_f1:.4f} flagged={int(current.func_flagged):,}",
        f"  Union:      P={current.union_precision:.4f} R={current.union_recall:.4f} F1={current.union_f1:.4f} flagged={int(current.union_flagged):,}",
        f"  Confirmed:  P={current.confirmed_precision:.4f} R={current.confirmed_recall:.4f} F1={current.confirmed_f1:.4f} flagged={int(current.confirmed_flagged):,}",
        "",
        "BEST FUNCTION F1",
        f"  t={best_f1.threshold:.2f} P={best_f1.func_precision:.4f} R={best_f1.func_recall:.4f} F1={best_f1.func_f1:.4f} flagged={int(best_f1.func_flagged):,}",
    ]
    if not policy.empty:
        row = policy.iloc[0]
        lines.extend(
            [
                "",
                "BEST PRECISION @ recall>=0.60 (function only)",
                f"  t={row.threshold:.2f} P={row.func_precision:.4f} R={row.func_recall:.4f} F1={row.func_f1:.4f} flagged={int(row.func_flagged):,}",
            ]
        )
    lines.extend(
        [
            "",
            "FUNCTION-LEVEL CURVE",
            " thr   prec  recall     f1  flagged    TP   FN   FP",
            "-" * 56,
        ]
    )
    for _, row in curve.iterrows():
        lines.append(
            f"{row.threshold:5.2f} {row.func_precision:6.3f} {row.func_recall:7.3f} "
            f"{row.func_f1:6.3f} {int(row.func_flagged):8,d} "
            f"{int(row.func_tp):5,d} {int(row.func_fn):4,d} {int(row.func_fp):5,d}"
        )

    report = "\n".join(lines) + "\n"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {csv_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
