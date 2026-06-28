"""Evaluate graduated signature boost vs plateau on PrimeVul and 10_TEST."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "09_WEB" / "back_end"
RESULTS_DIR = ROOT / "06_AGGREGATOR" / "results" / "graduated_boost_eval"
FEATURE_DIR = ROOT / "06_AGGREGATOR" / "results" / "corroboration_tune"
CLAUDE_MD = ROOT / "tests" / "claude.md"
TEST_DIR = ROOT / "10_TEST"
DEPLOYMENT_PATH = ROOT / "05_SCORE" / "window_stack" / "selected" / "calibrated_deployment.json"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from pipeline.signature_engine import load_catalog  # noqa: E402
from pipeline.signature_runtime import (  # noqa: E402
    compute_graduated_boost_score,
    compute_signature_plateau_score,
    corroboration_allows,
)


def deployment_threshold() -> float:
    payload = json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    return float(payload.get("deployment_threshold_calibrated", 0.32))


def load_feature_frame(split: str) -> pd.DataFrame:
    path = FEATURE_DIR / f"{split}_signature_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run tests/tune_corroboration_omega.py first")
    return pd.read_parquet(path)


def predict_scores(
    frame: pd.DataFrame,
    *,
    tau: float,
    boost_cfg: dict[str, Any],
    corroboration_cfg: dict[str, Any],
    policy: str,
) -> np.ndarray:
    floor = float(load_catalog().get("known_confidence_floor", 0.45))
    omega = float(corroboration_cfg.get("omega", 0.15))
    cfg = dict(boost_cfg)
    scores: list[float] = []

    for row in frame.itertuples(index=False):
        func_score = float(row.function_score_calibrated)
        support = float(row.ml_support)
        if not bool(row.boost_eligible) or float(row.signature_confidence) < floor:
            scores.append(func_score)
            continue
        if not corroboration_allows(support, threshold=tau, omega=omega):
            scores.append(func_score)
            continue

        confidence = float(row.signature_confidence)
        if policy == "plateau":
            boosted = compute_signature_plateau_score(
                current_score=func_score,
                threshold=tau,
                confidence=confidence,
                boost_cfg=cfg,
            )
        else:
            boosted = compute_graduated_boost_score(
                current_score=func_score,
                support=support,
                threshold=tau,
                confidence=confidence,
                boost_cfg=cfg,
            )
        scores.append(boosted)
    return np.asarray(scores, dtype=float)


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, *, tau: float) -> dict[str, float]:
    y_pred = scores >= tau
    tp = int(np.sum((y_true == 1) & y_pred))
    fp = int(np.sum((y_true == 0) & y_pred))
    fn = int(np.sum((y_true == 1) & ~y_pred))
    tn = int(np.sum((y_true == 0) & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else 0.0,
    }


def parse_claude_md(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|") or "File" in line or "---" in line:
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 3:
            continue
        m = re.search(r"(\d+)\s*%", parts[2])
        if not m:
            continue
        rows.append({
            "file": parts[0].strip(),
            "function": parts[1].strip("` "),
            "claude_pct": int(m.group(1)),
            "claude_vuln": int(m.group(1)) >= 50,
        })
    return rows


def evaluate_10_test(*, tau: float) -> pd.DataFrame:
    from pipeline.scan_pipeline import run_scan

    load_catalog.cache_clear()
    expected = parse_claude_md(CLAUDE_MD)
    rows: list[dict[str, Any]] = []
    for path in sorted(TEST_DIR.glob("test_*.c")):
        result = run_scan(
            source=path.read_text(encoding="utf-8"),
            filename=path.name,
            llm_provider="mock",
            max_functions=50,
        )
        for fn in result.functions:
            score = float(fn.get("function_score_calibrated") or 0.0)
            attr = fn.get("pattern_attribution") or {}
            rows.append({
                "file": path.name,
                "function": str(fn.get("name") or ""),
                "vulnera_pct": round(score * 100, 1),
                "vulnera_flagged": bool(fn.get("function_flagged")),
                "pattern_category": attr.get("category"),
                "boost_mode": attr.get("boost_mode"),
                "ml_support": attr.get("ml_support_score"),
            })

    by_key = {(r["file"], r["function"]): r for r in rows}
    compared: list[dict[str, Any]] = []
    for exp in expected:
        got = by_key.get((exp["file"], exp["function"]))
        if got is None:
            compared.append({**exp, "vulnera_pct": None, "vuln_match": False})
            continue
        vuln_match = (got["vulnera_pct"] >= tau * 100) == exp["claude_vuln"]
        compared.append({**exp, **got, "vuln_match": vuln_match, "delta": got["vulnera_pct"] - exp["claude_pct"]})
    return pd.DataFrame(compared)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tau = deployment_threshold()
    catalog = load_catalog()
    boost_cfg = catalog.get("signature_risk_boost") or {}
    corroboration_cfg = catalog.get("corroboration") or {}

    split_metrics: dict[str, Any] = {}
    for split in ("valid", "test"):
        frame = load_feature_frame(split)
        y_true = frame["label"].to_numpy(dtype=int)
        ml_scores = frame["function_score_calibrated"].to_numpy(dtype=float)
        plateau_scores = predict_scores(
            frame, tau=tau, boost_cfg=boost_cfg, corroboration_cfg=corroboration_cfg, policy="plateau"
        )
        grad_scores = predict_scores(
            frame, tau=tau, boost_cfg=boost_cfg, corroboration_cfg=corroboration_cfg, policy="graduated"
        )
        split_metrics[split] = {
            "ml_only": binary_metrics(y_true, ml_scores, tau=tau),
            "plateau": binary_metrics(y_true, plateau_scores, tau=tau),
            "graduated": binary_metrics(y_true, grad_scores, tau=tau),
        }
        print(
            f"{split.upper()} F1: ML={split_metrics[split]['ml_only']['f1']:.4f} "
            f"plateau={split_metrics[split]['plateau']['f1']:.4f} "
            f"graduated={split_metrics[split]['graduated']['f1']:.4f}",
            flush=True,
        )

    print("\nScanning 10_TEST ...", flush=True)
    claude_df = evaluate_10_test(tau=tau)
    claude_df.to_csv(RESULTS_DIR / "claude_comparison_graduated.csv", index=False)
    scored = claude_df[claude_df["vulnera_pct"].notna()]
    vuln = scored[scored["claude_vuln"]]
    safe = scored[~scored["claude_vuln"]]
    claude_stats = {
        "triage_agree": float(scored["vuln_match"].mean()),
        "vuln_recall_flagged": float(vuln["vulnera_flagged"].mean()),
        "safe_specificity": float((~safe["vulnera_flagged"]).mean()),
        "over_safe": int((safe["vulnera_pct"] >= tau * 100).sum()),
        "mean_abs_delta": float(scored["delta"].abs().mean()),
    }

    summary = {"tau": tau, "primevul": split_metrics, "claude_10_test": claude_stats}
    (RESULTS_DIR / "graduated_boost_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
