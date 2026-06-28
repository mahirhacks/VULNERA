"""
Range-trust meta learner.

Each tree is trusted differently depending on which probability bin its score falls in.
Training table per window: | RF | XGB | LIGHTGBM | ET | label |

Pooling modes at inference:
  rank_decay — risk_rank1/1 + risk_rank2/10 + risk_rank3/100 + risk_rank4/1000
  weighted   — sum(trust_i * risk_i) / sum(trust_i)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

TREE_COLUMNS = ("xgb", "random_forest", "extra_trees", "lightgbm")
DISPLAY_NAMES = {
    "xgb": "XGB",
    "random_forest": "RF",
    "extra_trees": "ET",
    "lightgbm": "LIGHTGBM",
}
RANK_DIVISORS = (1.0, 10.0, 100.0, 1000.0)
PoolingMode = Literal["rank_decay", "weighted"]


@dataclass
class RangeTrustMeta:
    """Per-tree probability-bin trust profiles."""

    n_bins: int = 20
    smoothing: float = 2.0
    pooling_mode: PoolingMode = "weighted"
    global_positive_rate: float = 0.5
    bin_edges: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 1.0, 21))
    # trust_profiles[tree][bin_index] = P(label=1 | tree prob in bin)
    trust_profiles: dict[str, np.ndarray] = field(default_factory=dict)
    bin_counts: dict[str, np.ndarray] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, *, label_column: str = "label") -> RangeTrustMeta:
        labels = frame[label_column].astype(int).to_numpy()
        self.global_positive_rate = float(labels.mean()) if len(labels) else 0.5
        self.bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)

        for tree in TREE_COLUMNS:
            probs = frame[tree].astype(np.float32).to_numpy()
            trust = np.zeros(self.n_bins, dtype=np.float64)
            counts = np.zeros(self.n_bins, dtype=np.int64)

            for b in range(self.n_bins):
                low, high = self.bin_edges[b], self.bin_edges[b + 1]
                if b < self.n_bins - 1:
                    mask = (probs >= low) & (probs < high)
                else:
                    mask = (probs >= low) & (probs <= high)
                n = int(mask.sum())
                counts[b] = n
                if n > 0:
                    trust[b] = float(labels[mask].mean())
                else:
                    trust[b] = self.global_positive_rate

            # Laplace-style smoothing toward global rate for sparse bins
            for b in range(self.n_bins):
                n = counts[b]
                raw = trust[b]
                trust[b] = (raw * n + self.global_positive_rate * self.smoothing) / (n + self.smoothing)

            self.trust_profiles[tree] = trust.astype(np.float32)
            self.bin_counts[tree] = counts

        return self

    def _bin_index(self, prob: float) -> int:
        p = float(np.clip(prob, 0.0, 1.0))
        idx = int(np.searchsorted(self.bin_edges, p, side="right") - 1)
        return int(np.clip(idx, 0, self.n_bins - 1))

    def trust_for(self, tree: str, prob: float) -> float:
        profile = self.trust_profiles[tree]
        return float(profile[self._bin_index(prob)])

    def bin_range_label(self, tree: str, prob: float) -> str:
        idx = self._bin_index(prob)
        low, high = self.bin_edges[idx], self.bin_edges[idx + 1]
        trust = self.trust_profiles[tree][idx]
        return f"[{low:.2f},{high:.2f}) trust={trust:.3f}"

    def predict_window(
        self,
        tree_probs: dict[str, float],
        *,
        pooling_mode: PoolingMode | None = None,
    ) -> dict[str, Any]:
        mode = pooling_mode or self.pooling_mode
        entries: list[dict[str, Any]] = []
        for tree in TREE_COLUMNS:
            risk = float(tree_probs[tree])
            trust = self.trust_for(tree, risk)
            entries.append(
                {
                    "tree": tree,
                    "display": DISPLAY_NAMES[tree],
                    "risk": risk,
                    "trust": trust,
                    "bin": self.bin_range_label(tree, risk),
                }
            )

        ranked = sorted(entries, key=lambda row: row["trust"], reverse=True)
        terms: list[dict[str, Any]] = []
        meta_score = 0.0

        if mode == "weighted":
            weights = np.array([row["trust"] for row in entries], dtype=np.float64)
            risks = np.array([row["risk"] for row in entries], dtype=np.float64)
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                meta_score = float(risks.mean())
                norm_by_tree = {row["tree"]: 1.0 / len(entries) for row in entries}
            else:
                norm_by_tree = {row["tree"]: float(row["trust"] / weight_sum) for row in entries}
                meta_score = float(sum(norm_by_tree[row["tree"]] * row["risk"] for row in entries))

            for rank, row in enumerate(ranked, start=1):
                w_norm = norm_by_tree[row["tree"]]
                contribution = float(w_norm * row["risk"])
                terms.append(
                    {
                        "rank": rank,
                        "tree": row["display"],
                        "risk": row["risk"],
                        "trust": row["trust"],
                        "weight": w_norm,
                        "contribution": contribution,
                    }
                )
        else:
            for rank, row in enumerate(ranked, start=1):
                divisor = RANK_DIVISORS[rank - 1]
                contribution = row["risk"] / divisor
                meta_score += contribution
                terms.append(
                    {
                        "rank": rank,
                        "tree": row["display"],
                        "risk": row["risk"],
                        "trust": row["trust"],
                        "divisor": divisor,
                        "contribution": contribution,
                    }
                )

        return {
            "pooling_mode": mode,
            "tree_probs": {DISPLAY_NAMES[k]: float(tree_probs[k]) for k in TREE_COLUMNS},
            "ranked": terms,
            "meta_score": float(meta_score),
        }

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        scores: list[float] = []
        for _, row in frame.iterrows():
            tree_probs = {c: float(row[c]) for c in TREE_COLUMNS}
            scores.append(self.predict_window(tree_probs)["meta_score"])
        return np.asarray(scores, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_bins": self.n_bins,
            "smoothing": self.smoothing,
            "pooling_mode": self.pooling_mode,
            "global_positive_rate": self.global_positive_rate,
            "bin_edges": self.bin_edges.tolist(),
            "trust_profiles": {k: v.tolist() for k, v in self.trust_profiles.items()},
            "bin_counts": {k: v.tolist() for k, v in self.bin_counts.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RangeTrustMeta:
        obj = cls(
            n_bins=int(payload["n_bins"]),
            smoothing=float(payload.get("smoothing", 2.0)),
            pooling_mode=str(payload.get("pooling_mode", "weighted")),  # type: ignore[arg-type]
            global_positive_rate=float(payload.get("global_positive_rate", 0.5)),
        )
        obj.bin_edges = np.asarray(payload["bin_edges"], dtype=np.float64)
        obj.trust_profiles = {k: np.asarray(v, dtype=np.float32) for k, v in payload["trust_profiles"].items()}
        obj.bin_counts = {k: np.asarray(v, dtype=np.int64) for k, v in payload.get("bin_counts", {}).items()}
        return obj

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RangeTrustMeta:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def summarize_profiles(meta: RangeTrustMeta) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tree in TREE_COLUMNS:
        for b in range(meta.n_bins):
            low, high = meta.bin_edges[b], meta.bin_edges[b + 1]
            rows.append(
                {
                    "tree": DISPLAY_NAMES[tree],
                    "bin_low": low,
                    "bin_high": high,
                    "trust": float(meta.trust_profiles[tree][b]),
                    "count": int(meta.bin_counts[tree][b]),
                }
            )
    return pd.DataFrame(rows)
