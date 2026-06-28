"""Metric helpers for baseline transfer benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_transfer import best_f1_threshold, binary_metrics  # noqa: E402


def test_binary_metrics_perfect_classifier() -> None:
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.9, 0.8])
    m = binary_metrics(y, prob, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_best_f1_threshold_finds_reasonable_cutoff() -> None:
    y = np.array([0, 0, 1, 1, 1])
    prob = np.array([0.1, 0.4, 0.55, 0.7, 0.9])
    threshold, f1 = best_f1_threshold(y, prob)
    assert 0.2 <= threshold <= 0.8
    assert f1 > 0.5
