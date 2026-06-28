"""Fine-grained scan progress — advances when each step finishes."""

from __future__ import annotations

from typing import Callable

ProgressCallback = Callable[[float, str], None]

TREE_LABELS = {
    "xgb": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
}


class ScanProgressTracker:
    def __init__(self, callback: ProgressCallback | None) -> None:
        self._callback = callback
        self._done = 0
        self._total = 1
        self._function_count = 0

    def set_plan(self, *, function_count: int, explain_steps: int) -> None:
        self._function_count = function_count
        self._total = max(4 + function_count * 7 + explain_steps + 1, 1)

    def set_explain_steps(self, explain_steps: int) -> None:
        self._total = max(4 + self._function_count * 7 + explain_steps + 1, 1)

    def complete(self, message: str) -> None:
        self._done += 1
        progress = min(self._done / self._total, 0.99)
        percent = int(progress * 100)
        if self._callback is not None:
            self._callback(progress, f"{message} · {percent}%")
