"""Load file-level scoring from 05_SCORE/file_score.py."""

from __future__ import annotations

import sys

from services.paths import PROJECT_ROOT

_SCORE_DIR = PROJECT_ROOT / "05_SCORE"
if str(_SCORE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCORE_DIR))

from file_score import build_file_score  # noqa: E402

__all__ = ["build_file_score"]
