"""Add app/ to sys.path when pipeline scripts are run as files."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_app_root(caller_file: str | Path) -> Path:
    """caller_file: __file__ of the script being executed."""
    root = Path(caller_file).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
