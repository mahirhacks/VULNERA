from __future__ import annotations

from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = BACKEND_ROOT.parent
PROJECT_ROOT = WEB_ROOT.parent
SCANS_DIR = WEB_ROOT / "scans"
PROJECTS_DIR = WEB_ROOT / "projects"
CONFIG_PATH = WEB_ROOT / "web_config.yaml"
LLM_CONFIG_PATH = PROJECT_ROOT / "08_LLM" / "llm_config.yaml"
LLM_SCRIPTS_PATH = PROJECT_ROOT / "08_LLM" / "training_scripts"
