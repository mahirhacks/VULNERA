#!/usr/bin/env python3
"""
VULNERA offline pipeline orchestrator.

Reads vulnera.yaml and runs numbered layers in order.

Examples:
    python scripts/run_vulnera_pipeline.py --list
    python scripts/run_vulnera_pipeline.py --stage data_prep
    python scripts/run_vulnera_pipeline.py --from embed --to calibrate
    python scripts/run_vulnera_pipeline.py --train
    python scripts/run_vulnera_pipeline.py --train --smoke-test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from vulnera_config import (  # noqa: E402
    DEFAULT_MANIFEST,
    expand_stage_list,
    format_args,
    load_manifest,
    stage_by_id,
)

PYTHON = sys.executable


def run_cmd(cmd: list[str], *, cwd: Path) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VULNERA offline ML pipeline.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--list", action="store_true", help="List pipeline stages and exit.")
    parser.add_argument("--stage", type=str, default=None, help="Run a single stage by id.")
    parser.add_argument("--from", dest="from_stage", type=str, default=None, help="First stage id (inclusive).")
    parser.add_argument("--to", dest="to_stage", type=str, default=None, help="Last stage id (inclusive).")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run train_bundle stages (trees → meta → calibrate → aggregate).",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Pass --smoke-test to supported training scripts.")
    parser.add_argument("--dataset-stage", choices=["1a", "1b", "1c"], default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if args.dataset_stage:
        manifest.setdefault("pipeline", {})["dataset_stage"] = args.dataset_stage

    all_stages = manifest.get("pipeline", {}).get("stages", [])
    if args.list:
        print("VULNERA pipeline stages:\n")
        for stage in all_stages:
            layer = stage.get("layer", "?")
            script = stage.get("script", "?")
            print(f"  {stage['id']:<16} [{layer}]  {script}")
        bundle = manifest.get("pipeline", {}).get("train_bundle", [])
        print(f"\ntrain_bundle: {', '.join(bundle)}")
        return

    if args.train:
        selected = expand_stage_list(manifest, ["train"])
    elif args.stage:
        selected = [stage_by_id(manifest, args.stage)]
    else:
        ids = [stage["id"] for stage in all_stages]
        start = ids.index(args.from_stage) if args.from_stage else 0
        end = ids.index(args.to_stage) if args.to_stage else len(ids) - 1
        if start > end:
            raise SystemExit("--from must precede --to in pipeline order")
        selected = all_stages[start : end + 1]

    smoke = ["--smoke-test"] if args.smoke_test else []
    for stage in selected:
        stage_id = stage["id"]
        layer_key = stage["layer"]
        layer_dir = PROJECT_ROOT / manifest["layers"][layer_key]["dir"]
        script_path = layer_dir / stage["script"]
        if not script_path.exists():
            raise FileNotFoundError(f"Stage {stage_id}: script not found: {script_path}")

        cmd = [PYTHON, stage["script"]]
        cmd.extend(format_args(stage.get("args", []), manifest))
        if args.smoke_test and stage_id in {"train_trees"}:
            cmd.extend(smoke)

        print(f"\n=== {stage_id} ({layer_key}) ===", flush=True)
        run_cmd(cmd, cwd=layer_dir)

    print("\nPipeline complete.", flush=True)


if __name__ == "__main__":
    main()
