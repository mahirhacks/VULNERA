"""Dataset pipeline orchestrator (steps 1–9)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "dataset_pipeline._runtime",
    Path(__file__).resolve().parent / "_runtime.py",
)
_runtime = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_runtime)
_runtime.ensure_app_root(__file__)

import argparse

from dataset_pipeline._loader import cfg as pcfg, load_step

run_normalize = load_step("1_normalizer").run_normalize
run_extract = load_step("2_extractor").run_extract
run_clean = load_step("3_cleaner").run_clean
run_dedup = load_step("4_deduplicator").run_dedup
run_temporal_split = load_step("5_temporal_splitter").run_temporal_split
run_build = load_step("6_builder").run_build
run_validate = load_step("7_validator").run_validate
run_balance = load_step("8_data_balancer").run_balance
run_window = load_step("9_batcher").run_window


def run_stage(stage: str, config_path=None, force_normalize: bool = False) -> None:
    cfg = pcfg.load_config(config_path)
    print(f"=== Stage {stage}: 1_normalizer (raw -> Parquet) ===")
    run_normalize(cfg, stage, force=force_normalize)
    print(f"=== Stage {stage}: 2_extractor (C/C++ only) ===")
    run_extract(cfg, stage)
    print(f"=== Stage {stage}: 3_cleaner ===")
    run_clean(cfg, stage)
    print(f"=== Stage {stage}: 4_deduplicator ===")
    run_dedup(cfg, stage)
    print(f"=== Stage {stage}: 5_temporal_splitter ===")
    run_temporal_split(cfg, stage)
    print(f"=== Stage {stage}: 6_builder ===")
    run_build(cfg, stage)
    print(f"=== Stage {stage}: 7_validator (gate) ===")
    if not run_validate(cfg, stage):
        sys.exit(1)
    print(f"=== Stage {stage}: 8_data_balancer ===")
    run_balance(cfg, stage)
    print(f"=== Stage {stage}: 9_batcher (token windowing) ===")
    run_window(cfg, stage)
    print(
        "Done: 1_normalizer -> 2_extractor -> 3_cleaner -> "
        "4_deduplicator -> 5_temporal_splitter -> 6_builder -> 7_validator -> "
        "8_data_balancer -> 9_batcher"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 dataset pipeline")
    parser.add_argument("--stage", choices=["1a", "1b", "1c"], default="1a")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--force-normalize",
        action="store_true",
        help="Overwrite existing data/interim/normalized/*.parquet",
    )
    args = parser.parse_args()
    run_stage(args.stage, args.config, force_normalize=args.force_normalize)


if __name__ == "__main__":
    main()
