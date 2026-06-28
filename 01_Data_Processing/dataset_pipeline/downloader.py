"""
Download raw vulnerability corpora into data/raw/.

Reads download_datasets.* from dataset_config.yaml. Which corpora to fetch
defaults to download_datasets.sources (mirrors training_shared.sources).

Usage:
    python dataset_pipeline/downloader.py
    python dataset_pipeline/downloader.py --datasets primevul diversevul
    python dataset_pipeline/downloader.py --stage 1c
    python dataset_pipeline/downloader.py --all
    python dataset_pipeline/downloader.py --list
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import yaml

_spec = importlib.util.spec_from_file_location(
    "dataset_pipeline._runtime",
    Path(__file__).resolve().parent / "_runtime.py",
)
_runtime = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_runtime)
_runtime.ensure_app_root(__file__)

from dataset_pipeline._loader import cfg as pcfg  # noqa: E402

DATASET_NAMES = pcfg.DATASET_NAMES


def _download_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    return dict(raw.get("download_datasets", {}))


def _source_toggles(cfg: dict[str, Any], raw: dict[str, Any]) -> dict[str, bool]:
    dl = _download_cfg(raw)
    if dl.get("sources"):
        return {name: bool(dl["sources"].get(name, False)) for name in DATASET_NAMES}
    return {name: bool(cfg.get("sources", {}).get(name, False)) for name in DATASET_NAMES}


def _artifact_specs(raw: dict[str, Any], dataset: str) -> dict[str, dict[str, Any]]:
    dl = _download_cfg(raw)
    block = dl.get(dataset)
    if not isinstance(block, dict):
        return {}
    return {name: dict(spec) for name, spec in block.items() if isinstance(spec, dict)}


def _resolve_dest(cfg: dict[str, Any], spec: dict[str, Any]) -> Path:
    if "dest_key" in spec:
        return pcfg.resolve_path(cfg, str(spec["dest_key"]))
    if "dest_dir_key" in spec:
        return pcfg.resolve_path(cfg, str(spec["dest_dir_key"]))
    raise KeyError(f"Artifact spec missing dest_key or dest_dir_key: {spec!r}")


def _skip_download(path: Path, *, skip_existing: bool, force: bool) -> bool:
    if force:
        return False
    if not skip_existing:
        return False
    if path.is_dir():
        return any(path.iterdir())
    return path.is_file() and path.stat().st_size > 0


def _download_huggingface(spec: dict[str, Any], dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    repo_id = str(spec["repo_id"])
    filename = str(spec["filename"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest.parent),
        local_dir_use_symlinks=False,
    )
    cached_path = Path(cached)
    if cached_path.resolve() != dest.resolve():
        if dest.exists():
            dest.unlink()
        shutil.move(str(cached_path), str(dest))


def _download_url(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "vulnera-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        chunk_size = 1024 * 1024
        with tmp.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
    if dest.exists():
        dest.unlink()
    tmp.replace(dest)


def _download_gdrive(spec: dict[str, Any], dest: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise ImportError("gdown is required for Google Drive downloads. pip install gdown") from exc

    file_id = str(spec["file_id"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dest), quiet=False, fuzzy=True)


def _download_zenodo(spec: dict[str, Any], dest: Path) -> None:
    record_id = str(spec["record_id"])
    filename = str(spec["filename"])
    url = f"https://zenodo.org/record/{record_id}/files/{filename}?download=1"
    _download_url(url, dest)


def _extract_zip_member(zip_path: Path, member_name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as src, dest.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def _extract_zip_all(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)


def _run_artifact(
    cfg: dict[str, Any],
    dataset: str,
    artifact_name: str,
    spec: dict[str, Any],
    *,
    skip_existing: bool,
    force: bool,
) -> None:
    kind = str(spec.get("kind", "url")).lower()
    dest = _resolve_dest(cfg, spec)

    if kind in {"huggingface", "hf"}:
        if _skip_download(dest, skip_existing=skip_existing, force=force):
            print(f"[skip] {dataset}/{artifact_name}: {dest} exists", flush=True)
            return
        print(f"[download] {dataset}/{artifact_name}: HF {spec['repo_id']}/{spec['filename']}", flush=True)
        _download_huggingface(spec, dest)
        print(f"  -> {dest}", flush=True)
        return

    if kind == "gdrive":
        if _skip_download(dest, skip_existing=skip_existing, force=force):
            print(f"[skip] {dataset}/{artifact_name}: {dest} exists", flush=True)
            return
        print(f"[download] {dataset}/{artifact_name}: Google Drive {spec['file_id']}", flush=True)
        _download_gdrive(spec, dest)
        print(f"  -> {dest}", flush=True)
        return

    if kind == "zenodo":
        with tempfile.TemporaryDirectory(prefix="vulnera_dl_") as tmp_dir:
            tmp_zip = Path(tmp_dir) / str(spec["filename"])
            check_path = dest
            if spec.get("rename_to") and "dest_key" in spec:
                check_path = dest.parent / str(spec["rename_to"])
            if spec.get("extract") and "dest_dir_key" in spec:
                check_path = dest
            if _skip_download(check_path, skip_existing=skip_existing, force=force):
                print(f"[skip] {dataset}/{artifact_name}: {check_path} exists", flush=True)
                return

            print(
                f"[download] {dataset}/{artifact_name}: Zenodo {spec['record_id']}/{spec['filename']}",
                flush=True,
            )
            _download_zenodo(spec, tmp_zip)

            if spec.get("extract_member"):
                _extract_zip_member(tmp_zip, str(spec["extract_member"]), dest)
                print(f"  -> {dest}", flush=True)
                return

            if spec.get("extract_glob"):
                pattern = str(spec["extract_glob"])
                with zipfile.ZipFile(tmp_zip) as archive:
                    matches = [name for name in archive.namelist() if Path(name).match(pattern)]
                    if not matches:
                        raise FileNotFoundError(f"No file matching {pattern!r} inside {tmp_zip.name}")
                    member = matches[0]
                    if spec.get("rename_to"):
                        out = dest.parent / str(spec["rename_to"])
                    elif "dest_dir_key" in spec:
                        out = dest / Path(member).name
                    else:
                        out = dest
                    _extract_zip_member(tmp_zip, member, out)
                    print(f"  -> {out}", flush=True)
                return

            if spec.get("extract"):
                _extract_zip_all(tmp_zip, dest)
                print(f"  -> {dest}/", flush=True)
                return

            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_zip), str(dest))
            print(f"  -> {dest}", flush=True)
        return

    if kind == "url":
        url = str(spec["url"])
        if _skip_download(dest, skip_existing=skip_existing, force=force):
            print(f"[skip] {dataset}/{artifact_name}: {dest} exists", flush=True)
            return
        print(f"[download] {dataset}/{artifact_name}: {url}", flush=True)
        _download_url(url, dest)
        print(f"  -> {dest}", flush=True)
        return

    raise ValueError(f"Unknown download kind {kind!r} for {dataset}/{artifact_name}")


def download_dataset(
    cfg: dict[str, Any],
    raw: dict[str, Any],
    dataset: str,
    *,
    skip_existing: bool,
    force: bool,
) -> None:
    specs = _artifact_specs(raw, dataset)
    if not specs:
        raise KeyError(f"No download_datasets.{dataset} block in dataset_config.yaml")

    for artifact_name, spec in specs.items():
        kind = str(spec.get("kind", "url")).lower()
        if kind in {"huggingface", "hf"} and str(spec.get("filename", "")).endswith(".zip"):
            dest = pcfg.resolve_path(cfg, str(spec["dest_key"]))
            zip_dest = dest.with_suffix(".zip")
            if _skip_download(dest, skip_existing=skip_existing, force=force):
                print(f"[skip] {dataset}/{artifact_name}: {dest} exists", flush=True)
                continue
            print(f"[download] {dataset}/{artifact_name}: HF {spec['repo_id']}/{spec['filename']}", flush=True)
            _download_huggingface(dict(spec), zip_dest)
            member = str(spec.get("extract_member", "MSR_data_cleaned.csv"))
            print(f"[extract] {dataset}/{artifact_name}: {member}", flush=True)
            _extract_zip_member(zip_dest, member, dest)
            zip_dest.unlink(missing_ok=True)
            print(f"  -> {dest}", flush=True)
            continue

        _run_artifact(
            cfg,
            dataset,
            artifact_name,
            spec,
            skip_existing=skip_existing,
            force=force,
        )


def selected_datasets(
    cfg: dict[str, Any],
    raw: dict[str, Any],
    *,
    cli_datasets: list[str] | None,
    stage: str | None,
    all_datasets: bool,
) -> list[str]:
    if cli_datasets:
        unknown = [name for name in cli_datasets if name not in DATASET_NAMES]
        if unknown:
            raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")
        return cli_datasets

    if stage:
        return list(pcfg.STAGE_SOURCES.get(stage, ()))

    if all_datasets:
        return list(DATASET_NAMES)

    toggles = _source_toggles(cfg, raw)
    enabled = [name for name in DATASET_NAMES if toggles.get(name, False)]
    if enabled:
        return enabled

    return list(pcfg.STAGE_SOURCES.get(str(cfg.get("stage", "1a")), ("primevul",)))


def list_plan(cfg: dict[str, Any], raw: dict[str, Any], datasets: list[str]) -> None:
    toggles = _source_toggles(cfg, raw)
    print("Configured download sources:")
    for name in DATASET_NAMES:
        flag = "yes" if toggles.get(name) else "no"
        print(f"  {name}: {flag}")
    print("\nArtifacts per selected dataset:")
    for dataset in datasets:
        specs = _artifact_specs(raw, dataset)
        print(f"  {dataset}:")
        for artifact_name, spec in specs.items():
            dest = _resolve_dest(cfg, spec)
            kind = spec.get("kind", "url")
            print(f"    - {artifact_name} ({kind}) -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw corpora into data/raw/.")
    parser.add_argument("--config", type=Path, default=None, help="Path to dataset_config.yaml")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_NAMES),
        help="Download only these corpora (e.g. primevul diversevul)",
    )
    parser.add_argument(
        "--stage",
        choices=["1a", "1b", "1c"],
        help="Download corpora required for pipeline stage 1a/1b/1c",
    )
    parser.add_argument("--all", action="store_true", help="Download all five corpora")
    parser.add_argument("--list", action="store_true", help="Show planned downloads and exit")
    parser.add_argument("--force", action="store_true", help="Re-download even when destination files exist")
    args = parser.parse_args()

    config_path = args.config or pcfg.APP_ROOT / "dataset_config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = pcfg.load_config(config_path)
    dl_cfg = _download_cfg(raw)
    skip_existing = bool(dl_cfg.get("skip_existing", True))
    force = bool(args.force or dl_cfg.get("force", False))

    datasets = selected_datasets(
        cfg,
        raw,
        cli_datasets=args.datasets,
        stage=args.stage,
        all_datasets=args.all,
    )
    if args.list:
        list_plan(cfg, raw, datasets)
        return

    if not datasets:
        print("No datasets selected. Enable download_datasets.sources or pass --datasets / --stage.", flush=True)
        return

    print(f"Downloading: {', '.join(datasets)}", flush=True)
    for dataset in datasets:
        download_dataset(cfg, raw, dataset, skip_existing=skip_existing, force=force)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
