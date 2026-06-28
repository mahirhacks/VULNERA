# VULNERA

C/C++ vulnerability triage: window-level ML ensemble, signature corroboration, and optional LLM explanations.

## Repository layout


| Layer         | Directory             | Config                   |
| ------------- | --------------------- | ------------------------ |
| 01 Data       | `01_Data_Processing/` | `dataset_config.yaml`    |
| 02 Encoder    | `02_ML_Model/`        | `ml_config.yaml`         |
| 03 Trees      | `03_TREE/`            | `tree_config.yaml`       |
| 04 Meta       | `04_META/`            | `meta_config.yaml`       |
| 05 Score      | `05_SCORE/`           | `score_config.yaml`      |
| 06 Aggregator | `06_AGGREGATOR/`      | `aggregator_config.yaml` |
| 07 XAI        | `07_XAI/`             | `xai_config.yaml`        |
| 08 LLM        | `08_LLM/`             | `llm_config.yaml`        |
| 09 Web        | `09_WEB/`             | `web_config.yaml`        |


**Root manifest:** `vulnera.yaml` — canonical paths and ordered pipeline stages.

**Architecture:** `00_Architecture/ML_Pipeline_Architecture.md`

## Quick start (reproducible pipeline)



### 1. Clone and install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

Install PyTorch for your CUDA build first if you have a GPU ([pytorch.org](https://pytorch.org)), then run `pip install -r requirements.txt`.

### 2. Download encoder weights

```bash
huggingface-cli download microsoft/graphcodebert-base --local-dir 02_ML_Model/graphcodebert-base
```



### 3. Download training data

Enable corpora in `01_Data_Processing/dataset_config.yaml` under `download_datasets.sources`, then run from `01_Data_Processing/`:

```bash
cd 01_Data_Processing

# Preview what will be downloaded (paths + sources)
python dataset_pipeline/downloader.py --list

# Download all five corpora (ignores source toggles)
python dataset_pipeline/downloader.py --all
```

Defaults mirror `training_shared.sources` (stage `1a` = PrimeVul only). Large corpora (CVEfixes ~13 GB, SecVulEval ~5 GB patches + SQL) are off by default.

Manual placement still works — see paths under `training_shared.paths` in `dataset_config.yaml`.

### 4. Run the offline pipeline

```bash
# List stages
python scripts/run_vulnera_pipeline.py --list

# Full path: data prep → embed → train → calibrate → evaluate
python scripts/run_vulnera_pipeline.py --from data_prep --to aggregate_test

# Smoke test (small slices)
python scripts/run_vulnera_pipeline.py --train --smoke-test
```

Equivalent manual bundle:

```bash
cd 06_AGGREGATOR
python training_scripts/run_window_stack_pipeline.py
```



### 5. Run the web app

See `09_WEB/README.md`. Short version:

```bash
# Terminal 1
cd 09_WEB/back_end && uvicorn main:app --port 8000

# Terminal 2
cd 09_WEB/front_end && npm install && npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

## What gets committed vs regenerated


| Committed                         | Gitignored (regenerate locally)             |
| --------------------------------- | ------------------------------------------- |
| Source code, YAML configs         | `01_Data_Processing/data/**`                |
| `vulnera.yaml`                    | Canonical paths and ordered pipeline stages |
| Tests, architecture docs          | `*.joblib` trained models                   |
| `.gitkeep` directory placeholders | Scan JSON, aggregator artifacts             |


To ship a **frozen deployment** (thresholds + calibrator), uncomment the exception in `.gitignore` for `05_SCORE/window_stack/selected/` or copy artifacts to a release branch.

## Configuration

- **Paths** — defined once in `vulnera.yaml` → `paths.`*; layer YAMLs reference `01_Data_Processing/data/embeddings`.
- **Dataset stage** — `vulnera.yaml` → `pipeline.dataset_stage` (`1a` / `1b` / `1c`).
- **Encoder** — `02_ML_Model/ml_config.yaml` → `encoder.model_path`
- **Scan settings** — `09_WEB/web_config.yaml`.



## Tests

```bash
python -m pytest tests/ -q
```

## Results

**Function-level** metrics at deployment threshold τ = 0.32 (window-stack ML + signature corroboration, ω = 0.15). Evaluated on held-out valid/test splits after the full offline pipeline (stage `1c` merged corpora).

**Temporal split** (`5_temporal_splitter`, optimized ~70/20/10 by CVE/commit year):

| Split | Commit years | Functions |
|-------|--------------|----------:|
| Train | 1999–2019 | 72,260 |
| Valid | 2020–2021 | 15,163 |
| Test  | 2022–2024 | 14,170 |

| Split | Precision | Recall | F1 |
|-------|----------:|-------:|---:|
| Valid | 0.367 | 0.742 | 0.491 |
| Test  | 0.355 | 0.727 | 0.477 |

### Baseline comparison (temporal transfer)

Same **valid/test functions** (commit years 2020–2024). VULNERA uses deployment τ = 0.32 (tuned on valid during pipeline training). Baselines are public Hugging Face checkpoints **fine-tuned on the Devign dataset** — evaluated here without retraining (cross-era transfer). Threshold per baseline = max F1 on valid.

```bash
# Requires processed splits + VULNERA aggregator artifacts (after aggregate_valid/test)
python scripts/benchmark_transfer.py

# Quick smoke (500 functions per split)
python scripts/benchmark_transfer.py --max-samples 500
```

| Model | Test P | Test R | Test F1 |
|-------|-------:|-------:|--------:|
| **VULNERA** (window-stack + corroboration) | **0.354** | **0.697** | **0.470** |
| GraphCodeBERT (Devign fine-tune, HF) | 0.300 | 1.000 | 0.461 |
| CodeBERT (Devign fine-tune, HF) | 0.300 | 0.996 | 0.461 |

*Full valid/test set (15,163 / 14,170 functions). Baselines are zero-shot transfer — not retrained on VULNERA data. Devign-family checkpoints tend toward very high recall and low precision on post-2020 commits; VULNERA keeps higher precision at comparable F1.*

Full table: `06_AGGREGATOR/results/baseline_transfer/baseline_transfer_report.md` (generated locally; commit after running if you want it on GitHub).

Published in-distribution Devign/LineVul paper scores are **not** directly comparable — they train and test on older static benchmarks. This table measures **forward generalization** to VULNERA's held-out era.

**10_TEST zero-day corpus vs Claude** (`tests/claude.md` reference scores on 71 synthetic C/C++ functions). Agreement = same vuln/safe triage as Claude (≥50% = vulnerable).

| Metric | Value |
|--------|------:|
| Overall triage agree | 67.6% |
| Vuln recall (flagged) | 68.8% |
| Safe specificity | 65.2% |

Sources: `06_AGGREGATOR/results/valid_aggregation_report.txt`, `test_aggregation_report.txt`, `corroboration_tune/corroboration_tune_summary.json`, `tests/claude_vs_vulnera_comparison.md`, `06_AGGREGATOR/results/baseline_transfer/baseline_transfer_report.md` (after `scripts/benchmark_transfer.py`).

## License

Final-year project — add your license here before public release.