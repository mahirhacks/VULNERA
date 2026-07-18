# VULNERA

C/C++ vulnerability triage using a window-level ML ensemble, signature corroboration, explainable attribution, and optional local LLM explanations.

## Results

### Evaluation protocol

VULNERA is evaluated at the **function level** using a forward temporal split rather than a random split. The stage `1c` experiment merges PrimeVul, DiverseVul, Big-Vul, CVEfixes, and SecVulEval, then assigns functions chronologically by CVE/commit year.

| Split | Commit years | Functions |
| --- | --- | ---: |
| Train | 1999–2019 | 72,260 |
| Validation | 2020–2021 | 15,163 |
| Test | 2022–2024 | 14,170 |

The deployment threshold is selected on validation data and then applied to the held-out test period. The final detector uses threshold `τ = 0.32`, with signature corroboration width `ω = 0.15`.

### Final function-level performance

| Split | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Validation | 0.367 | 0.742 | 0.491 |
| Test | 0.355 | 0.727 | 0.477 |

These results use the window-stack ML detector plus corroborated signature evidence. The system is designed as a conservative triage tool: it prioritizes vulnerability recall and presents flagged code for human review rather than replacing static analysis or expert assessment.

### Baseline comparison: temporal transfer

The transfer benchmark evaluates all models on the same 2020–2024 validation and test functions. VULNERA uses its validation-tuned deployment threshold. The public CodeBERT and GraphCodeBERT checkpoints were fine-tuned on Devign and are evaluated without retraining on VULNERA data; their thresholds are selected by maximum validation F1.

| Model | Test precision | Test recall | Test F1 |
| --- | ---: | ---: | ---: |
| **VULNERA window-stack ML** | **0.354** | 0.697 | **0.470** |
| GraphCodeBERT (Devign fine-tune) | 0.300 | **1.000** | 0.461 |
| CodeBERT (Devign fine-tune) | 0.300 | 0.996 | 0.461 |

The baseline table measures **cross-era transfer**, not in-distribution Devign performance. Published Devign or LineVul paper results are therefore not directly comparable. The full benchmark contains 15,163 validation and 14,170 test functions.

Full report: [`06_AGGREGATOR/results/baseline_transfer/baseline_transfer_report.md`](06_AGGREGATOR/results/baseline_transfer/baseline_transfer_report.md)

### Synthetic-corpus agreement analysis

The `10_TEST` corpus contains 71 synthetic C/C++ functions. The following values measure agreement with the reference Claude assessments in `tests/claude.md`; they are supplementary agreement results, not ground-truth accuracy.

| Metric | Value |
| --- | ---: |
| Overall triage agreement | 67.6% |
| Vulnerable-case recall | 68.8% |
| Safe-case specificity | 65.2% |

Result sources:

- `06_AGGREGATOR/results/valid_aggregation_report.txt`
- `06_AGGREGATOR/results/test_aggregation_report.txt`
- `06_AGGREGATOR/results/corroboration_tune/corroboration_tune_summary.json`
- `06_AGGREGATOR/results/baseline_transfer/baseline_transfer_report.md`
- `tests/claude_vs_vulnera_comparison.md`

## Architecture

### Detection pipeline

```text
C/C++ function
  → cleaned, token-aware windows
  → GraphCodeBERT CLS embeddings
  → XGBoost + LightGBM + Random Forest + Extra Trees
  → meta-learner
  → isotonic calibration
  → max-window aggregation + spread uplift
  → corroborated CWE signatures
  → optional SHAP attribution and local LLM explanation
  → function/file triage report
```

Training uses chronologically older vulnerability data. At runtime, the FastAPI backend applies the same preprocessing and scoring path to uploaded C/C++ files, while the React interface presents function risk, contributing windows, code markers, pattern evidence, and explanations.

Detailed architecture: [`00_Architecture/ML_Pipeline_Architecture.md`](00_Architecture/ML_Pipeline_Architecture.md)

### Repository layout

| Layer | Directory | Config |
| --- | --- | --- |
| 01 Data | `01_Data_Processing/` | `dataset_config.yaml` |
| 02 Encoder | `02_ML_Model/` | `ml_config.yaml` |
| 03 Trees | `03_TREE/` | `tree_config.yaml` |
| 04 Meta | `04_META/` | `meta_config.yaml` |
| 05 Score | `05_SCORE/` | `score_config.yaml` |
| 06 Aggregator | `06_AGGREGATOR/` | `aggregator_config.yaml` |
| 07 XAI | `07_XAI/` | `xai_config.yaml` |
| 08 LLM | `08_LLM/` | `llm_config.yaml` |
| 09 Web | `09_WEB/` | `web_config.yaml` |

The root manifest, [`vulnera.yaml`](vulnera.yaml), defines canonical paths and the ordered offline pipeline.

## Commands

### 1. Install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

For GPU execution, install the appropriate PyTorch build first, then install `requirements.txt`. See [pytorch.org](https://pytorch.org) for platform-specific installation commands.

### 2. Download GraphCodeBERT

```bash
huggingface-cli download microsoft/graphcodebert-base --local-dir 02_ML_Model/graphcodebert-base
```

### 3. Download training data

Enable the required corpora under `download_datasets.sources` in `01_Data_Processing/dataset_config.yaml`, then run:

```bash
cd 01_Data_Processing

# Preview sources and destinations
python dataset_pipeline/downloader.py --list

# Download all five corpora, regardless of source toggles
python dataset_pipeline/downloader.py --all
```

The default stage is `1a` (PrimeVul only). Reproducing the reported merged-corpus results requires stage `1c` and all five corpora. CVEfixes and SecVulEval are disabled by default because their downloads are large. Manual dataset placement is also supported through `training_shared.paths` in `dataset_config.yaml`.

### 4. Run the offline pipeline

```bash
# Show the ordered stages
python scripts/run_vulnera_pipeline.py --list

# Reproduce the reported merged-corpus experiment
python scripts/run_vulnera_pipeline.py --dataset-stage 1c --from data_prep --to aggregate_test

# Small end-to-end smoke test
python scripts/run_vulnera_pipeline.py --train --smoke-test
```

The equivalent model-training and evaluation bundle, after data preparation and embedding, is:

```bash
cd 06_AGGREGATOR
python training_scripts/run_window_stack_pipeline.py
```

### 5. Reproduce the transfer benchmark

This command requires the processed validation/test splits and VULNERA aggregation artifacts:

```bash
# Full validation and test sets
python scripts/benchmark_transfer.py

# Quick 500-function smoke run per split
python scripts/benchmark_transfer.py --max-samples 500
```

### 6. Run the web application

See [`09_WEB/README.md`](09_WEB/README.md) for complete instructions.

```bash
# Terminal 1: backend
cd 09_WEB/back_end
uvicorn main:app --port 8000

# Terminal 2: frontend
cd 09_WEB/front_end
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Configuration

- **Canonical paths:** `vulnera.yaml` → `paths.*`
- **Dataset stage:** `vulnera.yaml` → `pipeline.dataset_stage` (`1a`, `1b`, or `1c`)
- **Dataset processing:** `01_Data_Processing/dataset_config.yaml`
- **Encoder:** `02_ML_Model/ml_config.yaml`
- **Aggregation:** `06_AGGREGATOR/aggregator_config.yaml`
- **Scan and UI runtime:** `09_WEB/web_config.yaml`
- **Optional local LLM:** `08_LLM/llm_config.yaml`

## Tests

Run the Python test suite from the repository root:

```bash
python -m pytest tests/ -q
```

Validate the frontend separately:

```bash
cd 09_WEB/front_end
npm run lint
npm run build
```

## Generated artifacts

| Committed | Regenerated locally |
| --- | --- |
| Source code and YAML configuration | Raw and processed datasets |
| `vulnera.yaml` | Embedding Parquets |
| Tests and architecture documentation | Trained model artifacts |
| Directory placeholders | Scan JSON and most evaluation artifacts |

To ship a frozen deployment, explicitly include the selected tree models, meta-model, score calibrator, and calibrated deployment configuration in a release artifact. The default `.gitignore` keeps large generated files out of the source repository.

## License

MIT — see [`LICENSE`](LICENSE).
