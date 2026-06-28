# Vulnera — ML Pipeline Architecture

End-to-end view of how training data becomes models, and how uploaded C/C++ is scanned in the web app.

**Deployed detector:** window-stack ensemble (4 tree models → meta learner → isotonic calibration → function aggregation → signature fusion → optional LLM explanations).

---

## 1. Training data pipeline (`01_Data_Processing`)

Raw vulnerability corpora are normalized, cleaned, split, windowed, and embedded. Orchestrated by `dataset_pipeline/run_pipeline.py` (steps 1–9) and `10_embedder.py` (step 10).

```mermaid
flowchart TB
    subgraph RAW["Raw sources (data/raw/)"]
        PV[PrimeVul JSONL]
        DV[DiverseVul JSON]
        BV[Big-Vul CSV]
        CF[CVEfixes JSONL / SQL]
        SV[SecVulEval SQL + patches]
    end
```



```mermaid
flowchart TB
    subgraph STAGES["Stage presets (0_config.py)"]
    direction TB
        S1A["1a: PrimeVul"]
        S1B["1b: 1a + DiverseVul"]
        S1C["1c: 1b + BigVul, CVEfixes, SecVulEval"]
    end
```



```mermaid
flowchart TB

    RAW --> N1["1_normalizer.py<br/>core schema Parquet"]
    N1 --> N2["2_extractor.py<br/>C/C++ only"]
    N2 --> N3["3_cleaner.py<br/>strip comments, min tokens"]
    N3 --> N4["4_deduplicator.py<br/>exact + MinHash near-dup"]
    N4 --> N5["5_temporal_splitter.py<br/>train / valid / test by commit date"]
    N5 --> N6["6_builder.py<br/>train.parquet · valid · test"]
    N6 --> N7["7_validator.py<br/>gate: no cross-split leakage"]
    N7 --> N8["8_data_balancer.py<br/>~30% vuln / 70% benign"]
    N8 --> N9["9_batcher.py<br/>token windows ≤500 tok"]
    N9 --> SHARDS["data/processed/{split}/batch_*.parquet"]

    SHARDS --> E10["10_embedder.py<br/>GraphCodeBERT encode"]
    E10 --> WIN_EMB["{split}_window_embeddings.parquet<br/>one row per window"]
```

| Step | Output                        | Role                                              |
| ---- | ----------------------------- | ------------------------------------------------- |
| 1–6  | Whole-function Parquet splits | Labeled functions with metadata                   |
| 7    | Pass/fail gate                | Blocks pipeline on split leakage                  |
| 8    | Balanced splits               | Class ratio control                               |
| 9    | Window shards                 | Overlapping token windows per function            |
| 10   | Window embeddings             | One GraphCodeBERT vector per token window         |

Encoder checkpoint: `02_ML_Model/graphcodebert-base`. Config: `dataset_config.yaml` → `10_embedder`. Root manifest: `vulnera.yaml`.

Function-level risk is **not** computed by pooling embeddings. It is derived later by max-pooling **window probabilities** (see §2).

---



## 2. Window-stack model training (`03_TREE` → `06_AGGREGATOR`)

The production stack trains on **unpooled window embeddings**, then aggregates window probabilities to function level.

```mermaid
flowchart LR
    WIN_EMB["window_embeddings.parquet"] --> TWT["train_window_stack_trees.py<br/>XGB · LGBM · RF · ExtraTrees"]
    TWT --> WM["03_TREE/*/window/final/*.joblib"]

    WIN_EMB --> META_TBL["build_meta_table<br/>valid window predictions"]
    WM --> META_TBL
    META_TBL --> TM["train_meta.py<br/>logistic meta-learner"]
    TM --> MM["04_META/window_stack/selected/meta_model.joblib"]

    META_TBL --> CS["calibrate_scores.py<br/>isotonic calibration"]
    CS --> CAL["05_SCORE/window_stack/selected/<br/>score_calibrator.joblib<br/>calibrated_deployment.json"]

    WIN_EMB --> RA["run_aggregation.py<br/>window_stack_aggregate"]
    WM --> RA
    MM --> RA
    CAL --> RA
    RA --> ART["06_AGGREGATOR/artifacts<br/>function aggregation JSON"]
```



Orchestrator: `06_AGGREGATOR/training_scripts/run_window_stack_pipeline.py`

**Per-window scoring math:**
`embedding → 4 tree probs → meta learner → isotonic calibrator → window_prob`

**Per-function scoring math:**
`window_probs → composite max-pool + spread uplift → function_score_calibrated → deployment tiers (safe / review / vuln)`

Configs: `aggregator_config.yaml`, `meta_config.yaml`, `score_config.yaml`.

---



## 3. Runtime inference — web scan (`09_WEB`)

Upload path mirrors the training preprocessors, then reuses the trained window stack.

```mermaid
flowchart TB
    subgraph FE["09_WEB/front_end"]
        UP[Upload .c / .cpp]
        RES[ResultsPage · CodeEditor · FindingPeek]
        SET[SettingsPage · thresholds · LLM]
    end

    UP --> API["FastAPI main.py<br/>POST /scan"]
    API --> JOB["scan_worker.py<br/>async job + progress SSE"]
    JOB --> SCAN["scan_pipeline.run_scan()"]

    subgraph SCAN_PHASES["scan_pipeline phases"]
        direction TB
        P1["extract<br/>file_extractor.py"]
        P2["clean<br/>3_cleaner.py"]
        P3["window<br/>9_batcher.py"]
        P4["embed<br/>GraphCodeBERT · 10_embedder"]
        P5["trees + meta + calibrate<br/>score_function_windows()"]
        P6["aggregation<br/>tiers · flagged windows · markers"]
        P7["signature_match<br/>signature_engine + graduated boost"]
        P8["shap_tokens optional<br/>07_XAI token attribution"]
        P9["explain<br/>08_LLM Qwen or mock templates"]
        P10["file_score<br/>05_SCORE composite pool"]

        P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8 --> P9 --> P10
    end

    SCAN --> SCAN_PHASES
    P10 --> STORE["scan JSON store"]
    STORE --> RES
    SET --> API
```



**Model bundle** (`model_runtime.get_model_bundle()`): loads 4 window trees, meta model, calibrator, and deployment thresholds from `06_AGGREGATOR` + `05_SCORE` artifacts. Encoder: GraphCodeBERT (`02_ML_Model/ml_config.yaml`).

**Signature layer** (`signature_runtime.py`): regex / AST / comment CWE hints + optional embedding kNN; fuses with ML risk via graduated corroboration boost.

**XAI / LLM** (optional, settings-gated):

- SHAP token masking → re-embed → window prob delta (`shap_token_attribution.py`)
- Natural-language explanations via local Qwen2.5-Coder or template fallback (`grounded_explain.py`, `run_explainer.py`)

---



## 4. Full system map

```mermaid
flowchart TB
    subgraph OFFLINE["Offline (train once)"]
        DP["01_Data_Processing<br/>dataset pipeline 1–10"]
        TR["03_TREE · 04_META · 05_SCORE<br/>window-stack training"]
        DP --> TR
    end

    subgraph ARTIFACTS["Deployed artifacts"]
        ENC["GraphCodeBERT encoder"]
        TREES["4 × window tree models"]
        META["Meta learner"]
        CAL2["Calibrator + deployment JSON"]
        SIG["signature_catalog.yaml"]
        LLM["08_LLM local model optional"]
    end

    TR --> ARTIFACTS

    subgraph ONLINE["Online (per upload)"]
        WEB["09_WEB React UI"]
        BE["FastAPI backend"]
        WEB <--> BE
        BE --> PIPE["scan_pipeline"]
    end

    ARTIFACTS --> PIPE
    PIPE --> OUT["Per-function results<br/>markers · tiers · explanations"]
    OUT --> WEB
```



---



## 5. Key directories


| Path                                        | Purpose                                        |
| ------------------------------------------- | ---------------------------------------------- |
| `01_Data_Processing/dataset_pipeline/`      | Steps 1–10                                     |
| `01_Data_Processing/data/` | Raw, processed, embeddings |
| `02_ML_Model/`                              | GraphCodeBERT weights                          |
| `03_TREE/`                                  | Tree ensemble training + `window/final` models |
| `04_META/`                                  | Meta learner                                   |
| `05_SCORE/`                                 | Calibration + file-level composite scoring     |
| `06_AGGREGATOR/`                            | Aggregation eval + deployment config           |
| `07_XAI/`                                   | SHAP + explanation prompts                     |
| `08_LLM/`                                   | Local LLM config + weights                     |
| `09_WEB/`                                   | FastAPI backend + React frontend               |


