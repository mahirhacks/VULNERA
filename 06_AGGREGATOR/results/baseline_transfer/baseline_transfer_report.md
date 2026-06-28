# Baseline transfer benchmark

Pretrained Devign-dataset classifiers evaluated on **VULNERA's temporal valid/test**
without retraining. Threshold tuned on valid (max F1). VULNERA uses deployment τ.

- Generated: 2026-06-28T09:14:55.725956+00:00
- Device: cuda
- Valid functions: 15,163
- Test functions: 14,170

## Test F1 (primary)

| Model | Precision | Recall | F1 | Threshold |
|-------|----------:|-------:|---:|----------:|
| VULNERA (window-stack + corroboration) | 0.354 | 0.697 | 0.470 | 0.32 |
| CodeBERT (fine-tuned on Devign) | 0.300 | 0.996 | 0.461 | 0.22 |
| GraphCodeBERT (fine-tuned on Devign) | 0.300 | 1.000 | 0.461 | 0.38 |

## Valid (threshold selection)

| Model | Precision | Recall | F1 | Threshold |
|-------|----------:|-------:|---:|----------:|
| VULNERA (window-stack + corroboration) | 0.363 | 0.714 | 0.481 | 0.32 |
| CodeBERT (fine-tuned on Devign) | 0.300 | 0.998 | 0.462 | 0.22 |
| GraphCodeBERT (fine-tuned on Devign) | 0.300 | 1.000 | 0.462 | 0.38 |

## Protocol

- **VULNERA**: trained on chronologically older corpora; scores from offline aggregator artifacts.
- **Baselines**: Hugging Face checkpoints fine-tuned on the Devign dataset; zero-shot transfer to VULNERA splits.
- Compare **test F1** on the same post-2020 functions — measures forward-era generalization, not in-distribution Devign leaderboard scores.

Reproduce: `python scripts/benchmark_transfer.py`
