# Hybrid Lexical and Contextual Deep Learning for Non-Functional Requirement Classification

Code, data, and outputs for the paper *Hybrid Lexical and Contextual Deep Learning for Non-Functional Requirement Classification* (O. Newaishy, E. ElFakharany, M. M. Hafez; under review).

The study compares a hybrid classifier (NFReqANN-Hybrid: 5,000 TF-IDF features concatenated with a frozen 768-dimensional DistilBERT sentence embedding, fed to a dense network) with lexical, recurrent, frozen-transformer, and fine-tuned-transformer models for five-class NFR classification (Maintainability, Operability, Performance, Security, Usability).

## Repository contents

| File / folder | What it is |
|---|---|
| `PROMISE_ext.csv` | The 914-statement, five-class corpus (`class_name`, `sentence`). Identical to `NFR_five.csv` released by Rahman et al. (2024) — see *Data* below. |
| `nfr_five_provenance.csv` | The same 914 statements with provenance added: duplicate groups, traced PROMISE project, original label, de-duplication flag. Input to the revision notebook. |
| `nfr_classification.ipynb` | **Original experiments.** Nine model configurations on ten stratified 80/20 splits (seeds 34–43), with LIME, SHAP, and computational cost. |
| `NFR_revision_experiments.ipynb` | **Revision experiments.** Re-executes all models in one environment and adds tuned baselines, fine-tuned RoBERTa, a class-weighted hybrid, duplicate-aware and leave-one-project-out evaluation, block-level attribution, end-to-end timing, and corrected statistical tests. |
| `split_indices.csv` | The exact training, validation, and test rows of every split used in the revision (Protocols A, B, C). |
| `results/` | Outputs of the revision notebook: per-run predictions, summary tables, attribution, timing (see *Outputs* below). |
| `LICENSE` | MIT license (code). |

## Which notebook produced which part of the paper

| Paper element | Produced by |
|---|---|
| Table 6 (main results), Table 7 (uncleaned text), Tables 8–9 (hybrid per-class, confusion matrix), Table 12 (classifier cost), Tables 16–17 and Figures 3–4 (LIME, SHAP) | `nfr_classification.ipynb` |
| Table 10 (per-class F1 of all models) and Table 11 (corrected significance tests) | Computed from the per-seed outputs of `nfr_classification.ipynb`; the fine-tuned DistilBERT row of Table 10 comes from the revision notebook |
| Table 13 (end-to-end cost), Table 14 (tuned baselines, RoBERTa, class weighting), Table 15 (Protocols B and C), Table 18 (block-level attribution) | `NFR_revision_experiments.ipynb` |
| Dataset audit (Section 3.1, Table 4) | `nfr_five_provenance.csv` |

The revision notebook re-runs the original configurations with the same splits and settings; its Protocol A results reproduce Table 6 to within 0.011 macro-F1 for eight of the nine models (0.026 for NFReqBiLSTM). The paper's Tables 6–12 report the original runs; the revision results are reported separately.

## Data

`PROMISE_ext.csv` is the five-class corpus built by Rahman et al. (2024) from the PROMISE NFR dataset, PROMISE_exp, and the RE'17 Data Challenge dataset (published as `NFR_five.csv` at https://github.com/shanglapk/NFRDeepLearningFramework). It is used unchanged (MD5 `caf8c846e9f55cad9ccd23f0e4cc7a78`). Please cite the original authors when using it:

> Rahman K, Ghani A, Misra S, Rahman AU. A Deep Learning Framework for Non-Functional Requirement Classification. *Scientific Reports*. 2024;14:3216. https://doi.org/10.1038/s41598-024-52802-0

Class counts: Security 354, Usability 157, Operability 153, Maintainability 137, Performance 113.

### Provenance file (`nfr_five_provenance.csv`)

| Column | Meaning |
|---|---|
| `row_id` | Row number in `PROMISE_ext.csv` (0–913) |
| `class_name`, `sentence` | As in the corpus |
| `exact_dup_group` | Statements with the same id are identical after lower-casing and removing non-alphanumeric characters (102 duplicated pairs) |
| `near_dup_group` | Statements with character n-gram cosine similarity ≥ 0.90 share an id |
| `promise_exp_project`, `promise_exp_label` | Project and original label in PROMISE_exp, for the 348 statements that could be traced (15 projects); empty otherwise |
| `traceable_to_promise_exp` | Whether the statement was traced |
| `is_conflict` | Duplicate pair with conflicting labels (6 pairs) |
| `keep_dedup` | Row kept in the de-duplicated corpus (806 statements) |
| `source` | Traced PROMISE subset or other merged sources |

### Split file (`split_indices.csv`)

One row per statement per split: `protocol` (A, B, C), `fold`, `role` (`train`, `val`, `test`), `row_id`.

* **Protocol A** — the original ten stratified 80/20 splits (`seed34`–`seed43`) with a 15% stratified validation split.
* **Protocol B** — de-duplicated corpus, near-duplicate groups kept together, 2 × stratified group 5-fold cross-validation (`r34_k0` … `r35_k4`).
* **Protocol C** — leave-one-project-out over the 15 traceable projects (`P01` … `P15`); untraced statements are always in training, and near-duplicates of test statements are removed from training.

## How to reproduce

Both notebooks were run on Google Colab with a Tesla T4 GPU.

**Original experiments (`nfr_classification.ipynb`).** Upload `PROMISE_ext.csv` to Colab as `/content/NFR_five.csv` (the path the notebook expects) and run the cells. Each section trains one model family over the ten seeds and prints per-seed and aggregate results.

**Revision experiments (`NFR_revision_experiments.ipynb`).** Select a T4 GPU runtime and choose *Runtime → Run all*; upload `nfr_five_provenance.csv` when prompted. The notebook runs 37 jobs, each in its own process (TensorFlow models and PyTorch fine-tuning never share a process), checkpoints every fit, and resumes automatically if re-run. It takes about 3–4 hours and downloads `nfr_revision_results.zip` at the end. Set `USE_DRIVE = True` in the last cell to keep progress in Google Drive.

Software used for the revision run: Python 3.13, TensorFlow 2.21, PyTorch 2.11, Transformers 5.21, scikit-learn 1.6.1, NLTK 3.9.1 (exact versions in `results/environment_phase*.json`).

## Outputs (`results/`)

| Path | Contents |
|---|---|
| `runs/<protocol>/<model>/<fold>.json` | One file per fitted model: test row ids, true and predicted labels, class probabilities, selected hyper-parameters, training time |
| `summary/A_aggregate.csv`, `B_aggregate.csv` | Mean and s.d. of accuracy, macro-F1, and per-class F1 per model |
| `summary/A_tests_vs_hybrid.csv`, `B_tests_vs_hybrid.csv` | Corrected resampled t-tests against NFReqANN-Hybrid (accuracy and macro-F1, Holm-adjusted) |
| `summary/C_pooled.csv`, `C_bootstrap_vs_hybrid.csv`, `C_confusion_*.csv` | Leave-one-project-out results pooled over projects, project-level bootstrap tests, confusion matrices |
| `attribution/`, `summary/attribution_*.csv` | Block-level Shapley values, block ablation, expected-gradients shares, and top tokens for the hybrid (ten runs) |
| `timing_end_to_end_phase*.json` | End-to-end training and inference times on GPU and CPU |
| `transformer_lr_selection.json` | Validation macro-F1 for each candidate learning rate of the fine-tuned transformers |

## Model names

| Name in code | Model |
|---|---|
| `ANN` | TF-IDF, one hidden layer (2,048) |
| `NFReqANN-TFIDF` | TF-IDF, dense 2,048 → 256 → 128 |
| `NFReqANN-DistilBERT` | Frozen DistilBERT embedding, same dense network |
| `NFReqANN-Hybrid` | TF-IDF + frozen DistilBERT embedding, same dense network |
| `BiLSTM`, `NFReqBiLSTM` | Baseline and deeper bidirectional LSTM on word sequences |
| `SVM`, `LogReg` | Linear SVM and logistic regression on TF-IDF (default settings) |
| `*-tuned` | Hyper-parameters selected on the validation split |
| `NFReqANN-Hybrid-cw` | Hybrid with balanced class weights |
| `FT-DistilBERT`, `FT-RoBERTa` | Fully fine-tuned `distilbert-base-uncased` and `roberta-base` |

## License

The code is released under the MIT License (see `LICENSE`). The corpus is redistributed for reproducibility; its rights remain with its original creators (see *Data*).
