# LightGBM baseline reproduction

Date: 2026-10-24. Working branch: `feature/machine-learning`.
Working directory: `/Users/marinescucatinca/Hackaton/decode-detect-hackathon`.
The installed environment is CPython 3.13.2; dependencies are pinned in `requirements-ml.txt` (which includes `requirements-cleaning.txt`). No packages were installed or upgraded during this change.

## Commands

From the repository root:

```sh
# If creating a new environment:
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-ml.txt

# Execute every cell in a new kernel using this Python executable:
.venv/bin/python notebooks/run_ml_baseline.py
```

The runner creates a temporary kernel specification, starts a new kernel with the current Python executable, runs every cell from the repository root, closes the kernel, and saves the executed notebook under ignored `outputs/02_ml_baseline.executed.ipynb`. It exits with an error if execution fails and retains notebook diagnostics.

## Scope and method

Input is only `data/processed/train_cleaned.parquet`, with 40,131 unique transactions and 1,250 positive labels. Official validation/test files are not read. The original user-written loading cell was retained and extended. Cleaning datasets and existing unrelated source notebooks/predictions are unchanged.

The fixed stratified split is `test_size=0.20`, `random_state=42`: 32,104 internal training rows (1,000 positive), 8,027 internal holdout rows (250 positive). A saved hash manifest lets other team models reuse these exact assignments. There are 18 explicitly allowlisted baseline features; IDs, raw JSON, raw datetime fields and annotation metadata are excluded. Category vocabularies are learned from the internal training subset only. `TRANSACTION_TYPE` and `FLAGS` are constant and retained for baseline consistency; they carry no fitted importance. No imputation, class weighting or resampling was added.

The user's team parameters were kept: 700 maximum estimators, learning rate 0.02, 63 leaves, depth 8, 150 minimum child samples, binary objective, no class weighting, column subsampling 0.8, L2 regularisation 1.0, random seed 0 and verbosity -1. Extra operational settings use four CPU threads, deterministic histograms and only Average Precision for evaluation. The installed LightGBM 4.7.0 API uses `eval_X`/`eval_y`.

Early stopping uses patience 50 and selects the best AP iteration, with at most 700 boosting rounds. The best model is iteration 674. Average Precision is verified against `sklearn.metrics.average_precision_score`; no fallback to `aucpr` is used. AP is not trapezoidal PR-AUC. Threshold metrics use a fixed 0.5; the top 1% diagnostic takes `ceil(0.01 * n)` rows, score descending and hash ascending for ties.

## Measured internal development results

| Metric | Value |
|---|---:|
| Average Precision | 0.341644 |
| Positive fraction (random-ranking reference) | 0.031145 |
| ROC AUC | 0.862222 |
| Precision at 0.5 | 0.793103 |
| Recall at 0.5 | 0.092000 |
| F1 at 0.5 | 0.164875 |
| Top 1% precision (81 transactions) | 0.543210 |
| Top 1% recall | 0.176000 |

Confusion matrix at 0.5: TN=7,771, FP=6, FN=227, TP=23. This threshold misses 227 of the 250 positive examples. No claim of final deployment/submission readiness is made.

All five code cells passed in a fresh kernel. Checks verified disjoint split assignments, complete hash coverage, numeric feature validity, finite scores in [0,1], exact native/sklearn AP agreement, CSV round trips and model reload prediction agreement. There were no missing features in either subset.

## Artifacts and interpretation

Local artifacts under ignored `outputs/ml_baseline/`: `metrics.json`, `split_manifest.csv`, `holdout_predictions.csv`, `feature_importance.csv`, `internal_train_model.txt`. The report includes input SHA-256, versions, feature order, category vocabularies, seeds and full parameters. The internal model is trained on 80% only. Existing `src/predictions.csv` was not overwritten, no official test predictions were generated, and no final 100%-train refit was performed.

The same holdout chooses the boosting iteration and reports metrics, so this is a development estimate, not an independent final test. Accounts/time periods can overlap in a random split. Timestamp and hourly price are prominent features, which warrants a later temporal/group audit. Graph/behaviour features from teammates are not yet integrated. The requested team 80/20 split remains unchanged; no hyperparameter or threshold search was performed.

References: [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html#metric-parameters) and [early stopping](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html). The pasted GeeksforGeeks article supplied general context; implementation and metric semantics were checked against the installed package and official documentation.

## Optional teammate features

The notebook now accepts explicitly configured graph/behaviour training feature files, with strict joins and isolated experiment outputs. Both default to `None`, preserving the original baseline. See [feature handoff instructions](feature_handoff.md). The notebook now has six code cells; real enriched scores are pending teammate delivery.
