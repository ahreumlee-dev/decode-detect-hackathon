# Thanmaya behaviour feature comparison — 2026-10-24

Source: [Thanmaya's notebook at commit 1935686](https://github.com/ahreumlee-dev/decode-detect-hackathon/blob/193568616947812e5d2e7d1db1180b06c3827752/notebooks/04_behaviour_features.ipynb), branch `behaviour-features-pr-04`, Person 4.

The branch contains a self-contained behaviour notebook rather than exported datasets. It was imported selectively, without merging the whole branch (which removes unrelated cleaning files). The original implementation and independent reference checks were inspected before execution. All six feature definitions and payment valuation code are preserved. Local adaptations generate train only, isolate outputs, and retain our established seed-42 split rather than the seed-0 suggestion in upstream prose. The installed environment was reused unchanged.

## Reproduction

From repository root, with the installed `.venv` and `requirements-ml.txt`:

```sh
.venv/bin/python notebooks/run_behaviour_features.py
.venv/bin/python notebooks/run_ml_baseline.py
.venv/bin/python -m unittest discover -s tests -p test_ml_feature_inputs.py -v
```

The ML notebook is currently configured for `outputs/thanmaya_behaviour/train_cleaned_features.parquet`. Set the behaviour path to `None` to reproduce the original baseline. The source and executed behaviour notebooks, provenance JSON and feature table are retained under ignored `outputs/`. No official validation/test files were read or scored.

## Feature definitions and checks

The six added features are sender transaction counts over 1h and 24h, seconds since the last strictly earlier outgoing transaction, unique recipients over 24h, sender outflow USD over 24h and sender inflow USD over 24h. Windows are [t-width,t), excluding all simultaneous payments. Each uses this split's label-free history. Recipient value uses delivered amount; sender principal uses the first executed transfer leg, never SendMax or the sum of conversion legs. Prices use the containing transaction hour and issuer/currency.

Generation produced 40,131 unique training hashes with exact cleaned input coverage and no labels in the feature table. There are 678 missing `seconds_since_previous_tx` values for first observed payments; these are intentionally preserved, not changed to zero. The internal train/holdout subsets contain 529/149 of these missing gaps. Other feature values are complete. LightGBM handles the gaps natively without an imputer. The feature adapter only permits missing values for explicitly configured groups, while rejecting infinities, invalid IDs, labels and annotation metadata.

## Measured comparison

Both experiments use **exactly the same saved transaction hash assignments**: 32,104 training rows (1,000 positive), 8,027 holdout rows (250 positive), stratified 80/20 seed 42. Cleaned input SHA-256, model parameters and operational settings were checked equal. The 700 / 0.02 / 63 team configuration remains unchanged; no tuning was performed.

| Metric | Cleaning baseline | + Thanmaya behaviour |
|---|---:|---:|
| Feature count | 18 | 24 |
| Average Precision | 0.341644 | 0.417188 |
| 100 × AP (internal reference) | 34.1644 | 41.7188 |
| Precision at 0.5 | 0.793103 | 0.921053 |
| Recall at 0.5 | 0.092000 | 0.140000 |
| F1 at 0.5 | 0.164875 | 0.243056 |
| Top 1% precision (81 rows) | 0.543210 | 0.691358 |
| Best iteration / rounds trained | 674 / 700 | 699 / 700 |

**AP change: +0.075545.** At the fixed 0.5 threshold, true positives increased 23 → 35 and false positives decreased 6 → 3. The new model still misses 215 of the 250 suspicious holdout transactions, so it is not a final solution.

These are internal development results: the same holdout selects the boosting iteration. Histories are computed across the train split before the random-row holdout, without labels; accounts and periods can overlap. This is the existing offline evaluation setup, not a chronological forecasting evaluation. No guarantee is made about official test improvement. Graph features are not included in this experiment.

Checks passed: Thanmaya's independent boundary/history reference tests, 12 adapter tests, both notebook fresh-kernel executions, exact split manifest equality, native/sklearn AP agreement, feature/model/output round trips and model reload. Result files: `outputs/ml_baseline_behaviour/metrics.json` and `comparison_to_baseline.csv`. Generated feature tables, models and experiment files remain local under ignored `outputs/`; they can be recreated by the commands above. Source notebooks include the explanation and verified ML metric outputs.
