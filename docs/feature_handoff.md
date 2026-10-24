# Ready for Person 2 / Person 4 feature handoff

The current AP is a baseline reference. Teammate features may improve it; measure this on the same internal split instead of assuming an improvement. The ML notebook is prepared for LightGBM integration without changing the team's original parameters or 80/20 split.

## Files to receive

For the current experiments, receive separate **train-only** feature tables, for example:

- `data/processed/train_graph_features.csv` from Person 2.
- `data/processed/train_behaviour_features.csv` from Person 4.

CSV and Parquet are supported, and filenames are configurable. Each table must have `TRANSACTION_HASH` plus numeric/boolean features, exactly one row for each of the 40,131 cleaned training hashes. The row order can differ. No `is_suspicious`, family/difficulty/scenario/cohort metadata, predictions or scores are permitted. Missing values fail by default; the current Thanmaya behaviour input explicitly permits meaningful missing numeric gaps, which LightGBM handles natively. Infinite or nonnumeric feature values always fail; the feature engineer must define a justified missing-value policy. Labels stay solely in the cleaned training table.

Ask teammates to document feature definitions, time windows, currency/USD semantics and whether they use whole-window history. They must not use target labels to compute features. The join checks cannot detect a feature whose values were secretly derived from labels. Follow the team's agreed information-availability assumptions; a random 80/20 split does not by itself prevent account/time dependence.

Later final prediction work also needs corresponding **test-only** feature tables with the same feature names, one row for each of the 40,101 test hashes. Do not combine train and test into the same feature input file for this notebook. Official validation/test are not opened by the current notebook.

## Change one configuration cell

In `notebooks/02_ml_baseline.ipynb`, under **Feature handoff**, set:

```python
FEATURE_FILES = {
    "graph": ROOT / "data/processed/train_graph_features.csv",
    "behaviour": ROOT / "data/processed/train_behaviour_features.csv",
}
```

Leave either value as `None` to omit that group. Leave both as `None` for the original baseline. No graph or behaviour values are fabricated while waiting for delivery.

Run the entire notebook in a new kernel, or from the repository root:

```sh
.venv/bin/python notebooks/run_ml_baseline.py
```

Four LightGBM combinations are supported: baseline, baseline+graph, baseline+behaviour, baseline+graph+behaviour. RF and the full eight-model experiment grid described in the role document are a later team task, not implemented by this handoff adapter.

## Validation and results

`notebooks/ml_feature_inputs.py` performs strict one-to-one joins, preserves the original row/hash/index order, checks exact hash coverage, and prefixes features with `graph__` or `behaviour__` to avoid collisions. Original labels are preserved and only the explicit baseline and added feature columns enter X. The stratified 80/20 seed 42 split remains identical across feature groups. Params remain 700 maximum estimators, 0.02 learning rate, 63 leaves and the other original B_PARAMS.

Experiments save to separate ignored directories:

| Inputs | Output directory |
|---|---|
| Baseline only | `outputs/ml_baseline` |
| Graph only | `outputs/ml_baseline_graph` |
| Behaviour only | `outputs/ml_baseline_behaviour` |
| Both | `outputs/ml_baseline_graph_behaviour` |

Each report includes source file paths and SHA-256 fingerprints. If a baseline report exists and input checksum, split and model parameters match, the enriched experiment saves `comparison_to_baseline.csv` and prints the AP difference. Otherwise it explains why comparison was skipped. No baseline files are overwritten by an enriched experiment. The executed notebook preview is refreshed at `outputs/02_ml_baseline.executed.ipynb`.

Run adapter tests with:

```sh
.venv/bin/python -m unittest discover -s tests -p test_ml_feature_inputs.py -v
```

Ten synthetic tests passed for shuffled row alignment, unchanged baseline, missing/extra/duplicate hashes, forbidden metadata, invalid values/types, two-group naming and Parquet support. These fixtures test plumbing only; no synthetic model score is presented as a real feature experiment. Real graph/behaviour model scores remain pending until files arrive.

## Current tested delivery

Thanmaya (Person 4) is now integrated from commit `1935686`. Generate the train-only features with `.venv/bin/python notebooks/run_behaviour_features.py` before running ML. The notebook currently selects that behaviour file; graph remains `None`. Twelve adapter tests pass. See [the measured comparison](thanmaya_behaviour_comparison.md).
