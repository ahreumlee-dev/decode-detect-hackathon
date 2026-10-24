# Methodology (fill in on event day)

## Data and preprocessing

## Feature groups

## Train/validation strategy and leakage safeguards

## Models and AP results

Initial LightGBM baseline: internal stratified 80/20 split (seed 42), team parameters unchanged. Average Precision 0.341644; precision/recall/F1 at threshold 0.5: 0.793103 / 0.092000 / 0.164875. The same holdout selects the best boosting iteration (674), so these are development scores. Official validation/test were not used. See [ML baseline report](ml_baseline.md).

## Reproducibility

Cleaning: [reproduction guide](data_cleaning.md). ML baseline: `requirements-ml.txt`; run `.venv/bin/python notebooks/run_ml_baseline.py` from the repository root using Python 3.13.2. Artifacts and the executed notebook are local under ignored `outputs/`.

## Person 4 behaviour comparison

Thanmaya's six behaviour features increased internal AP from 0.341644 to 0.417188 using identical seed-42 split hashes and the original LightGBM parameters. No official validation/test data were read. Missing first-transaction gaps were preserved natively. See [full comparison and provenance](thanmaya_behaviour_comparison.md).
