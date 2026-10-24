# XRPL AML Hackathon — team workspace

Collaboration scaffold only. Build judged models and features during the event.

## Teams / outputs
- 01 preprocessing: transaction-level cleaned train/test
- 02 graph analysis: transaction-level graph features
- 03 ML: baseline, Random Forest / LightGBM, average precision
- 04 behaviour: transaction-level temporal features
- 05 integration: merge on `TRANSACTION_HASH`, validate and submit

All feature tables must have exactly one row per `TRANSACTION_HASH`. Never merge labels into test data. Share processed data only as permitted by the organiser.

## Instructions to fill in on event day
Document dataset paths, package versions, exact run commands, random seeds, model selection and score here.

## Behaviour features: run order and reproducibility

**01 cleaning → 04 behaviour → 03 ML/prediction.** Numbers identify roles, not execution order. Person 4's notebook is `notebooks/04_behaviour_features.ipynb`; the current model integration is `src/starter_notebook.ipynb` and retains stratified 80/20 validation. Follow [the run guide](docs/behaviour_features.md) to regenerate tables, compare models and predict. Generated tables are ignored; the PR adds source code only.
