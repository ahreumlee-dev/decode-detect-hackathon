# BathBytes XRPL AML - organised source snapshot

This archive reorganises the provided **39-point submission ZIP**; it does not claim to reproduce 39 AP without datasets nor improve it.

## Folders
- `src/`: source modules (`payments.py`, behaviour processing and generation).
- `notebooks/01_data_cleaning.ipynb`: cleaned transaction table generation.
- `notebooks/02_graph_experimental.ipynb`: graph research, NOT yet wired into final model.
- `notebooks/04_behaviour_features.ipynb`: generate six historical behaviour features.
- `notebooks/03_lightgbm_behaviour.ipynb`: recommended LightGBM experiment and predictions (12 baseline vs 18 with behaviour).
- `notebooks/run_data_cleaning.py`: runs cleaning and behaviour notebooks in fresh kernels.
- `docs/`: team notes; some originated from a separate logistic integration and may describe different scores.
- `archive/`: older, competing notebooks for reference only.

## From project root
1. Install: `python -m pip install -r requirements.txt`.
2. Place organiser source data in `data/raw/train/`, `data/raw/test/`, and `data/raw/validation/` as required. **Not included in ZIP.**
3. Run cleaning: `python notebooks/run_data_cleaning.py`.
4. Generate behaviour: `python -m src.generate_behaviour` (or run `notebooks/04_behaviour_features.ipynb`).
5. Run all cells in `notebooks/03_lightgbm_behaviour.ipynb` (from root or notebooks folder).
6. Submission file: `outputs/predictions_behaviour_fixed.csv`. Upload that CSV separately from the source-code ZIP.

## Validation and caveats
- Model trains on 18 features and tests on the same ordered 18 columns, matching by `TRANSACTION_HASH`.
- Assert that train rows = 40,131 and test rows = 40,101; test CSV has exactly `TRANSACTION_HASH,score`.
- The LightGBM notebook has **not** been run on raw data in this packaging environment.
- Original submission's `src/predictions.csv` is intentionally excluded as a generated artifact to prevent accidental stale uploads.
- Older notebooks had conflicting model/submission flows; do not run `archive/` for submission.
- The experimental graph notebook is not integrated with the LightGBM notebook.
- Check actual environment/library versions and choose a project LICENSE before final source submission (none was provided).
- Current cleaning notebook may rely on a validation/ split even though model scoring only needs train/test.
