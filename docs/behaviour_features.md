# Person 4 behaviour features and Person 3 handoff

Notebook numbers identify roles, not execution order: **01 cleaning → 04 behaviour → 03 ML/prediction**. Person 2 graph features are a parallel optional input and are not part of this PR's model. The current role-03 integration lives in src/starter_notebook.ipynb.

## Run from the repository root

Use Python 3.12 (verified with 3.12.11), with authorised raw data in data/raw/train, data/raw/validation and data/raw/test. No new dataset files are supplied by this PR. The existing main branch already tracks raw data and an older predictions CSV; those files are unchanged.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-behaviour.txt
python -m pytest -q tests
python notebooks/run_data_cleaning.py
python notebooks/run_data_cleaning.py --notebook 04_behaviour_features.ipynb
python compare_behaviour.py --train data/raw/train
python run_notebook_check.py --train data/raw/train --test data/raw/test
```

The notebook runner starts a fresh kernel using the current interpreter and stores executed notebooks under ignored outputs/. Alternatively run all cells of 01 and then 04 in VS Code using this environment. To run 04 without a kernel: `python -m src.generate_behaviour --train data/raw/train --test data/raw/test`.

01 writes data/processed/{train,validation,test}_cleaned.parquet and matching transfers, ignored by this PR. 04 writes outputs/train_behaviour_features.parquet and outputs/test_behaviour_features.parquet, with provenance manifests. Re-running replaces only generated files. The final model writes outputs/predictions.csv; do not use upstream src/predictions.csv as the current result.

## Contract

One output row per TRANSACTION_HASH, no labels, with six columns:

| Feature | Definition |
| --- | --- |
| sender_tx_count_1h | Earlier outgoing transactions in [t-3600,t) |
| sender_tx_count_24h | Earlier outgoing transactions in [t-86400,t) |
| seconds_since_previous_tx | Gap to latest strictly earlier outgoing timestamp across all supplied history; missing when unobserved |
| sender_unique_recipients_24h | Distinct final recipients in [t-86400,t) |
| sender_outflow_usd_24h | Actual sender-paid principal USD in earlier outgoing payments in [t-86400,t) |
| sender_inflow_usd_24h | Delivered USD received by this sender in earlier payments in [t-86400,t) |

All same-timestamp transactions are excluded, irrespective of transaction index. Empty windows have zero count/flow; windows containing unknown amounts have unknown sums. No missing payment amount is imputed to zero. Each split uses only its own history. Inputs are copied, not modified. Sorting/binary searches and sliding recipient counters avoid per-row full-data filtering.

The connector validates cleaned IDs, sender, timestamp and payment JSON against raw transactions. ACCOUNT supplies the original sender; Destination supplies the final recipient. Recipient USD is delivered amount at the containing-hour issuer-aware price; actual paid USD is the first executed transfer leg's asset value at its own reference price. No intermediate-leg sum, SendMax substitution, or fee inclusion. Person 1's delivered_usd is cross-checked. Raw transfer fallback without market rates is rejected.

Standalone cleaned-schema mapping is supported by src/behaviour_features.py using docs/behaviour_column_map.example.json. Notebook 04 uses the verified Person 1 connector automatically.

## Model and validation

Role 03 merges by TRANSACTION_HASH with one-to-one validation and exact hash coverage. load_for_ml rejects missing, stale or modified feature tables. The model includes starter features, Person 1's four transfer counts and three calendar fields, then six behaviour fields. It restores raw transaction order before the split so Person 1's export ordering cannot change holdout membership.

Stratified train_test_split(test_size=0.2, random_state=0) is unchanged: 32,104 fit and 8,027 holdout rows, 250 positives in holdout. Features are computed within full label-free train history before row holdout; the existing baseline includes whole-window/future context. This is offline random-row evaluation, not chronological forecasting or unseen-wallet evaluation. The organiser validation folder is unlabelled and is NOT the 20% holdout. Train and test histories are never concatenated.

Missing feature values remain missing in outputs. Constant -1 plus missing indicators and scaling are inside the model pipeline fitted on the training fold. Final fitting uses all labelled train rows for test scoring. No label metadata enters the feature matrices.

Paired comparison holds model, preprocessing and rows constant; only the six behaviour inputs differ. The holdout hash digest is recorded in outputs/behaviour_comparison.json. Previous current-calendar result was 25.1491 → 27.8683 (100 × AP); confirm the current run's JSON. A holdout gain is not a guaranteed leaderboard gain.

## Source provenance and scope

Person 1's self-contained cleaning notebook and regression tests are imported as SOURCE ONLY from commit 0011134 on feature/data-cleaning. The notebook now defines its function before use, fixes the weekend typo, checks datetime/timestamp consistency, uses exact containing-hour prices, and validates all exports. Its runner is extended only to support fresh-kernel execution of 04 too. It retains unique_tokens as a symbol count, not issuer-specific asset count.

This PR is based directly on origin/main, not on teammate dataset-containing merge commits. No new datasets, generated outputs, secrets or notebook execution outputs are included. Person 2's graph notebook and partial-coverage runner are intentionally outside this focused contribution.

## Verified on the PR branch

Fresh-kernel runs of 01 and 04 passed under Python 3.12.11. Cleaning exported 40,131 train, 33,862 validation and 40,101 test transactions with complete round-trip checks. 04 produced 40,131 train and 40,101 test behaviour rows. All 34 synthetic/regression tests passed. The model generated 40,101 unique valid test scores. The paired holdout reproduced 25.149131 without behaviour versus 27.868333 with behaviour, with identical original holdout hash digest 44463f987b60b356ac3697a4f8933708a807e8c77675897f719f812d568ee53b. Model, comparison and kernel execution outputs are local/ignored, not committed.
