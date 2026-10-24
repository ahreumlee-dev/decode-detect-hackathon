# Graph features for train_cleaned.parquet
AI assistance: OpenAI Codex adapted and tested this script on 2026-10-24.
Source repository unavailable; this is a standalone adaptation of the uploaded schema, not a repository edit.

## Run in Colab after uploading script and cleaned parquet
%pip install pandas==2.2.3 numpy==2.3.5 pyarrow==26.0.0
!python graph_features_cleaned.py --input /content/train_cleaned.parquet --output-dir /content/data/processed --split train
For a test file, run a separate invocation with --split test and its own --input. Histories are never combined.

## Semantics
All 40,131 unique hashes retained; 40,131 topology-supported; 9,668 native XRP FIFO-supported.
Windows [t-window,t) exclude all simultaneous events, include the lower boundary.
Partial-history rows retained. Seven-day counts describe available history, not seven complete days.
Observed counts of zero are valid; unsupported numerical measurements remain missing.
FIFO includes dust but excludes conversions, multiple legs, self-payments, SendMax and Paths.
Unsupported debits clear source inventory; simultaneous sends clear source inventory and remain missing.
Fees and opening balances are unknown; FIFO is a provenance assumption, not proof.
Topology uses ACCOUNT and JSON Destination once per customer transaction, never transfer legs.
No amount aggregation across currencies. Delivered USD is never interpreted as sender-paid USD.
No service registry was supplied: no services have been verified or automatically classified.
No retrospective forwarding, cycles or peel-chain claims are made by this causal feature export.

## Joining
joined = base.merge(behaviour, on="TRANSACTION_HASH", how="left", validate="one_to_one")
joined = joined.merge(graph, on="TRANSACTION_HASH", how="left", validate="one_to_one")
assert len(joined) == len(base)
Keep labels separate until modelling. Never inner-join or drop rows missing graph values.
Person 4's six features are unchanged. Fit missing-value handling on training holdout only.

## Comparison with Person 3 (pending team coordination)
Use the actual same stratified 80/20 hash manifest, seed 0, not merely the same seed with reordered rows.
Both arms include identical Person 4 features, same preprocessing/model and same holdout hashes; toggle only graph columns.
Preserve all rows, use train-fitted imputation with missingness indicators or a model supporting NaNs.
Agree the registry cutoff, model, metric, and whether label-free within-split history is allowed before reporting performance.
No external messages sent, repository edits, commits, or pushes performed.

## Checks passed
Uploaded-data row count, hash uniqueness, topology/FIFO coverage, no labels, graph prefixes.
Synthetic simultaneous timestamps, inclusive lower/exclusive upper boundaries, token and conversion NaNs, all-row preservation.
