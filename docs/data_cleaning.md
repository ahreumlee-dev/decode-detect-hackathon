# Reproduce data cleaning

Validated on 2026-10-24, branch `feature/data-cleaning`, base commit `f0fd5f4`.
Working directory: `/Users/marinescucatinca/Hackaton/decode-detect-hackathon`.
Environment: CPython 3.13.2, macOS-26.6-arm64-arm-64bit-Mach-O, existing `.venv`.
Dependencies are pinned separately in `requirements-cleaning.txt`; the team's model dependencies are unchanged.

- `pandas==3.1.0`
- `numpy==2.5.4`
- `pyarrow==26.0.0`
- `nbformat==5.11.1`
- `nbclient==0.11.0`
- `nbconvert==7.17.2`
- `ipykernel==7.4.0`
- `jupyter_client==8.10.0`

## Commands

From the repository root, with the provided raw data in `data/raw/{train,validation,test}`:

```sh
# For a newly created environment using Python 3.13.2:
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-cleaning.txt

# Exact command used for the fresh-kernel execution:
.venv/bin/python notebooks/run_data_cleaning.py

# Regression checks:
.venv/bin/python -m unittest discover -s tests -p test_data_cleaning.py -v
```

The existing environment was inspected and reused without installing or upgrading packages.
The runner launches a new kernel from the current Python executable with a temporary kernelspec, executes every cell from the repository root, closes the kernel, and writes the executed notebook to ignored `outputs/01_data_cleaning.executed.ipynb`. Failures produce a nonzero exit status and retain notebook diagnostics. The source notebook contains no saved outputs or execution state.

## Code changes

The local working notebook already contained a complete `process_dataset` definition in its final section, after earlier calls. Commit `f0fd5f4` did not save that definition; commit `3f22682` contains the historical implementation with the `x["is_weekend"]` typo. The replacement uses the local definition, cross-checked against this history, defined before its first call. Exploratory/repeated exports, `inspect.getsource`, dynamic source rewriting and IPython history recovery were removed. The pre-existing untracked `notebooks/data_cleaning_function_fixed.py` remains untouched and is not imported.

Cleaning preserves identifiers, raw payment JSON, numeric ledger/payment fields, UTC calendar features, transfer counts and payment flags. Only wholly empty optional columns are removed. `unique_tokens` retains the original meaning: distinct currency symbols; USD pricing separately identifies assets by issuer and currency. There is no one-hot encoding fitted independently per split, no ML training and no change to the team's 80/20 model split.

Rates join on `(floor(TIMESTAMP / 3600) * 3600, issuer:currency)` (or `XRP`). Missing, duplicate, nonfinite or nonpositive prices fail explicitly. There is no backward lookup or zero-value fallback. `delivered_usd` is the recipient-received `DELIVERED_AMOUNT` valued in USD; it is not actual sender spending. `SendMax` is only a spending ceiling and transfer legs are never summed.

Before export, checks enforce unique nonmissing transaction hashes, exact input/output hash coverage, valid transfer keys with no orphan hashes, matching datetime/timestamp, finite amounts, identical ordered feature columns and dtypes across splits, and full training-label coverage. Only `is_suspicious` is imported from training labels; `family`, `difficulty`, `scenario_id` and `cohort_id` are excluded. Validation and test contain no targets. Transactions without transfers retain zero count features.

One final export cell writes all six files only after every split passes validation. It reloads each Parquet and checks its complete contents against the in-memory result; transfers are unchanged. Files are replaced by a successful run; publication of all six files is not an atomic filesystem transaction.

## Fresh-kernel results

| Split | Transactions | Transfers | Cleaned columns | Missing USD |
|---|---:|---:|---:|---:|
| train | 40,131 | 60,557 | 36 | 0 |
| validation | 33,862 | 55,897 | 35 | 0 |
| test | 40,101 | 65,272 | 35 | 0 |

Outputs: `data/processed/{split}_cleaned.parquet` and `data/processed/{split}_transfers.parquet` for train, validation and test.

All notebook cells and six output round trips passed. Eleven synthetic regression tests passed, covering exact hourly and issuer-aware pricing, recipient value, SendMax independence, weekend flags, zero transfer counts, missing/stale rates, wrong issuers, duplicate/missing hashes, orphan transfers, duplicate rates, invalid amounts and training-label coverage/leakage. No data validation failures occurred. Dataset and output files were not staged; no commit or push was made.
