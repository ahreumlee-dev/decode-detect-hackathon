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

## Get the repaired cleaning data

The reproducible cleaning pipeline and the six ready-to-use Parquet files are on `feature/data-cleaning`. To bring them into your own feature branch, first commit or stash your local work, then run:

```sh
git fetch origin
git merge origin/feature/data-cleaning
```

For a fresh checkout:

```sh
git clone --branch feature/data-cleaning https://github.com/ahreumlee-dev/decode-detect-hackathon.git
cd decode-detect-hackathon
```

Use `data/processed/{train,validation,test}_cleaned.parquet` and the corresponding `{split}_transfers.parquet`. The raw files remain under `data/raw/{split}`. Read paths relative to the repository root, not relative to your notebook's directory.

Example from a notebook inside this repository:

```python
from pathlib import Path
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "data/processed/train_cleaned.parquet").is_file())
train = pd.read_parquet(ROOT / "data/processed/train_cleaned.parquet")
validation = pd.read_parquet(ROOT / "data/processed/validation_cleaned.parquet")
test = pd.read_parquet(ROOT / "data/processed/test_cleaned.parquet")
```

The ready-to-use exports do not require rerunning cleaning. To regenerate them in a new kernel, see [the cleaning reproduction guide](docs/data_cleaning.md). Only train contains `is_suspicious`; join downstream features on `TRANSACTION_HASH`. `delivered_usd` measures recipient-received value, not sender spending.
