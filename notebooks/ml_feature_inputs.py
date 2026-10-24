"""Strict, label-free transaction feature inputs for the ML notebook."""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

KEY = "TRANSACTION_HASH"
FORBIDDEN = {"is_suspicious", "family", "difficulty", "scenario_id", "cohort_id",
             "label", "target", "score", "prediction", "predictions"}


def merge_team_features(transactions, feature_files, *, allow_missing_groups=()):
    """Require exact hash coverage; preserve transaction order and prefix features."""
    if transactions[KEY].isna().any() or not transactions[KEY].is_unique:
        raise ValueError("Base transaction hashes must be unique and nonmissing")
    result = transactions.copy()
    added = []
    sources = {}
    for group, filename in feature_files.items():
        if filename is None:
            continue
        if group not in {"graph", "behaviour"}:
            raise ValueError(f"Unsupported feature group: {group}")
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"{group}: feature file not found: {path}")
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            raise ValueError(f"{group}: use CSV or Parquet")
        if not frame.columns.is_unique or KEY not in frame:
            raise ValueError(f"{group}: unique column names and {KEY} are required")
        hashes = frame[KEY]
        if hashes.isna().any() or hashes.astype(str).str.strip().eq("").any() or not hashes.is_unique:
            raise ValueError(f"{group}: missing, blank or duplicate transaction hashes")
        expected, actual = set(transactions[KEY]), set(hashes)
        if expected != actual:
            raise ValueError(f"{group}: hash coverage differs: {len(expected - actual)} missing, {len(actual - expected)} extra")
        columns = frame.columns.drop(KEY).tolist()
        if not columns:
            raise ValueError(f"{group}: no features supplied")
        for column in columns:
            if column.lower() in FORBIDDEN:
                raise ValueError(f"{group}: forbidden label/metadata/prediction column: {column}")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column):
                raise ValueError(f"{group}: invalid feature name: {column}")
            if not (is_numeric_dtype(frame[column]) or is_bool_dtype(frame[column])):
                raise ValueError(f"{group}: feature must be numeric or boolean: {column}")
            if ((frame[column].isna().any() and group not in allow_missing_groups)
                    or np.isinf(frame[column].to_numpy(dtype=float, na_value=np.nan)).any()):
                raise ValueError(f"{group}: missing or nonfinite feature values: {column}")
        rename = {c: f"{group}__{c}" for c in columns}
        if set(rename.values()).intersection(result.columns):
            raise ValueError(f"{group}: prefixed feature columns collide with existing columns")
        original_hashes = result[KEY].tolist()
        result = result.merge(frame.rename(columns=rename), on=KEY, how="left", validate="one_to_one", sort=False)
        if result[KEY].tolist() != original_hashes:
            raise ValueError(f"{group}: merge changed transaction order")
        added.extend(rename.values())
        sources[group] = str(path.resolve())
    # Keep caller indices for the exact existing train/holdout assignment.
    result.index = transactions.index
    return result, added, sources
