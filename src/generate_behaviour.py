"""Generate separate label-free train/test tables for role 04; consumed by role 03."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

from .behaviour_features import FEATURES, build_features
from .behaviour_payments import connect_cleaned_payments


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def generate(cleaned_path, raw_directory, output_path):
    cleaned_path, output_path = Path(cleaned_path), Path(output_path)
    if not cleaned_path.exists():
        raise FileNotFoundError(f'{cleaned_path} missing. Run python notebooks/run_data_cleaning.py first.')
    if cleaned_path.resolve() == output_path.resolve():
        raise ValueError('Output cannot overwrite cleaned input.')
    payments = connect_cleaned_payments(pd.read_parquet(cleaned_path), raw_directory)
    features = build_features(payments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    manifest = {'cleaned_sha256': digest(cleaned_path), 'features_sha256': digest(output_path), 'rows': len(features)}
    output_path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'04: wrote {len(features)} rows to {output_path}')
    return features


def load_for_ml(path, cleaned_path, expected_hashes):
    path = Path(path)
    manifest_path = path.with_suffix('.manifest.json')
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError('Run notebooks/04_behaviour_features.ipynb (or python -m src.generate_behaviour) before 03 ML.')
    manifest = json.loads(manifest_path.read_text())
    if manifest['cleaned_sha256'] != digest(cleaned_path) or manifest['features_sha256'] != digest(path):
        raise ValueError('Behaviour output is stale or modified. Rerun 04 before 03 ML.')
    frame = pd.read_parquet(path)
    if list(frame.columns) != ['TRANSACTION_HASH'] + FEATURES:
        raise ValueError('Expected hash plus exactly six behaviour fields, no labels.')
    if frame.TRANSACTION_HASH.isna().any() or not frame.TRANSACTION_HASH.is_unique or set(frame.TRANSACTION_HASH) != set(expected_hashes):
        raise ValueError('Behaviour table hashes do not match model input.')
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=Path('data/raw/train'))
    parser.add_argument('--test', type=Path, default=Path('data/raw/test'))
    parser.add_argument('--outputs', type=Path, default=Path('outputs'))
    parser.add_argument('--cleaned-dir', type=Path, default=Path('data/processed'))
    args = parser.parse_args()
    for split, raw in [('train', args.train), ('test', args.test)]:
        generate(args.cleaned_dir/f'{split}_cleaned.parquet', raw, args.outputs/f'{split}_behaviour_features.parquet')


if __name__ == '__main__':
    main()
