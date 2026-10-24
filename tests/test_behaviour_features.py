import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.behaviour_features import Columns, FEATURES, build_features


def frame(rows):
    return pd.DataFrame(rows, columns=['TRANSACTION_HASH', 'TIMESTAMP', 'FROM_ADDRESS', 'TO_ADDRESS', 'PAID_USD', 'RECEIVED_USD'])


def oracle(data):
    rows = []
    for row in data.itertuples(index=False):
        earlier = data[(data.TIMESTAMP >= row.TIMESTAMP - 86400) & (data.TIMESTAMP < row.TIMESTAMP)]
        outgoing = earlier[earlier.FROM_ADDRESS == row.FROM_ADDRESS]
        incoming = earlier[earlier.TO_ADDRESS == row.FROM_ADDRESS]
        past = data[(data.FROM_ADDRESS == row.FROM_ADDRESS) & (data.TIMESTAMP < row.TIMESTAMP)]
        def total(values):
            return np.nan if values.isna().any() else values.sum()
        rows.append([row.TRANSACTION_HASH, int((outgoing.TIMESTAMP >= row.TIMESTAMP - 3600).sum()),
                     len(outgoing), np.nan if past.empty else row.TIMESTAMP - past.TIMESTAMP.max(),
                     outgoing.TO_ADDRESS.nunique(), total(outgoing.PAID_USD), total(incoming.RECEIVED_USD)])
    return pd.DataFrame(rows, columns=['TRANSACTION_HASH'] + FEATURES)


def test_boundaries_ties_first_transactions_and_separate_wallets():
    data = frame([
        ('a', 0, 'A', 'B', 10., 9.), ('b', 0, 'A', 'C', 20., 18.),
        ('c', 1, 'B', 'A', 7., 6.), ('d', 3600, 'A', 'C', 30., 28.),
        ('e', 3600, 'A', 'D', 40., 37.), ('f', 86400, 'A', 'B', 50., 48.),
        ('g', 86401, 'A', 'B', 60., 57.), ('h', 172802, 'A', 'B', 70., 65.),
        ('i', 3600, 'Z', 'A', 100., 90.),
    ]).sample(frac=1, random_state=7).reset_index(drop=True)
    original = data.copy(deep=True)
    actual = build_features(data)
    assert_frame_equal(actual, oracle(data), check_dtype=False)
    assert_frame_equal(data, original)
    at = actual.set_index('TRANSACTION_HASH')
    assert at.loc['d', 'sender_tx_count_1h'] == 2
    assert at.loc['e', 'sender_tx_count_1h'] == 2
    assert at.loc['f', 'sender_tx_count_24h'] == 4
    assert at.loc['g', 'sender_tx_count_24h'] == 3
    assert at.loc['h', 'seconds_since_previous_tx'] == 86401
    assert at.loc['d', 'sender_inflow_usd_24h'] == 6
    assert np.isnan(at.loc['i', 'seconds_since_previous_tx'])


def test_missing_amount_is_unknown_not_zero_and_expires():
    data = frame([('a', 0, 'A', 'B', np.nan, np.nan), ('b', 1, 'B', 'A', 5., 4.),
                  ('c', 1, 'A', 'C', 3., 2.), ('d', 86401, 'A', 'C', 1., 1.)])
    actual = build_features(data)
    assert_frame_equal(actual, oracle(data), check_dtype=False)
    assert np.isnan(actual.loc[1, 'sender_inflow_usd_24h'])
    assert np.isnan(actual.loc[2, 'sender_outflow_usd_24h'])
    assert actual.loc[3, 'sender_outflow_usd_24h'] == 3


def test_random_histories_match_independent_brute_force():
    rng = np.random.default_rng(42)
    n = 400
    data = frame(list(zip([f't{i}' for i in range(n)], rng.choice([0, 1, 3600, 3601, 86400, 86401, 172800], n),
                         rng.choice(['A', 'B', 'C', 'D'], n), rng.choice(['A', 'B', 'C', 'D'], n),
                         rng.uniform(0, 1000, n), rng.uniform(0, 1000, n))))
    data.loc[[4, 40], 'PAID_USD'] = np.nan
    data.loc[[9, 90], 'RECEIVED_USD'] = np.nan
    assert_frame_equal(build_features(data), oracle(data), check_dtype=False, atol=1e-8)


@pytest.mark.parametrize('field,value', [('TRANSACTION_HASH', None), ('FROM_ADDRESS', ''),
    ('TO_ADDRESS', None), ('TIMESTAMP', np.nan), ('TIMESTAMP', 1.5), ('TIMESTAMP', 1700000000000),
    ('TIMESTAMP', True), ('PAID_USD', -1), ('RECEIVED_USD', np.inf), ('PAID_USD', 'bad')])
def test_invalid_input(field, value):
    data = frame([('a', 1, 'A', 'B', 5., 4.)]).astype(object)
    data.loc[0, field] = value
    with pytest.raises(ValueError):
        build_features(data)


def test_duplicate_and_missing_columns_rejected():
    data = frame([('a', 1, 'A', 'B', 5., 4.)])
    with pytest.raises(ValueError, match='duplicate hashes'):
        build_features(pd.concat([data, data]))
    with pytest.raises(ValueError, match='Missing required'):
        build_features(data.drop(columns='PAID_USD'))


def test_mapping_labels_empty_and_independent_calls():
    data = frame([('a', 1, 'A', 'B', 5., 4.), ('b', 2, 'A', 'C', 8., 7.)])
    data['label'] = [1, 0]
    mapped = data.rename(columns={'PAID_USD': 'sender_debit', 'RECEIVED_USD': 'delivered'})
    assert_frame_equal(build_features(data), build_features(mapped, Columns(paid_usd='sender_debit', received_usd='delivered')))
    assert list(build_features(data).columns) == ['TRANSACTION_HASH'] + FEATURES
    assert build_features(data.iloc[:0]).empty
    assert build_features(data.iloc[1:]).loc[0, 'sender_tx_count_24h'] == 0


@pytest.mark.parametrize('suffix', ['csv', 'parquet'])
def test_cli_roundtrip_and_no_overwrite(tmp_path, suffix):
    import subprocess
    import sys
    data = frame([('a', 1, 'A', 'B', 5., 4.), ('b', 2, 'A', 'C', 8., 7.)])
    source = tmp_path / f'clean.{suffix}'
    destination = tmp_path / f'features.{suffix}'
    if suffix == 'csv':
        data.to_csv(source, index=False)
    else:
        data.to_parquet(source, index=False)
    original = source.read_bytes()
    command = [sys.executable, '-m', 'src.behaviour_features', '--input', str(source), '--output', str(destination)]
    subprocess.run(command, check=True, capture_output=True)
    result = pd.read_csv(destination) if suffix == 'csv' else pd.read_parquet(destination)
    assert_frame_equal(result, build_features(data))
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert source.read_bytes() == original
