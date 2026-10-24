import pandas as pd
import pytest
from src import behaviour_payments as module


@pytest.fixture
def payment_files(tmp_path, monkeypatch):
    pd.DataFrame({'TIMESTAMP': [0, 0], 'ASSET': ['issuer:USD', 'other:EUR'], 'PRICE_MICRO_USD': [1000000, 2000000]}).to_csv(tmp_path/'market_rates.csv', index=False)
    legs = pd.DataFrame([
        ['t', 0, 1, 'A', 'LP', 'issuer', 'USD', 10.],
        ['t', 1, 1, 'LP', 'B', 'other', 'EUR', 4.],
    ], columns=['TRANSACTION_HASH', 'TRANSFER_INDEX', 'TIMESTAMP', 'FROM_ADDRESS', 'TO_ADDRESS', 'TOKEN_ADDRESS', 'SYMBOL', 'VALUE'])
    legs.to_parquet(tmp_path/'transfers.parquet')
    monkeypatch.setattr(module, 'load_payments', lambda _: [{'TRANSACTION_HASH':'t','TIMESTAMP':1,'FROM_ADDRESS':'A','TO_ADDRESS':'B','VALUE':8.}])
    return tmp_path


def test_conversion_uses_first_leg_not_sum_or_delivered(payment_files):
    result = module.load_behaviour_payments(payment_files)
    assert len(result) == 1
    assert result.PAID_USD.iloc[0] == 10.
    assert result.RECEIVED_USD.iloc[0] == 8.


def test_missing_rate_rejected(payment_files):
    rates = pd.read_csv(payment_files/'market_rates.csv').iloc[1:]
    rates.to_csv(payment_files/'market_rates.csv', index=False)
    with pytest.raises(ValueError, match='Missing reference rate'):
        module.load_behaviour_payments(payment_files)


def test_duplicate_legs_rejected(payment_files):
    legs=pd.read_parquet(payment_files/'transfers.parquet')
    pd.concat([legs, legs.iloc[:1]]).to_parquet(payment_files/'transfers.parquet')
    with pytest.raises(ValueError, match='duplicate transfer'):
        module.load_behaviour_payments(payment_files)


def test_absent_rates_never_falls_back(tmp_path):
    with pytest.raises(ValueError, match='Reference market rates'):
        module.load_behaviour_payments(tmp_path)


def test_connect_cleaned_keeps_counts_and_excludes_labels(payment_files):
    clean = pd.DataFrame([{'TRANSACTION_HASH':'t','ACCOUNT':'A','TIMESTAMP':1,
        'EXTRA_TRANSACTION_FIELDS':'{}','DELIVERED_AMOUNT':'"1"', 'is_suspicious':1,
        'scenario_id':'private','transfer_count':2,'unique_senders':2,'unique_receivers':2,'unique_tokens':2}])
    clean.to_parquet(payment_files/'transactions.parquet', index=False)
    result = module.connect_cleaned_payments(clean, payment_files)
    assert result.transfer_count.iloc[0] == 2
    assert 'is_suspicious' not in result and 'scenario_id' not in result
    altered = clean.copy()
    altered.loc[0, 'ACCOUNT'] = 'different'
    with pytest.raises(ValueError, match='ACCOUNT differs'):
        module.connect_cleaned_payments(altered, payment_files)
    with pytest.raises(ValueError, match='hash sets differ'):
        module.connect_cleaned_payments(clean.iloc[:0], payment_files)
