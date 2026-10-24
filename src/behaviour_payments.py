"""Extend the organiser's projection with verified actual sender-paid USD."""
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .payments import load_payments
except ImportError:  # Notebook launched from src/.
    from payments import load_payments


def load_behaviour_payments(directory):
    directory = Path(directory)
    if not (directory / 'market_rates.csv').is_file():
        raise ValueError('Reference market rates are required; raw-leg fallback is not allowed.')
    payments = pd.DataFrame(load_payments(directory))
    transfers = pd.read_parquet(directory / 'transfers.parquet')
    if payments.TRANSACTION_HASH.isna().any() or not payments.TRANSACTION_HASH.is_unique:
        raise ValueError('Payment hashes must be nonmissing and unique.')
    required = ['TRANSACTION_HASH', 'TRANSFER_INDEX', 'TIMESTAMP', 'FROM_ADDRESS', 'TO_ADDRESS', 'TOKEN_ADDRESS', 'SYMBOL', 'VALUE']
    if not set(required).issubset(transfers.columns):
        raise ValueError('Missing transfer fields.')
    if transfers[required[:-1]].isna().any().any() or transfers.duplicated(['TRANSACTION_HASH', 'TRANSFER_INDEX']).any():
        raise ValueError('Invalid or duplicate transfer keys/metadata.')
    if set(transfers.TRANSACTION_HASH) != set(payments.TRANSACTION_HASH):
        raise ValueError('Transfer and transaction hash sets differ.')
    ordered = transfers.sort_values(['TRANSACTION_HASH', 'TRANSFER_INDEX'])
    first = ordered.drop_duplicates('TRANSACTION_HASH').set_index('TRANSACTION_HASH').reindex(payments.TRANSACTION_HASH)
    last = ordered.drop_duplicates('TRANSACTION_HASH', keep='last').set_index('TRANSACTION_HASH').reindex(payments.TRANSACTION_HASH)
    if not np.array_equal(first.FROM_ADDRESS.to_numpy(), payments.FROM_ADDRESS.to_numpy()) or not np.array_equal(last.TO_ADDRESS.to_numpy(), payments.TO_ADDRESS.to_numpy()):
        raise ValueError('Transfer endpoints do not match payment endpoints.')
    stamps = payments.set_index('TRANSACTION_HASH').TIMESTAMP
    if not (transfers.TIMESTAMP.to_numpy() == transfers.TRANSACTION_HASH.map(stamps).to_numpy()).all():
        raise ValueError('Transfer timestamps do not match payments.')
    rates = pd.read_csv(directory / 'market_rates.csv')
    if rates[['TIMESTAMP', 'ASSET', 'PRICE_MICRO_USD']].isna().any().any() or rates.duplicated(['TIMESTAMP', 'ASSET']).any():
        raise ValueError('Missing or duplicate reference rates.')
    if not np.isfinite(rates.PRICE_MICRO_USD).all() or (rates.PRICE_MICRO_USD <= 0).any():
        raise ValueError('Reference prices must be positive and finite.')
    def value(legs):
        assets = legs.TOKEN_ADDRESS + ':' + legs.SYMBOL
        assets = assets.where(legs.TOKEN_ADDRESS != '0x1111222211112222111122221111222211112222', 'XRP')
        keys = pd.MultiIndex.from_arrays([legs.TIMESTAMP.astype('int64') // 3600 * 3600, assets])
        prices = rates.set_index(['TIMESTAMP', 'ASSET']).PRICE_MICRO_USD.reindex(keys).to_numpy(dtype=float)
        if np.isnan(prices).any():
            raise ValueError('Missing reference rate for executed asset/hour.')
        amounts = pd.to_numeric(legs.VALUE, errors='raise').to_numpy(dtype=float)
        if np.isinf(amounts).any() or (amounts < 0).any():
            raise ValueError('Invalid executed amount.')
        return amounts * (prices / 1_000_000)
    payments['PAID_USD'] = value(first)
    payments['RECEIVED_USD'] = payments.VALUE.astype(float)
    final_value = value(last)
    known = np.isfinite(final_value)
    if not np.allclose(final_value[known], payments.RECEIVED_USD.to_numpy()[known], rtol=1e-9, atol=1e-6):
        raise ValueError('Final transfer value disagrees with delivered payment value.')
    return payments


def connect_cleaned_payments(cleaned, directory):
    """Connect Person 1's transaction-level output, without copying its labels."""
    needed = ['TRANSACTION_HASH', 'ACCOUNT', 'TIMESTAMP', 'EXTRA_TRANSACTION_FIELDS', 'DELIVERED_AMOUNT']
    if not set(needed).issubset(cleaned.columns):
        raise ValueError('Cleaned output lacks required payment fields.')
    if cleaned.TRANSACTION_HASH.isna().any() or not cleaned.TRANSACTION_HASH.is_unique:
        raise ValueError('Cleaned hashes must be nonmissing and unique.')
    payments = load_behaviour_payments(directory)
    if set(cleaned.TRANSACTION_HASH) != set(payments.TRANSACTION_HASH):
        raise ValueError('Cleaned/raw hash sets differ: confirm any filtering before using history.')
    original = pd.read_parquet(Path(directory) / 'transactions.parquet', columns=needed).set_index('TRANSACTION_HASH')
    aligned = cleaned.set_index('TRANSACTION_HASH').reindex(original.index)
    for column in needed[1:]:
        if not aligned[column].eq(original[column]).all():
            raise ValueError(f'Cleaned {column} differs from raw transactions; review cleaning semantics.')
    counts = ['transfer_count', 'unique_senders', 'unique_receivers', 'unique_tokens']
    if not set(counts).issubset(cleaned.columns):
        raise ValueError('Expected Person 1 transfer-count features.')
    extras = [c for c in ['hour', 'day_of_week', 'is_weekend'] if c in cleaned]
    if 'delivered_usd' in cleaned:
        expected = payments.set_index('TRANSACTION_HASH').RECEIVED_USD.reindex(cleaned.TRANSACTION_HASH).to_numpy()
        if not np.allclose(cleaned.delivered_usd.to_numpy(dtype=float), expected, rtol=1e-9, atol=1e-6, equal_nan=True):
            raise ValueError('Person 1 delivered_usd differs from containing-hour reference valuation.')
    result = cleaned[['TRANSACTION_HASH'] + counts + extras].merge(payments, on='TRANSACTION_HASH', validate='one_to_one')
    return result
