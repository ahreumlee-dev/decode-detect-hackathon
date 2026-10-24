"""Strictly historical, transaction-level behaviour features (no labels)."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    'sender_tx_count_1h', 'sender_tx_count_24h', 'seconds_since_previous_tx',
    'sender_unique_recipients_24h', 'sender_outflow_usd_24h', 'sender_inflow_usd_24h',
]


@dataclass(frozen=True)
class Columns:
    transaction_hash: str = 'TRANSACTION_HASH'
    timestamp: str = 'TIMESTAMP'
    sender: str = 'FROM_ADDRESS'
    recipient: str = 'TO_ADDRESS'
    paid_usd: str = 'PAID_USD'
    received_usd: str = 'RECEIVED_USD'


def _validated(frame: pd.DataFrame, columns: Columns) -> pd.DataFrame:
    names = list(asdict(columns).values())
    if not frame.columns.is_unique or len(set(names)) != len(names):
        raise ValueError('Input columns and mapped field names must be unique.')
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    data = frame[names].copy().reset_index(drop=True)
    data.columns = list(asdict(columns))
    for field in ['transaction_hash', 'sender', 'recipient']:
        if not data[field].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f'{field} must contain nonempty strings with no missing values.')
    if data.transaction_hash.duplicated().any():
        raise ValueError('Expected one row per TRANSACTION_HASH; duplicate hashes found.')
    # Explicit integer Unix seconds: never guess milliseconds, strings or timezone.
    try:
        times = pd.to_numeric(data.timestamp, errors='raise').to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError('timestamp must be integer Unix seconds.') from exc
    if (not np.isfinite(times).all() or (times < 0).any()
            or (times > 10_000_000_000).any() or (times != np.floor(times)).any()
            or data.timestamp.map(lambda x: isinstance(x, (bool, np.bool_))).any()):
        raise ValueError('timestamp must be nonmissing integer Unix seconds, not milliseconds.')
    data['timestamp'] = times.astype(np.int64)
    for field in ['paid_usd', 'received_usd']:
        try:
            amounts = pd.to_numeric(data[field], errors='raise').to_numpy(dtype=float, na_value=np.nan)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{field} must contain numeric USD amounts or missing values.') from exc
        if np.isinf(amounts).any() or (amounts < 0).any() or data[field].map(lambda x: isinstance(x, (bool, np.bool_))).any():
            raise ValueError(f'{field} must contain nonnegative finite USD amounts or missing values.')
        data[field] = amounts
    return data


def _window_sum(times, amounts, query_times):
    """Unknown amount in a nonempty window => NaN; empty window => zero."""
    left = np.searchsorted(times, query_times - 86400, side='left')
    right = np.searchsorted(times, query_times, side='left')
    known = ~np.isnan(amounts)
    # Missing values are excluded only from arithmetic; a separate count ensures
    # they always make the resulting window unknown, never a zero-valued payment.
    values = np.zeros(len(amounts), dtype=float)
    values[known] = amounts[known]
    with np.errstate(over='raise', invalid='raise'):
        sums = np.concatenate(([0.0], np.cumsum(values)))
        result = sums[right] - sums[left]
    unknown = np.concatenate(([0], np.cumsum(~known)))
    result[unknown[right] > unknown[left]] = np.nan
    return result


def build_features(frame: pd.DataFrame, columns: Columns = Columns()) -> pd.DataFrame:
    """Use only this frame's history. Preserve row order and never mutate input.

    Windows are [t-width, t). All rows at t are excluded, irrespective of index.
    Payment principal only: fees and intermediate liquidity-provider legs must
    already be excluded from the two supplied USD fields.
    """
    data = _validated(frame, columns)
    n = len(data)
    output = pd.DataFrame({'TRANSACTION_HASH': data.transaction_hash.to_numpy(copy=True)})
    for name in FEATURES:
        output[name] = np.zeros(n, dtype=np.int64 if name in (FEATURES[0], FEATURES[1], FEATURES[3]) else float)
    output['seconds_since_previous_tx'] = np.nan
    ordered = data.sort_values('timestamp', kind='stable')
    incoming = {wallet: group for wallet, group in ordered.groupby('recipient', sort=False)}
    for wallet, group in ordered.groupby('sender', sort=False):
        indexes = group.index.to_numpy()
        times = group.timestamp.to_numpy()
        before = np.searchsorted(times, times, side='left')
        lower24 = np.searchsorted(times, times - 86400, side='left')
        output.loc[indexes, FEATURES[0]] = before - np.searchsorted(times, times - 3600, side='left')
        output.loc[indexes, FEATURES[1]] = before - lower24
        gaps = np.full(len(times), np.nan)
        observed = before > 0
        gaps[observed] = times[observed] - times[before[observed] - 1]
        output.loc[indexes, FEATURES[2]] = gaps
        # Each outgoing payment enters/leaves the counter once, including ties.
        recipients = group.recipient.to_numpy()
        counts = Counter()
        added = removed = 0
        unique = np.zeros(len(times), dtype=np.int64)
        for i in range(len(times)):
            while added < before[i]:
                counts[recipients[added]] += 1
                added += 1
            while removed < lower24[i]:
                recipient = recipients[removed]
                counts[recipient] -= 1
                if counts[recipient] == 0:
                    del counts[recipient]
                removed += 1
            unique[i] = len(counts)
        output.loc[indexes, FEATURES[3]] = unique
        output.loc[indexes, FEATURES[4]] = _window_sum(times, group.paid_usd.to_numpy(), times)
        received = incoming.get(wallet)
        if received is not None:
            output.loc[indexes, FEATURES[5]] = _window_sum(
                received.timestamp.to_numpy(), received.received_usd.to_numpy(), times)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='One cleaned split, CSV or Parquet.')
    parser.add_argument('--output', type=Path, required=True, help='New CSV or Parquet output path.')
    parser.add_argument('--column-map', type=Path, help='JSON mapping Columns field names to actual input names.')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error('Output must be a new file and must not replace the input.')
    if args.input.suffix.lower() not in ('.csv', '.parquet') or args.output.suffix.lower() not in ('.csv', '.parquet'):
        parser.error('Use .csv or .parquet files.')
    columns = Columns(**json.loads(args.column_map.read_text())) if args.column_map else Columns()
    if args.input.suffix.lower() == '.csv':
        frame = pd.read_csv(args.input, dtype={columns.transaction_hash: str, columns.sender: str, columns.recipient: str})
    else:
        frame = pd.read_parquet(args.input)
    result = build_features(frame, columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == '.csv':
        result.to_csv(args.output, index=False)
    else:
        result.to_parquet(args.output, index=False)
    print(f'Wrote {len(result)} unique transaction rows to {args.output}')


if __name__ == '__main__':
    main()
