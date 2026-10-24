"""Regression checks for the saved notebook, using temporary synthetic inputs."""
from pathlib import Path
import json
import tempfile
import unittest

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class CleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = nbformat.read(ROOT / "notebooks/01_data_cleaning.ipynb", as_version=4)
        namespace = {}
        for cell in notebook.cells[1:3]:
            exec(compile(cell.source, "01_data_cleaning.ipynb", "exec"), namespace)
        cls.process = staticmethod(namespace["process_dataset"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.tx = pd.DataFrame({
            "TRANSACTION_HASH": ["a", "b"], "ACCOUNT": ["sender", "sender"],
            "TIMESTAMP": [1788221144, 1788224400],
            "DATETIME": pd.to_datetime([1788221144, 1788224400], unit="s", utc=True),
            "LEDGER_INDEX": [1, 2], "TRANSACTION_INDEX": [0, 0], "SOURCE_FEE": [10, 10],
            "SEQUENCE": [1, 2], "FLAGS": [0, 0], "LAST_LEDGER_SEQUENCE": [21, 22],
            "DELIVERED_AMOUNT": [json.dumps({"issuer": "issuer-A", "currency": "USD", "value": "3"}), json.dumps("2000000")],
            "EXTRA_TRANSACTION_FIELDS": [json.dumps({"SendMax": "999999999", "Paths": [[{}]]}), "{}"],
        })
        self.tr = pd.DataFrame({"TRANSACTION_HASH": ["a", "a"], "TRANSFER_INDEX": [0, 1],
                                "FROM_ADDRESS": ["sender", "provider"], "TO_ADDRESS": ["provider", "recipient"],
                                "SYMBOL": ["XRP", "USD"]})
        self.rates = pd.DataFrame({"TIMESTAMP": [1788220800, 1788220800, 1788224400, 1788220800],
                                  "ASSET": ["issuer-A:USD", "issuer-B:USD", "XRP", "XRP"],
                                  "PRICE_MICRO_USD": [2000000, 9000000, 4000000, 1000000]})
        self.labels = pd.DataFrame({"TRANSACTION_HASH": ["a", "b"], "is_suspicious": [0, 1], "family": ["private", "private"]})

    def run_cleaning(self, labels=False):
        self.tx.to_parquet(self.folder / "transactions.parquet", index=False)
        self.tr.to_parquet(self.folder / "transfers.parquet", index=False)
        self.rates.to_csv(self.folder / "market_rates.csv", index=False)
        self.labels.to_csv(self.folder / "labels.csv", index=False)
        return self.process(self.folder, include_labels=labels)

    def test_containing_hour_issuer_and_recipient_value(self):
        tx, transfers = self.run_cleaning()
        self.assertEqual(tx.delivered_usd.tolist(), [6.0, 8.0])
        self.assertEqual(tx.transfer_count.tolist(), [2, 0])
        self.assertEqual(tx.has_sendmax.tolist(), [1, 0])
        self.assertEqual(tx.has_paths.tolist(), [1, 0])
        self.assertEqual(tx.is_weekend.tolist(), [0, 0])
        self.assertNotIn("is_suspicious", tx)
        pd.testing.assert_frame_equal(transfers, self.tr)

    def test_weekend(self):
        self.tx["TIMESTAMP"] = [1788652800, 1788652800]  # Sunday UTC
        self.tx["DATETIME"] = pd.to_datetime(self.tx.TIMESTAMP, unit="s", utc=True)
        self.rates = self.rates.iloc[[0, 2]].copy()
        self.rates["TIMESTAMP"] = 1788652800
        tx, _ = self.run_cleaning()
        self.assertEqual(tx.is_weekend.tolist(), [1, 1])

    def test_missing_current_hour_does_not_use_stale_price(self):
        self.rates = self.rates.drop(index=2)  # Earlier XRP rate remains.
        with self.assertRaisesRegex(ValueError, "missing containing-hour rates"):
            self.run_cleaning()

    def test_wrong_issuer_does_not_match_same_currency(self):
        self.rates = self.rates.drop(index=0)  # Another issuer's USD remains.
        with self.assertRaisesRegex(ValueError, "missing containing-hour rates"):
            self.run_cleaning()

    def test_duplicate_hash(self):
        self.tx.loc[1, "TRANSACTION_HASH"] = "a"
        with self.assertRaisesRegex(ValueError, "duplicate transaction hashes"):
            self.run_cleaning()

    def test_missing_hash(self):
        self.tx.loc[1, "TRANSACTION_HASH"] = None
        with self.assertRaisesRegex(ValueError, "missing, blank"):
            self.run_cleaning()

    def test_orphan_transfer(self):
        self.tr.loc[0, "TRANSACTION_HASH"] = "orphan"
        with self.assertRaisesRegex(ValueError, "orphan transfer"):
            self.run_cleaning()

    def test_duplicate_rate(self):
        self.rates = pd.concat([self.rates, self.rates.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate hourly"):
            self.run_cleaning()

    def test_invalid_amount(self):
        self.tx.loc[1, "DELIVERED_AMOUNT"] = json.dumps("NaN")
        with self.assertRaisesRegex(ValueError, "invalid delivered amounts"):
            self.run_cleaning()

    def test_only_target_from_labels(self):
        tx, _ = self.run_cleaning(labels=True)
        self.assertEqual(tx.is_suspicious.tolist(), [0, 1])
        self.assertNotIn("family", tx)

    def test_incomplete_label_coverage(self):
        self.labels = self.labels.iloc[:1]
        with self.assertRaisesRegex(ValueError, "labels do not cover exactly"):
            self.run_cleaning(labels=True)


if __name__ == "__main__":
    unittest.main()
