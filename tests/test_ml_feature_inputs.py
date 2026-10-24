from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from ml_feature_inputs import merge_team_features


class FeatureHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "graph.csv"
        self.base = pd.DataFrame({"TRANSACTION_HASH": ["a", "b"], "is_suspicious": [0, 1]}, index=[5, 9])
        self.features = pd.DataFrame({"TRANSACTION_HASH": ["b", "a"], "sender_in_degree": [20, 10]})

    def merge(self):
        self.features.to_csv(self.path, index=False)
        return merge_team_features(self.base, {"graph": self.path})

    def test_shuffled_features_align_on_hash_and_preserve_index(self):
        result, columns, _ = self.merge()
        self.assertEqual(result.index.tolist(), [5, 9])
        self.assertEqual(result.TRANSACTION_HASH.tolist(), ["a", "b"])
        self.assertEqual(result["graph__sender_in_degree"].tolist(), [10, 20])
        self.assertEqual(columns, ["graph__sender_in_degree"])
        self.assertEqual(result.is_suspicious.tolist(), [0, 1])

    def test_no_features_is_unchanged_baseline(self):
        result, columns, sources = merge_team_features(self.base, {"graph": None, "behaviour": None})
        pd.testing.assert_frame_equal(result, self.base)
        self.assertEqual((columns, sources), ([], {}))

    def test_missing_hash_fails(self):
        self.features = self.features.iloc[:1]
        with self.assertRaisesRegex(ValueError, "1 missing"):
            self.merge()

    def test_extra_hash_fails(self):
        self.features.loc[2] = ["extra", 7]
        with self.assertRaisesRegex(ValueError, "1 extra"):
            self.merge()

    def test_duplicate_hash_fails(self):
        self.features.loc[1, "TRANSACTION_HASH"] = "b"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.merge()

    def test_label_metadata_fails(self):
        for name in ["is_suspicious", "family", "difficulty", "scenario_id", "cohort_id", "score"]:
            with self.subTest(name=name):
                self.features[name] = [0, 1]
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    self.merge()
                self.features = self.features.drop(columns=name)

    def test_missing_and_infinite_values_fail(self):
        for value in [None, float("inf")]:
            with self.subTest(value=value):
                self.features["sender_in_degree"] = [value, 1]
                with self.assertRaisesRegex(ValueError, "missing or nonfinite"):
                    self.merge()

    def test_string_identifiers_fail(self):
        self.features["wallet"] = ["rA", "rB"]
        with self.assertRaisesRegex(ValueError, "numeric or boolean"):
            self.merge()

    def test_two_groups_with_same_feature_names_do_not_collide(self):
        self.features.to_csv(self.path, index=False)
        result, columns, _ = merge_team_features(self.base, {"graph": self.path, "behaviour": self.path})
        self.assertEqual(len(columns), 2)
        self.assertEqual(result["graph__sender_in_degree"].tolist(), result["behaviour__sender_in_degree"].tolist())

    def test_explicit_behaviour_missing_is_preserved(self):
        self.features["sender_in_degree"] = [None, 1]
        self.features.to_csv(self.path, index=False)
        result, _, _ = merge_team_features(self.base, {"behaviour": self.path}, allow_missing_groups={"behaviour"})
        self.assertTrue(pd.isna(result.loc[9, "behaviour__sender_in_degree"]))
        self.assertEqual(result.loc[5, "behaviour__sender_in_degree"], 1)

    def test_allowing_missing_never_allows_infinity(self):
        self.features["sender_in_degree"] = [float("inf"), 1]
        self.features.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            merge_team_features(self.base, {"behaviour": self.path}, allow_missing_groups={"behaviour"})

    def test_parquet_features_supported(self):
        path = self.path.with_suffix(".parquet")
        self.features.to_parquet(path, index=False)
        result, _, _ = merge_team_features(self.base, {"graph": path})
        self.assertEqual(result["graph__sender_in_degree"].tolist(), [10, 20])


if __name__ == "__main__":
    unittest.main()
