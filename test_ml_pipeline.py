import json
import os
import unittest

import joblib

from train_model import FEATURES, load_and_clean


class TestMLPipeline(unittest.TestCase):

    def test_dataset_has_required_columns(self):
        df = load_and_clean()
        for col in FEATURES + ["cardio"]:
            self.assertIn(col, df.columns)

    def test_cleaned_data_is_valid(self):
        df = load_and_clean()
        self.assertGreater(len(df), 60000)
        self.assertEqual(int(df.isna().sum().sum()), 0)
        self.assertTrue((df["ap_hi"] > df["ap_lo"]).all())

    def test_model_and_metrics_files_exist(self):
        self.assertTrue(os.path.exists("cardio_risk_model.pkl"))
        self.assertTrue(os.path.exists("metrics.json"))

    def test_model_predicts_binary_output(self):
        model = joblib.load("cardio_risk_model.pkl")
        df = load_and_clean().head(50)
        preds = model.predict(df[FEATURES])
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_metrics_are_valid(self):
        with open("metrics.json") as f:
            m = json.load(f)
        self.assertTrue(0 <= m["accuracy"] <= 1)
        self.assertTrue(0 <= m["f1_score"] <= 1)


if __name__ == "__main__":
    unittest.main()
