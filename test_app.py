import unittest

from app import app

HIGH_RISK = {
    "age_years": 62, "gender": 1, "height": 160, "weight": 92,
    "ap_hi": 160, "ap_lo": 100, "cholesterol": 3, "gluc": 2,
    "smoke": 0, "alco": 0, "active": 0,
}

LOW_RISK = {
    "age_years": 35, "gender": 1, "height": 170, "weight": 62,
    "ap_hi": 110, "ap_lo": 70, "cholesterol": 1, "gluc": 1,
    "smoke": 0, "alco": 0, "active": 1,
}


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_high_risk_prediction(self):
        response = self.client.post("/predict", json=HIGH_RISK)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "NO_RISK")

    def test_low_risk_prediction(self):
        response = self.client.post("/predict", json=LOW_RISK)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "NO_RISK")

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict", json={"age_years": 50, "gender": 1}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())


if __name__ == "__main__":
    unittest.main()
