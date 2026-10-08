from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("cardio_risk_model.pkl")
FEATURES = [
    "age_years", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "cardio_risk_model.pkl was not found. "
            "Run the training pipeline first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "cardiovascular-disease-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [f for f in FEATURES if f not in data]
    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([{f: data[f] for f in FEATURES}])

    model = load_model()
    prediction_code = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1])
    prediction = "CARDIO_RISK" if prediction_code == 1 else "NO_RISK"

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code,
        "risk_probability": round(probability, 4)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
