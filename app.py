import logging
import sys
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, g, jsonify, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("cardio-risk-api")

app = Flask(__name__)

MODEL_PATH = Path("cardio_risk_model.pkl")
FEATURES = [
    "age_years", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
]

# Only add NON-sensitive feature names here if input values must be logged.
SAFE_INPUT_FIELDS = set()


def load_model():
    if not MODEL_PATH.exists():
        logger.error("event=model_load_failed model_file=%s reason=file_not_found", MODEL_PATH)
        raise FileNotFoundError(
            "cardio_risk_model.pkl was not found. "
            "Run the training pipeline first."
        )
    model = joblib.load(MODEL_PATH)
    logger.info("event=model_loaded model_file=%s", MODEL_PATH)
    return model


# Try to load the model once at startup so the log shows it (never crash here).
try:
    load_model()
except Exception:
    logger.exception("event=startup_model_check_failed")


@app.before_request
def start_timer():
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    g.start = time.perf_counter()


@app.after_request
def log_request(response):
    duration_ms = (time.perf_counter() - g.start) * 1000
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        g.request_id, request.method, request.path,
        response.status_code, duration_ms,
    )
    response.headers["X-Request-ID"] = g.request_id
    return response


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route not found"}), 404


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "cardiovascular-disease-prediction",
        "version": "v2"
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        logger.warning("event=validation_error reason=missing_or_invalid_json")
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [f for f in FEATURES if f not in data]
    if missing_fields:
        logger.warning(
            "event=validation_error missing_fields=%s error_count=%s",
            missing_fields, len(missing_fields),
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        safe_input = {k: v for k, v in data.items() if k in SAFE_INPUT_FIELDS}
        logger.info(
            "event=prediction_request request_id=%s feature_names=%s safe_input=%s",
            g.request_id, sorted(FEATURES), safe_input,
        )

        sample = pd.DataFrame([{f: data[f] for f in FEATURES}])

        model = load_model()
        prediction_code = int(model.predict(sample)[0])
        probability = float(model.predict_proba(sample)[0][1])
        prediction = "CARDIO_RISK" if prediction_code == 1 else "NO_RISK"

        logger.info(
            "event=prediction_success request_id=%s prediction=%s",
            g.request_id, prediction,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
            "risk_probability": round(probability, 4)
        })

    except Exception:
        logger.exception("event=prediction_failed request_id=%s", g.request_id)
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
