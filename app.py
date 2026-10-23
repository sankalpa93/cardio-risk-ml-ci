import logging
import sys
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, g, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("cardio-risk-api")

# ---------- Prometheus metrics (defined once, at module level) ----------
HTTP_REQUESTS = Counter(
    "cardio_api_http_requests_total",
    "Total HTTP requests received by the Cardio Risk Prediction API",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "cardio_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)
PREDICTIONS = Counter(
    "cardio_api_predictions_total",
    "Total Cardio Risk prediction attempts",
    ["result"],
)
MODEL_LOADED = Gauge(
    "cardio_api_model_loaded",
    "Whether the Cardio Risk model is loaded (1 = loaded, 0 = not loaded)",
)

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
        MODEL_LOADED.set(0)
        logger.error("event=model_load_failed model_file=%s reason=file_not_found", MODEL_PATH)
        raise FileNotFoundError(
            "cardio_risk_model.pkl was not found. "
            "Run the training pipeline first."
        )
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED.set(1)
    logger.info("event=model_loaded model_file=%s", MODEL_PATH)
    return model


# Try to load the model once at startup so the gauge and log show it (never crash here).
try:
    load_model()
except Exception:
    MODEL_LOADED.set(0)
    logger.exception("event=startup_model_check_failed")


@app.before_request
def start_timer():
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    g.start = time.perf_counter()


@app.after_request
def log_and_measure(response):
    duration = time.perf_counter() - g.start

    # Metrics: do not count Prometheus' own scrapes of /metrics.
    # Unknown URLs are grouped as "unmatched" to keep labels small and safe.
    if request.path != "/metrics":
        route = request.url_rule.rule if request.url_rule else "unmatched"
        HTTP_REQUESTS.labels(
            method=request.method, path=route, status=str(response.status_code)
        ).inc()
        HTTP_LATENCY.labels(method=request.method, path=route).observe(duration)

    # Practical-11 logging (unchanged)
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        g.request_id, request.method, request.path,
        response.status_code, duration * 1000,
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


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


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

        PREDICTIONS.labels(result="success").inc()
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
        PREDICTIONS.labels(result="error").inc()
        logger.exception("event=prediction_failed request_id=%s", g.request_id)
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
