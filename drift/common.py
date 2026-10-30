"""Shared settings for the drift scripts (Practical 15)."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"

DATASET = PROJECT_ROOT / "cardio_train.csv"
APPROVED_MODEL = PROJECT_ROOT / "cardio_risk_model.pkl"

FEATURES = [
    "age_years", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
]
# Continuous features checked with the KS test
DRIFT_FEATURES = ["age_years", "height", "weight", "ap_hi", "ap_lo"]
TARGET = "cardio"

HISTORICAL = DRIFT_DIR / "historical_labeled.csv"      # features + cardio (used for training)
REFERENCE = DRIFT_DIR / "reference_data.csv"           # features only (baseline for drift)
POOL = DRIFT_DIR / "production_pool.csv"               # pretend "future" labeled data
CURRENT = DRIFT_DIR / "current_data.csv"               # recent batch, features only
CURRENT_LABELED = DRIFT_DIR / "current_labeled_data.csv"
DRIFT_REPORT = DRIFT_DIR / "drift_report.json"
CANDIDATE_MODEL = DRIFT_DIR / "cardio_risk_model_candidate.pkl"
METRICS_FILE = DRIFT_DIR / "retraining_metrics.json"


def load_clean(path=DATASET):
    """Load cardio_train.csv (';' separated) and remove impossible values."""
    df = pd.read_csv(path, sep=";")
    df["age_years"] = (df["age"] / 365.25).round().astype(int)
    df = df[df["ap_hi"].between(80, 250) & df["ap_lo"].between(40, 150)]
    df = df[df["ap_hi"] > df["ap_lo"]]
    df = df[df["height"].between(120, 220) & df["weight"].between(30, 200)]
    return df[FEATURES + [TARGET]].reset_index(drop=True)
