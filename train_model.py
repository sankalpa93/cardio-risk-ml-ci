import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = "cardio_train.csv"
MODEL_PATH = "cardio_risk_model.pkl"
METRICS_PATH = "metrics.json"

FEATURES = [
    "age_years", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
]
TARGET = "cardio"


def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path, sep=";")
    df["age_years"] = (df["age"] / 365.25).round().astype(int)

    # Remove physiologically impossible records (data-entry errors)
    df = df[
        df["ap_hi"].between(70, 250)
        & df["ap_lo"].between(40, 200)
        & (df["ap_hi"] > df["ap_lo"])
        & df["height"].between(120, 220)
        & df["weight"].between(30, 200)
    ]
    return df


def main():
    df = load_and_clean()
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1_score": round(float(f1_score(y_test, pred)), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
    }

    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model trained and saved to", MODEL_PATH)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
