"""Step 4: retrain a CANDIDATE model, but only when drift was detected."""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from common import (CANDIDATE_MODEL, CURRENT_LABELED, DRIFT_REPORT, FEATURES,
                    HISTORICAL, METRICS_FILE, TARGET)

with open(DRIFT_REPORT, encoding="utf-8") as f:
    drift_report = json.load(f)

if not drift_report["overall_drift"]:
    print("No overall drift detected.")
    print("Retraining skipped.")
    raise SystemExit(0)

if not CURRENT_LABELED.exists():
    raise FileNotFoundError(
        "Drift was detected, but current_labeled_data.csv is not available. "
        "Retraining must wait until ground-truth labels exist."
    )

historical = pd.read_csv(HISTORICAL)
recent = pd.read_csv(CURRENT_LABELED)

required = FEATURES + [TARGET]
combined = pd.concat(
    [historical[required], recent[required]], ignore_index=True
).dropna(subset=[TARGET])

X = combined[FEATURES]
y = combined[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

candidate = RandomForestClassifier(
    n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
)
candidate.fit(X_train, y_train)
accuracy = accuracy_score(y_test, candidate.predict(X_test))

joblib.dump(candidate, CANDIDATE_MODEL)

metrics = {
    "candidate_accuracy": round(float(accuracy), 6),
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "historical_rows": int(len(historical)),
    "recent_labeled_rows": int(len(recent)),
}
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
print("Candidate model saved (NOT promoted yet):", CANDIDATE_MODEL)
