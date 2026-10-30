"""Step 3: compare recent data with the reference using the KS test."""
import json

import pandas as pd
from scipy.stats import ks_2samp

from common import CURRENT, DRIFT_FEATURES, DRIFT_REPORT, REFERENCE

P_VALUE_THRESHOLD = 0.05
MIN_DRIFTED_FEATURES = 2

reference = pd.read_csv(REFERENCE)
current = pd.read_csv(CURRENT)

results = {}
drifted_features = []

for feature in DRIFT_FEATURES:
    ref_values = reference[feature].dropna()
    cur_values = current[feature].dropna()

    statistic, p_value = ks_2samp(ref_values, cur_values)
    drift_detected = bool(p_value < P_VALUE_THRESHOLD)

    results[feature] = {
        "ks_statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 6),
        "drift_detected": drift_detected,
        "reference_mean": round(float(ref_values.mean()), 4),
        "current_mean": round(float(cur_values.mean()), 4),
    }
    if drift_detected:
        drifted_features.append(feature)

overall_drift = len(drifted_features) >= MIN_DRIFTED_FEATURES

report = {
    "method": "two-sample KS test",
    "p_value_threshold": P_VALUE_THRESHOLD,
    "min_drifted_features": MIN_DRIFTED_FEATURES,
    "reference_rows": int(len(reference)),
    "current_rows": int(len(current)),
    "drifted_feature_count": len(drifted_features),
    "drifted_features": drifted_features,
    "overall_drift": overall_drift,
    "features": results,
}

with open(DRIFT_REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
print("\nOVERALL DRIFT:", overall_drift)
if overall_drift:
    print("Action: investigate and retrain if recent labels are available.")
else:
    print("Action: continue monitoring. Retraining is not needed.")
