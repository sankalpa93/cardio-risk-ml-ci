"""Step 5: quality gate. Promote the candidate only if accuracy is high enough."""
import json
import os
import shutil

from common import APPROVED_MODEL, CANDIDATE_MODEL, METRICS_FILE

# Use the same minimum accuracy as your Practical 3 quality gate.
MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY", "0.70"))

if not METRICS_FILE.exists():
    print("No candidate model was trained (no drift or no labels).")
    print("Current approved model is unchanged.")
    raise SystemExit(0)

with open(METRICS_FILE, encoding="utf-8") as f:
    metrics = json.load(f)

accuracy = metrics["candidate_accuracy"]
print("Candidate accuracy:", accuracy)
print("Required accuracy :", MIN_ACCURACY)

if accuracy < MIN_ACCURACY:
    print("QUALITY GATE: FAIL")
    print("Current approved model is unchanged.")
    raise SystemExit(1)

shutil.copy2(CANDIDATE_MODEL, APPROVED_MODEL)
print("QUALITY GATE: PASS")
print("Candidate promoted to:", APPROVED_MODEL)
