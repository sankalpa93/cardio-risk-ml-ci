import json
import sys

MIN_ACCURACY = 0.70

with open("metrics.json") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]
print(f"Accuracy: {accuracy} (minimum required: {MIN_ACCURACY})")

if accuracy < MIN_ACCURACY:
    print("QUALITY GATE FAILED: model accuracy is below the threshold.")
    sys.exit(1)

print("QUALITY GATE PASSED")
