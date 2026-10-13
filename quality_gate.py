
import json
import sys

METRICS_PATH = "artifacts/metrics.json"
MIN_ACCURACY = 0.85

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]

print(f"Model accuracy: {accuracy:.4f}")
print(f"Required accuracy: {MIN_ACCURACY:.2f}")

if accuracy < MIN_ACCURACY:
    print("QUALITY GATE FAILED: Accuracy is below the required threshold.")
    sys.exit(1)

print("QUALITY GATE PASSED: Model accuracy meets the requirement.")
