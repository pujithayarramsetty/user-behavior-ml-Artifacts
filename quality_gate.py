import json
import sys

# Load metrics
with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print(f"Model Accuracy: {accuracy}")

# Quality threshold
MIN_ACCURACY = 0.90

if accuracy >= MIN_ACCURACY:
    print("QUALITY GATE PASSED")
    print(f"Accuracy {accuracy} >= {MIN_ACCURACY}")
else:
    print("QUALITY GATE FAILED")
    print(f"Accuracy {accuracy} < {MIN_ACCURACY}")
    sys.exit(1)
