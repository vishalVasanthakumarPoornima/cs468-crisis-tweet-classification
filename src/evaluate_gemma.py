import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

df = pd.read_csv("results/gemma_predictions.csv")

def clean_label(label):
    if not isinstance(label, str):
        return "unknown"
    return label.strip().lower()

df["true_label"] = df["class_label"].apply(clean_label)
df["predicted_label"] = df["prediction"].apply(clean_label)

y_true = df["true_label"]
y_pred = df["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)

weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)

macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

metrics = {
    "accuracy": accuracy,
    "weighted_f1": weighted_f1,
    "macro_f1": macro_f1
}

print("\nGemma Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

with open("results/gemma_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

report = classification_report(y_true, y_pred, zero_division=0)

with open("results/gemma_classification_report.txt", "w") as f:
    f.write(report)