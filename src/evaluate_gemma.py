import pandas as pd
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results/gemma_few_shot.csv")

y_true = df["class_label"]
y_pred = df["prediction"]

# metrics
metrics = {
    "accuracy": accuracy_score(y_true, y_pred)
}

with open("results/gemma_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# classification report
report = classification_report(y_true, y_pred)

with open("results/gemma_classification_report.txt", "w") as f:
    f.write(report)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d")
plt.savefig("results/gemma_confusion_matrix.png")

print("Saved all evaluation files.")