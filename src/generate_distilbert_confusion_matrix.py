from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


RESULTS_DIR = Path("results")
PREDICTIONS_PATH = RESULTS_DIR / "distilbert_predictions.csv"
OUTPUT_PATH = RESULTS_DIR / "distilbert_confusion_matrix.png"


def main():
    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = ["true_label", "predicted_label"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    labels = sorted(y_true.unique())

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", values_format="d")

    plt.title("DistilBERT Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()

    print(f"Saved confusion matrix to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()