import os
import json
import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

DATA_PATH = "data/processed/humaid_combined.csv"
LABEL_MAPPING_PATH = "data/processed/label_mapping.json"
MODEL_PATH = "models/distilbert_humaid"
RESULTS_DIR = "results"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["split"] == "test"]
    df = df.dropna(subset=["clean_text", "label_id"])
    df["label_id"] = df["label_id"].astype(int)
    return df


def load_label_mapping():
    with open(LABEL_MAPPING_PATH, "r") as f:
        label_mapping = json.load(f)

    id_to_label = {v: k for k, v in label_mapping.items()}
    return id_to_label


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data()
    id_to_label = load_label_mapping()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    dataset = Dataset.from_pandas(df[["clean_text", "label_id"]])

    def tokenize(batch):
        return tokenizer(batch["clean_text"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label_id", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    trainer = Trainer(model=model)

    predictions = trainer.predict(dataset)

    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    prediction_df = pd.DataFrame({
        "true_label": [id_to_label[i] for i in y_true],
        "predicted_label": [id_to_label[i] for i in y_pred]
    })

    prediction_df.to_csv(os.path.join(RESULTS_DIR, "distilbert_predictions.csv"), index=False)

    print("Saved distilbert_predictions.csv")


if __name__ == "__main__":
    main()
