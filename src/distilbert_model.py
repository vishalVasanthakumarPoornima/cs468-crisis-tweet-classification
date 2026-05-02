import os
import json
import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

DATA_PATH = "data/processed/humaid_combined.csv"
LABEL_MAPPING_PATH = "data/processed/label_mapping.json"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/distilbert_humaid"
RESULTS_DIR = "results"


def load_data():
    df = pd.read_csv(DATA_PATH)

    required_columns = ["clean_text", "label_id", "split"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=["clean_text", "label_id", "split"])
    df["label_id"] = df["label_id"].astype(int)

    train_df = df[df["split"] == "train"]
    dev_df = df[df["split"] == "dev"]
    test_df = df[df["split"] == "test"]

    print(f"Train size: {len(train_df)}")
    print(f"Dev size: {len(dev_df)}")
    print(f"Test size: {len(test_df)}")

    return train_df, dev_df, test_df


def load_label_mapping():
    with open(LABEL_MAPPING_PATH, "r") as f:
        label_mapping = json.load(f)

    id_to_label = {v: k for k, v in label_mapping.items()}
    label_to_id = label_mapping

    return label_to_id, id_to_label


def tokenize_data(train_df, dev_df, test_df, tokenizer):
    train_dataset = Dataset.from_pandas(train_df[["clean_text", "label_id"]])
    dev_dataset = Dataset.from_pandas(dev_df[["clean_text", "label_id"]])
    test_dataset = Dataset.from_pandas(test_df[["clean_text", "label_id"]])

    def tokenize(batch):
        return tokenizer(
            batch["clean_text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    dev_dataset = dev_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.rename_column("label_id", "labels")
    dev_dataset = dev_dataset.rename_column("label_id", "labels")
    test_dataset = test_dataset.rename_column("label_id", "labels")

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, dev_dataset, test_dataset


def compute_metrics(pred):
    labels = pred.label_ids
    predictions = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_df, dev_df, test_df = load_data()
    label_to_id, id_to_label = load_label_mapping()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_to_id),
        id2label={int(k): v for k, v in id_to_label.items()},
        label2id=label_to_id
    )

    train_dataset, dev_dataset, test_dataset = tokenize_data(
        train_df,
        dev_df,
        test_df,
        tokenizer
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics
    )   

    print("Starting training...")
    trainer.train()

    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    with open(os.path.join(RESULTS_DIR, "distilbert_test_metrics.json"), "w") as f:
        json.dump(test_results, f, indent=4)

    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=[id_to_label[i] for i in range(len(id_to_label))],
        zero_division=0
    )

    with open(os.path.join(RESULTS_DIR, "distilbert_classification_report.txt"), "w") as f:
        f.write(report)

    print(report)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("DistilBERT baseline training complete.")


if __name__ == "__main__":
    main()