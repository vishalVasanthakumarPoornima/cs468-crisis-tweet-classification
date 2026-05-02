import os
import pandas as pd
import json

RAW_DATA_PATH = "data/raw/events"
PROCESSED_PATH = "data/processed"

def load_all_data():
    all_data = []

    for event_folder in os.listdir(RAW_DATA_PATH):
        event_path = os.path.join(RAW_DATA_PATH, event_folder)

        if not os.path.isdir(event_path):
            continue

        for file in os.listdir(event_path):
            if file.endswith(".tsv"):
                split = "train" if "train" in file else "dev" if "dev" in file else "test"

                file_path = os.path.join(event_path, file)

                df = pd.read_csv(file_path, sep="\t")

                df["event_name"] = event_folder
                df["split"] = split

                all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def clean_text(text):
    text = str(text)

    # remove URLs
    text = pd.Series(text).str.replace(r"http\S+", "", regex=True).iloc[0]

    # remove mentions
    text = pd.Series(text).str.replace(r"@\w+", "", regex=True).iloc[0]

    # remove extra whitespace
    text = text.strip()

    return text


def preprocess():
    print("Loading data...")
    df = load_all_data()

    print("Cleaning text...")
    df["clean_text"] = df["tweet_text"].apply(clean_text)

    print("Encoding labels...")
    labels = sorted(df["class_label"].unique())
    label_mapping = {label: i for i, label in enumerate(labels)}

    df["label_id"] = df["class_label"].map(label_mapping)

    os.makedirs(PROCESSED_PATH, exist_ok=True)

    print("Saving dataset...")
    df.to_csv(os.path.join(PROCESSED_PATH, "humaid_combined.csv"), index=False)

    print("Saving label mapping...")
    with open(os.path.join(PROCESSED_PATH, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=4)

    print("Saving class distribution...")
    class_counts = df["class_label"].value_counts()
    class_counts.to_csv("results/class_distribution.csv")

    print("Done!")


if __name__ == "__main__":
    preprocess()