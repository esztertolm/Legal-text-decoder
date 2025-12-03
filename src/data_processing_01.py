import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from config import MODEL_NAME, MAX_LENGTH, SEED, DF_PATH, RAW_DATA_FOLDER_PATH
import json
from collections import Counter
from pathlib import Path
from utils.logger import logger


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts.iloc[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels.iloc[idx]), dtype=torch.long)
        return item
    
def _extract_from_results(results_list):
    out = []
    for res in results_list:
        val = res.get("value") if isinstance(res, dict) else None
        if not isinstance(val, dict):
            continue
        ch = val.get("choices")
        if ch:
            out.extend(ch)
    return out

def process_single_json(json_path):
    logger.info(f"Processing: {json_path}")
    
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for item in data:
        text = str(item.get("data", {}).get("text"))
        choices = []

        for ann in item.get("annotations", []):
            results = ann.get("result", [])
            choices.extend(_extract_from_results(results))

        if not choices:
            for pred in item.get("predictions", []):
                results = pred.get("result", [])
                choices.extend(_extract_from_results(results))

        if choices:
            choices = list(dict.fromkeys(choices))
            label = str(choices[0])
            rows.append({"text": text, "label": label})

        

    return pd.DataFrame(rows)

def prepare_df_from_folder(folder_path):
    folder = Path(folder_path)
    all_dfs = []

    for json_file in folder.rglob("*.json"):
        df = process_single_json(json_file)
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(columns=["text", "label"])
    
    return pd.concat(all_dfs, ignore_index=True)


def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df["label_orig"] = df["label"]

    unique_labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: str(label) for label, i in label2id.items()}
    df["label"] = df["label"].map(label2id)

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = TextDataset(train_df["text"], train_df["label"], tokenizer)
    val_dataset = TextDataset(val_df["text"], val_df["label"], tokenizer)
    test_dataset = TextDataset(test_df["text"], test_df["label"], tokenizer)

    counts = Counter(train_df["label"])
    num_samples = len(train_df)
    num_classes = len(counts)
    weights = [num_samples / (num_classes * counts[i]) for i in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float)


    return train_dataset, val_dataset, test_dataset, tokenizer, id2label, len(label2id), class_weights

def _clean_df(df):
    df['word_count'] = df['text'].str.split().str.len()
    df_clean = df[df['word_count'] >= 7].copy()
    df_clean = df_clean.drop("word_count", axis=1)
    logger.info(f"Data cleaning is ready. Original size of data: {len(df)}, Size of data after cleaning: {len(df_clean)}")
    return df_clean

if __name__=="__main__":
    df = prepare_df_from_folder(RAW_DATA_FOLDER_PATH)
    df = _clean_df(df)
    df.to_csv(DF_PATH, index=False)
