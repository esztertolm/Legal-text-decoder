import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from config import MODEL_NAME, MAX_LENGTH, SEED, DF_PATH
import json
from collections import Counter


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
    
def preparing_df(json_path):
    print(f"Loading from: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        text = item.get("data", {}).get("text")
        choices = []


        for ann in item.get("annotations", []):
            results = ann.get("result", [])
            choices.extend(_extract_from_results(results))

        if not choices:
            for pred in item.get("predictions", []):
                results = pred.get("result", [])
                choices.extend(_extract_from_results(results))

        if choices:
            seen = set()
            unique = []
            for c in choices:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            choices = unique
        else:
            choices = None

        rows.append({"text": text, "label": choices[0]})

    df = pd.DataFrame(rows)
    df.to_csv(DF_PATH)


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

if __name__=="__main__":
    preparing_df("src/data/legal_text_dataset.json")
