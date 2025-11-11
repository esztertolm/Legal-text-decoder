import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data_processing_01 import load_and_prepare_data
from config import BATCH_SIZE, DF_PATH, MODEL_OUTPUT

def evaluate(model_path=MODEL_OUTPUT, data_path=DF_PATH):
    # seed is the same everywhere, so the val dataset will be the same as well
    _train_dataset, _val_dataset, test_dataset, _tokenizer, _id2label, _num_labels, _class_weights = load_and_prepare_data(data_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    preds, true = [], []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            true.extend(batch["labels"].cpu().tolist())

    print(classification_report(true, preds, zero_division=0))
    print("Accuracy:", accuracy_score(true, preds))

if __name__ == "__main__":
    evaluate()
