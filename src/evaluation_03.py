import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from data_processing_01 import load_and_prepare_data
from modules.LegalBERT import LegalBERT
import os
from config import BATCH_SIZE, DF_PATH, MODEL_OUTPUT
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils.logger import logger

def evaluate(model_path=MODEL_OUTPUT, data_path=DF_PATH):
    # seed is the same everywhere, so the val dataset will be the same as well
    _train_dataset, _val_dataset, test_dataset, _tokenizer, id2label, _num_labels, _class_weights = load_and_prepare_data(data_path)
    model = LegalBERT(num_labels=5)
    weights_path = os.path.join(model_path, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
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

    logger.info(classification_report(true, preds, zero_division=0))
    logger.info(f"Accuracy:{accuracy_score(true, preds)}")

    cm = confusion_matrix(true, preds)
    labels = [id2label[i] for i in range(len(id2label))]
    plt.figure(figsize=(17, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix')
    
    output_file = os.path.join(model_path, "confusion_matrix.png")
    plt.savefig(output_file)
    logger.info(f"Confusion matrix is saved: {output_file}")

if __name__ == "__main__":
    evaluate()
