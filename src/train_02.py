import os
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, accuracy_score
from tqdm.auto import tqdm
from data_processing_01 import load_and_prepare_data
from modules.LegalBERT import LegalBERT
from config import EPOCHS, BATCH_SIZE, LR, DF_PATH, MODEL_OUTPUT

def train():
    train_dataset, val_dataset, _test_dataset, tokenizer, _id2label, num_labels, class_weights = load_and_prepare_data(DF_PATH)

    model = LegalBERT(num_labels=num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    print(class_weights)

    loss_fct = CrossEntropyLoss(weight=class_weights.to(device))

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
            loss = loss_fct(outputs.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Train loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
                preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                true.extend(batch["labels"].cpu().tolist())

        print(classification_report(true, preds, zero_division=0))
        print("Accuracy:", accuracy_score(true, preds))

    os.makedirs(MODEL_OUTPUT, exist_ok=True)
    tokenizer.save_pretrained(MODEL_OUTPUT)
    torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT, "pytorch_model.bin"))
    model.bert.config.save_pretrained(MODEL_OUTPUT)

if __name__ == "__main__":
    train()
