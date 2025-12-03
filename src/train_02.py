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
from config import EPOCHS, BATCH_SIZE, LR, DF_PATH, MODEL_OUTPUT, EARLY_STOPPING_PATIENCE, SEED

import random
import numpy as np

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

from utils.logger import logger

def train():
    logger.info("-" * 30)

    logger.info("The hyperparameters are the following:")
    logger.info(f"Number of epochs: {EPOCHS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LR}")
    train_dataset, val_dataset, _test_dataset, tokenizer, _id2label, num_labels, class_weights = load_and_prepare_data(DF_PATH)

    model = LegalBERT(num_labels=num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info("Model:")
    logger.info(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    logger.info(f"Number of parameters: {total_params:,}")
    logger.info(f"Number of trainable parameters: {trainable_params:,}")
    logger.info(f"Number of not trainable parameters: {non_trainable_params:,}")
    logger.info("-" * 30)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    logger.info(f" Class weights:{class_weights}")

    loss_fct = CrossEntropyLoss(weight=class_weights.to(device))
    best_val_loss = float('inf')
    early_stopping = 0

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
        logger.info(f"Train loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        preds, true = [], []
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
                loss = loss_fct(outputs.logits, batch["labels"])
                preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().tolist())
                true.extend(batch["labels"].cpu().tolist())
                total_loss += loss.item()
        val_loss = total_loss/len(train_loader)
        logger.info(f"Validation loss: {val_loss:.4f}")
        logger.info(classification_report(true, preds, zero_division=0))
        logger.info(f"Accuracy:{accuracy_score(true, preds)}")

        if val_loss< best_val_loss:
            best_val_loss = val_loss
            early_stopping = 0
        else:
            early_stopping+=1
        
        if early_stopping >= EARLY_STOPPING_PATIENCE:
            break

    os.makedirs(MODEL_OUTPUT, exist_ok=True)
    tokenizer.save_pretrained(MODEL_OUTPUT)
    torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT, "pytorch_model.bin"))
    model.bert.config.save_pretrained(MODEL_OUTPUT)

if __name__ == "__main__":
    train()
