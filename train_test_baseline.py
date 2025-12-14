import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import pandas as pd
from datetime import datetime

MODEL_NAME = 'lucasresck/bert-base-cased-ag-news'
SUBSET_PERCENTAGE = 0.002
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
CACHE_DIR = 'cached_embeddings'
MODELS_DIR = 'models'
RESULTS_DIR = 'test_results'

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def eval_model(model, dataloader, device):
    model.eval()
    preds = []
    labels = []
    
    with torch.no_grad():
        for emb_batch, label_batch in tqdm(dataloader, desc="Eval"):
            emb_batch = emb_batch.to(device)
            label_batch = label_batch.to(device)
            
            batch_size = emb_batch.size(0)
            emb_batch = emb_batch.unsqueeze(1)
            attn_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            
            outputs = model(inputs_embeds=emb_batch, attention_mask=attn_mask)
            pred = outputs.logits.argmax(dim=-1)
            
            preds.extend(pred.cpu().numpy())
            labels.extend(label_batch.cpu().numpy())
    
    acc = accuracy_score(labels, preds)
    rec = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    prec = precision_score(labels, preds, average='weighted')
    
    return acc, rec, f1, prec

if __name__ == '__main__':
    print("Training baseline...")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    
    dataset_size = int(SUBSET_PERCENTAGE * 10000)
    train_emb = np.load(os.path.join(CACHE_DIR, f'train_embeddings_{dataset_size}.npy'))
    train_labels = np.load(os.path.join(CACHE_DIR, f'train_labels_{dataset_size}.npy'))
    val_emb = np.load(os.path.join(CACHE_DIR, f'val_embeddings_{dataset_size}.npy'))
    val_labels = np.load(os.path.join(CACHE_DIR, f'val_labels_{dataset_size}.npy'))
    test_emb = np.load(os.path.join(CACHE_DIR, f'test_embeddings_{dataset_size}.npy'))
    test_labels = np.load(os.path.join(CACHE_DIR, f'test_labels_{dataset_size}.npy'))
    
    train_labels = np.clip(train_labels, 0, 3).astype(np.int64)
    val_labels = np.clip(val_labels, 0, 3).astype(np.int64)
    test_labels = np.clip(test_labels, 0, 3).astype(np.int64)
    
    train_ds = TensorDataset(torch.FloatTensor(train_emb), torch.LongTensor(train_labels))
    val_ds = TensorDataset(torch.FloatTensor(val_emb), torch.LongTensor(val_labels))
    test_ds = TensorDataset(torch.FloatTensor(test_emb), torch.LongTensor(test_labels))
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dl) * NUM_EPOCHS)
    
    best_val = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        
        for emb_batch, label_batch in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
            emb_batch = emb_batch.to(device)
            label_batch = label_batch.to(device)
            label_batch = torch.clamp(label_batch, 0, 3)
            
            batch_size = emb_batch.size(0)
            emb_batch = emb_batch.unsqueeze(1)
            attn_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            
            outputs = model(inputs_embeds=emb_batch, attention_mask=attn_mask, labels=label_batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        val_acc, _, _, _ = eval_model(model, val_dl, device)
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
    
    print("\nEvaluating...")
    train_acc, train_rec, train_f1, train_prec = eval_model(model, train_dl, device)
    val_acc, val_rec, val_f1, val_prec = eval_model(model, val_dl, device)
    test_acc, test_rec, test_f1, test_prec = eval_model(model, test_dl, device)
    
    model_path = os.path.join(MODELS_DIR, 'baseline_model')
    model.save_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(model_path)
    
    results = pd.DataFrame([{
        'experiment': 'baseline',
        'train_accuracy': train_acc,
        'train_recall': train_rec,
        'train_f1': train_f1,
        'train_precision': train_prec,
        'val_accuracy': val_acc,
        'val_recall': val_rec,
        'val_f1': val_f1,
        'val_precision': val_prec,
        'test_accuracy': test_acc,
        'test_recall': test_rec,
        'test_f1': test_f1,
        'test_precision': test_prec,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }])
    results.to_csv(os.path.join(RESULTS_DIR, f'baseline_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'), index=False)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Train - Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}")
    print(f"Val   - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}")
    print(f"Test  - Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}")
    print("="*60)
