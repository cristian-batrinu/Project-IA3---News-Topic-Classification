import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from preprocess_and_cache import SUBSET_PERCENTAGE
from utils import get_dataloader_kwargs

MODEL_NAME = 'lucasresck/bert-base-cased-ag-news'
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
CACHE_DIR = 'cached_embeddings'
MODELS_DIR = 'models'
RESULTS_DIR = 'test_results'
MAX_PERTURBATIONS = 5

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def hotflip_attack(model, tokenizer, input_ids, attn_mask, label, device, max_perturbations=5):
    model.train()
    input_ids = input_ids.clone()
    vocab_emb = model.bert.embeddings.word_embeddings.weight
    
    count = 0
    for _ in range(max_perturbations):
        with torch.enable_grad():
            input_emb = model.bert.embeddings.word_embeddings(input_ids).detach().clone()
            input_emb.requires_grad_(True)
            
            outputs = model(inputs_embeds=input_emb, attention_mask=attn_mask)
            if label.dim() == 0:
                label_tensor = label.unsqueeze(0)
            else:
                label_tensor = label.view(1)
            loss = nn.CrossEntropyLoss()(outputs.logits, label_tensor)
            
            model.zero_grad()
            loss.backward()
            
            if input_emb.grad is None:
                break
            
            grad_emb = input_emb.grad[0]
        
        special_ids = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
        if tokenizer.unk_token_id:
            special_ids.append(tokenizer.unk_token_id)
        
        grad_norms = grad_emb.norm(dim=1)
        for i in range(len(grad_norms)):
            if input_ids[0, i].item() in special_ids:
                grad_norms[i] = -float('inf')
        
        token_idx = grad_norms.argmax().item()
        if token_idx >= input_ids.size(1) or grad_norms[token_idx] == -float('inf'):
            break
        
        grad_dir = grad_emb[token_idx] / (grad_emb[token_idx].norm() + 1e-8)
        grad_expanded = grad_dir.unsqueeze(0).expand(vocab_emb.size(0), -1)
        similarities = torch.cosine_similarity(vocab_emb, grad_expanded, dim=1)
        
        current_token = input_ids[0, token_idx].item()
        if current_token < len(similarities):
            similarities[current_token] = -1
        for sid in special_ids:
            if sid is not None and sid < len(similarities):
                similarities[sid] = -1
        
        best_token = similarities.argmax().item()
        input_ids[0, token_idx] = best_token
        count += 1
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            pred = outputs.logits.argmax(dim=-1).item()
            label_val = label.item() if label.dim() == 0 else label[0].item()
            if pred != label_val:
                break
        model.train()
    
    model.eval()
    return input_ids

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
    print("Training with HotFlip...")
    
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
    
    test_ds = load_dataset("ag_news", split="test")
    test_df = test_ds.to_pandas()
    test_subset, _ = train_test_split(test_df, test_size=1-SUBSET_PERCENTAGE, stratify=test_df['label'], random_state=42)
    test_texts = test_subset['text'].tolist()
    
    train_ds = TensorDataset(torch.FloatTensor(train_emb), torch.LongTensor(train_labels))
    val_ds = TensorDataset(torch.FloatTensor(val_emb), torch.LongTensor(val_labels))
    test_ds = TensorDataset(torch.FloatTensor(test_emb), torch.LongTensor(test_labels))
    
    train_dl = DataLoader(train_ds, **get_dataloader_kwargs(BATCH_SIZE, shuffle=True))
    val_dl = DataLoader(val_ds, **get_dataloader_kwargs(BATCH_SIZE, shuffle=False))
    test_dl = DataLoader(test_ds, **get_dataloader_kwargs(BATCH_SIZE, shuffle=False))
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dl) * NUM_EPOCHS)
    
    print("Training...")
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
    
    print("\nEvaluating...")
    train_acc, train_rec, train_f1, train_prec = eval_model(model, train_dl, device)
    val_acc, val_rec, val_f1, val_prec = eval_model(model, val_dl, device)
    test_acc, test_rec, test_f1, test_prec = eval_model(model, test_dl, device)
    
    print("Testing under attack...")
    attack_correct = 0
    attack_total = min(100, len(test_texts))
    
    model.eval()
    for i in tqdm(range(attack_total), desc="Attack"):
        text = test_texts[i]
        label = torch.tensor(test_labels[i], dtype=torch.long).to(device)
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        input_ids = inputs['input_ids'].to(device)
        attn_mask = inputs['attention_mask'].to(device)
        
        adv_ids = hotflip_attack(model, tokenizer, input_ids, attn_mask, label, device, MAX_PERTURBATIONS)
        
        with torch.no_grad():
            outputs = model(input_ids=adv_ids, attention_mask=attn_mask)
            pred = outputs.logits.argmax(dim=-1).item()
        
        if pred == test_labels[i]:
            attack_correct += 1
    
    attack_acc = attack_correct / attack_total if attack_total > 0 else 0
    attack_success = 1 - attack_acc
    
    model_path = os.path.join(MODELS_DIR, 'hotflip_model')
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    results = pd.DataFrame([{
        'experiment': 'hotflip',
        'train_accuracy': round(train_acc, 4),
        'train_recall': round(train_rec, 4),
        'train_f1': round(train_f1, 4),
        'train_precision': round(train_prec, 4),
        'val_accuracy': round(val_acc, 4),
        'val_recall': round(val_rec, 4),
        'val_f1': round(val_f1, 4),
        'val_precision': round(val_prec, 4),
        'test_accuracy': round(test_acc, 4),
        'test_recall': round(test_rec, 4),
        'test_f1': round(test_f1, 4),
        'test_precision': round(test_prec, 4),
        'attack_accuracy': round(attack_acc, 4),
        'attack_success_rate': round(attack_success, 4),
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'max_perturbations': MAX_PERTURBATIONS,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }])
    results.to_csv(os.path.join(RESULTS_DIR, f'hotflip_results.csv'), index=False)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Train - Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}")
    print(f"Val   - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}")
    print(f"Test  - Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}")
    print(f"Attack - Acc: {attack_acc:.4f}, Success: {attack_success:.4f}")
    print("="*60)
