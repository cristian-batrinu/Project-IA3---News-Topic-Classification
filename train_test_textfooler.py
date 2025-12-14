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
from nltk.corpus import wordnet
from nltk.tag import pos_tag
import nltk

MODEL_NAME = 'lucasresck/bert-base-cased-ag-news'
SUBSET_PERCENTAGE = 0.002
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
CACHE_DIR = 'cached_embeddings'
MODELS_DIR = 'models'
RESULTS_DIR = 'test_results'
MAX_PERTURBATIONS = 10

try:
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except:
    pass

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def get_synonyms(word, pos_tag_word=None):
    synonyms = set()
    if pos_tag_word:
        pos_map = {'N': 'n', 'V': 'v', 'J': 'a', 'R': 'r'}
        pos = pos_map.get(pos_tag_word[0], None)
        if pos:
            synsets = wordnet.synsets(word, pos=pos)
        else:
            synsets = wordnet.synsets(word)
    else:
        synsets = wordnet.synsets(word)
    
    for syn in synsets:
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != word and synonym.isalpha():
                synonyms.add(synonym)
    
    return list(synonyms)[:10]

def textfooler_attack(model, tokenizer, text, label, device, max_perturbations=10):
    model.eval()
    words = text.split()
    if len(words) == 0:
        return text
    
    pos_tags = pos_tag(words)
    perturbed = words.copy()
    count = 0
    
    for i, (word, pos) in enumerate(pos_tags):
        if count >= max_perturbations:
            break
        
        synonyms = get_synonyms(word, pos)
        if not synonyms:
            continue
        
        best_syn = None
        best_loss = float('inf')
        
        for syn in synonyms[:5]:
            test_words = perturbed.copy()
            test_words[i] = syn
            test_text = ' '.join(test_words)
            
            inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                pred = outputs.logits.argmax(dim=-1).item()
                loss = nn.CrossEntropyLoss()(outputs.logits, torch.tensor([label]).to(device)).item()
            
            if pred != label and loss < best_loss:
                best_syn = syn
                best_loss = loss
        
        if best_syn:
            perturbed[i] = best_syn
            count += 1
    
    return ' '.join(perturbed)

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
    print("Training with TextFooler...")
    
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
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
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
        label = test_labels[i]
        
        adv_text = textfooler_attack(model, tokenizer, text, label, device, MAX_PERTURBATIONS)
        
        inputs = tokenizer(adv_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.logits.argmax(dim=-1).item()
        
        if pred == label:
            attack_correct += 1
    
    attack_acc = attack_correct / attack_total if attack_total > 0 else 0
    attack_success = 1 - attack_acc
    
    model_path = os.path.join(MODELS_DIR, 'textfooler_model')
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    results = pd.DataFrame([{
        'experiment': 'textfooler',
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
        'attack_accuracy': attack_acc,
        'attack_success_rate': attack_success,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'max_perturbations': MAX_PERTURBATIONS,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }])
    results.to_csv(os.path.join(RESULTS_DIR, f'textfooler_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'), index=False)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Train - Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}")
    print(f"Val   - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}")
    print(f"Test  - Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}")
    print(f"Attack - Acc: {attack_acc:.4f}, Success: {attack_success:.4f}")
    print("="*60)
