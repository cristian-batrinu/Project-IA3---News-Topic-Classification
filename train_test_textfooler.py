import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from nltk.corpus import wordnet, stopwords
from nltk.tag import pos_tag
import nltk
from preprocess_and_cache import SUBSET_PERCENTAGE
from utils import get_dataloader_kwargs, parallel_map

MODEL_NAME = 'lucasresck/bert-base-cased-ag-news'
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
CACHE_DIR = 'cached_embeddings'
MODELS_DIR = 'models'
RESULTS_DIR = 'test_results'

MAX_PERTURBATIONS = 10
SIM_SCORE_THRESHOLD = 0.7
SIM_SCORE_WINDOW = 15
IMPORT_SCORE_THRESHOLD = -1.0
SYNONYM_NUM = 50
COS_SIM_THRESHOLD = 0.5

try:
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def get_stopwords():
    try:
        return set(stopwords.words('english'))
    except:
        return {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
                'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}

def get_pos(text_ls):
    pos_tags = pos_tag(text_ls)
    return [pos for word, pos in pos_tags]

def get_synonyms(word, pos_tag_word=None, synonym_num=50):
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
            if synonym != word and synonym.isalpha() and len(synonym.split()) == 1:
                synonyms.add(synonym)
            if len(synonyms) >= synonym_num:
                break
        if len(synonyms) >= synonym_num:
            break
    
    return list(synonyms)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def semantic_similarity(text1, text2, model, tokenizer, device):
    try:
        inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}
        
        with torch.no_grad():
            outputs1 = model.bert(**inputs1)
            outputs2 = model.bert(**inputs2)
            
            emb1 = outputs1.last_hidden_state[:, 0, :]
            emb2 = outputs2.last_hidden_state[:, 0, :]
            
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)
            
            sim = (emb1 * emb2).sum(dim=1).item()
        
        return sim
    except Exception as e:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

def textfooler_attack(model, tokenizer, text, label, device, 
                      max_perturbations=10, sim_score_threshold=0.7, 
                      sim_score_window=15, import_score_threshold=-1.0,
                      synonym_num=50, cos_sim_threshold=0.5, batch_size=32):
    model.eval()
    words = text.split()
    if len(words) == 0:
        return text
    
    orig_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    orig_inputs = {k: v.to(device) for k, v in orig_inputs.items()}
    
    with torch.no_grad():
        orig_outputs = model(**orig_inputs)
        orig_probs = F.softmax(orig_outputs.logits, dim=-1).squeeze()
        orig_label = torch.argmax(orig_probs).item()
        orig_prob = orig_probs[orig_label].item()
    
    if orig_label != label:
        return text
    
    len_text = len(words)
    if len_text < sim_score_window:
        sim_score_threshold = 0.1
    
    leave_1_texts = []
    for i in range(len_text):
        test_words = words.copy()
        test_words[i] = tokenizer.unk_token if tokenizer.unk_token else '[UNK]'
        leave_1_texts.append(' '.join(test_words))
    
    leave_1_probs = []
    with torch.no_grad():
        for i in range(0, len(leave_1_texts), batch_size):
            batch_texts = leave_1_texts[i:i+batch_size]
            batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, 
                                     truncation=True, max_length=MAX_LENGTH)
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            
            batch_outputs = model(**batch_inputs)
            batch_probs = F.softmax(batch_outputs.logits, dim=-1)
            leave_1_probs.append(batch_probs.cpu())
    
    leave_1_probs = torch.cat(leave_1_probs, dim=0).to(device)
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    
    import_scores = []
    for i in range(len_text):
        prob_orig_label = leave_1_probs[i][orig_label].item()
        if leave_1_probs_argmax[i].item() != orig_label:
            new_label = leave_1_probs_argmax[i].item()
            prob_new_label = leave_1_probs[i][new_label].item()
            score = orig_prob - prob_orig_label + (prob_new_label - orig_probs[new_label].item())
        else:
            score = orig_prob - prob_orig_label
        import_scores.append(score)
    
    import_scores = np.array(import_scores)
    
    stop_words_set = get_stopwords()
    pos_tags = get_pos(words)
    
    words_perturb = []
    for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
        if words[idx].lower() not in stop_words_set and score > import_score_threshold:
            words_perturb.append((idx, words[idx], pos_tags[idx], score))
    
    perturbed = words.copy()
    num_changed = 0
    
    for idx, word, pos, import_score in words_perturb:
        if num_changed >= max_perturbations:
            break
        
        synonyms = get_synonyms(word, pos, synonym_num=synonym_num)
        if not synonyms:
            continue
        
        candidate_synonyms = []
        for syn in synonyms[:20]:
            syn_pos_tags = get_pos([syn])
            if syn_pos_tags and syn_pos_tags[0][0] == pos[0]:
                candidate_synonyms.append(syn)
        
        if not candidate_synonyms:
            continue
        
        best_syn = None
        best_prob = orig_prob
        
        for syn in candidate_synonyms[:10]:
            test_words = perturbed.copy()
            test_words[idx] = syn
            test_text = ' '.join(test_words)
            
            if len_text >= sim_score_window:
                start_idx = max(0, idx - sim_score_window // 2)
                end_idx = min(len_text, idx + sim_score_window // 2 + 1)
                orig_window = ' '.join(words[start_idx:end_idx])
                test_window = ' '.join(test_words[start_idx:end_idx])
                
                try:
                    sim_score = semantic_similarity(orig_window, test_window, model, tokenizer, device)
                    if sim_score < sim_score_threshold:
                        continue
                except:
                    pass
            
            test_inputs = tokenizer(test_text, return_tensors="pt", padding=True, 
                                    truncation=True, max_length=MAX_LENGTH)
            test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
            
            with torch.no_grad():
                test_outputs = model(**test_inputs)
                test_probs = F.softmax(test_outputs.logits, dim=-1).squeeze()
                test_pred = torch.argmax(test_probs).item()
                test_prob_orig_label = test_probs[orig_label].item()
                
                if test_pred != orig_label:
                    return test_text
                
                if test_prob_orig_label < best_prob:
                    best_syn = syn
                    best_prob = test_prob_orig_label
        
        if best_syn:
            perturbed[idx] = best_syn
            num_changed += 1
            
            final_text = ' '.join(perturbed)
            final_inputs = tokenizer(final_text, return_tensors="pt", padding=True, 
                                    truncation=True, max_length=MAX_LENGTH)
            final_inputs = {k: v.to(device) for k, v in final_inputs.items()}
            
            with torch.no_grad():
                final_outputs = model(**final_inputs)
                final_pred = torch.argmax(final_outputs.logits, dim=-1).item()
            
            if final_pred != orig_label:
                return final_text
    
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
    torch.backends.cudnn.benchmark = True
    
    with torch.no_grad():
        for i in tqdm(range(attack_total), desc="Attack"):
            text = test_texts[i]
            label = test_labels[i]
            
            adv_text = textfooler_attack(
                model, tokenizer, text, label, device,
                max_perturbations=MAX_PERTURBATIONS,
                sim_score_threshold=SIM_SCORE_THRESHOLD,
                sim_score_window=SIM_SCORE_WINDOW,
                import_score_threshold=IMPORT_SCORE_THRESHOLD,
                synonym_num=SYNONYM_NUM,
                cos_sim_threshold=COS_SIM_THRESHOLD,
                batch_size=BATCH_SIZE
            )
            
            inputs = tokenizer(adv_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
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
    results.to_csv(os.path.join(RESULTS_DIR, f'textfooler_results.csv'), index=False)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Train - Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}")
    print(f"Val   - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}")
    print(f"Test  - Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}")
    print(f"Attack - Acc: {attack_acc:.4f}, Success: {attack_success:.4f}")
    print("="*60)
