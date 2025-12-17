import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import ast
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import sys
import os
import math
from joblib import Parallel, delayed
import warnings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    total_threads = os.cpu_count()
    NUM_WORKERS = max(1, int(total_threads * 0.90))
    torch.set_num_threads(NUM_WORKERS)
    device = torch.device('cpu')


def text_to_indices(tokens, vocab, max_len=100):
    if isinstance(tokens, str):
        tokens = tokens.split()
    
    indices = [vocab.get(w, 0) for w in tokens]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

class NewsDataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        indices = text_to_indices(row['tokens'], self.vocab)
        return torch.tensor(indices, dtype=torch.long), torch.tensor(row['label'], dtype=torch.long)

def calc_accuracy(preds, y):
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embed_dim)) 
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

    def forward_from_embeddings(self, embedded):
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

def run_single_attack(idx, model_state, vocab, index_to_word, text_tensor, label_tensor):
    torch.set_num_threads(1) 
    
    device = torch.device('cpu')
    text_tensor = text_tensor.unsqueeze(0).to(device)
    true_label = label_tensor.unsqueeze(0).to(device)
    label_val = true_label.item()
    
    criterion = nn.CrossEntropyLoss()
    model.eval() 
    
    orig_cpu = text_tensor[0]
    orig_text_list = [index_to_word.get(x.item(),'') for x in orig_cpu if x.item()!=0]
    orig_text_str = " ".join(orig_text_list)

    with torch.no_grad():
        initial_pred = model(text_tensor).argmax(1).item()
    
    if initial_pred != label_val:
        return (orig_text_str, orig_text_str, label_val, False, False) 
        
    embedding_layer = model.embedding
    embeds_original = embedding_layer(text_tensor).detach()
    embeds_original.requires_grad = True 
    
    output = model.forward_from_embeddings(embeds_original)
    loss = criterion(output, true_label)
    loss.backward()
    
    grad = embeds_original.grad.squeeze(0)
    embedding_matrix = embedding_layer.weight.detach()
    
    best_score = -float('inf')
    best_flip = None
    
    len_seq = text_tensor.shape[1]
    for i in range(len_seq):
        token_id = text_tensor[0, i].item()
        if token_id == 0: continue 
        
        scores = torch.matmul(embedding_matrix, grad[i])
        candidate_vals, candidate_idxs = torch.topk(scores, 10)
        
        for v, idx_new in zip(candidate_vals, candidate_idxs):
            if idx_new.item() != token_id and idx_new.item() != 0:
                score = v.item()
                if score > best_score:
                    best_score = score
                    best_flip = (i, idx_new.item())
    
    if best_flip:
        pos, new_id = best_flip
        new_text = text_tensor.clone()
        new_text[0, pos] = new_id
        
        with torch.no_grad():
            new_output = model(new_text)
        
        if new_output.argmax(1).item() != label_val:
            new_cpu = new_text[0]
            adv_text_list = [index_to_word.get(x.item(),'') for x in new_cpu if x.item()!=0]
            adv_text_str = " ".join(adv_text_list)
            return (orig_text_str, adv_text_str, label_val, True, True) 
            
    return (orig_text_str, orig_text_str, label_val, True, False)


if __name__ == "__main__":
    print("--- 1. Încărcare Date ---")
    if not os.path.exists('ag_news_tokens.csv'):
        print("Eroare: Fișierul 'ag_news_tokens.csv' nu există.")
        sys.exit()
        
    df = pd.read_csv('ag_news_tokens.csv')
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    if df['label'].min() > 0:
        df['label'] = df['label'] - 1 

    print("--- 2. Construire Vocabular ---")
    all_words = [word for tokens in df['tokens'] for word in tokens]
    word_counts = Counter(all_words)
    vocab_size = 10000 
    most_common = word_counts.most_common(vocab_size)
    vocab = {word: i+1 for i, (word, count) in enumerate(most_common)}
    vocab['<PAD>'] = 0
    index_to_word = {i: word for word, i in vocab.items()}
    final_vocab_size = len(vocab)

    dataset = NewsDataset(df, vocab)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=4)

    model = TextCNN(final_vocab_size, embed_dim=100, n_filters=100, filter_sizes=[3,4,5], output_dim=4, dropout=0.5)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    print(f"--- 3. Antrenare Model ---")
    model.train()
    for epoch in range(20): 
        epoch_loss = 0
        epoch_acc = 0
        for text, labels in loader:
            text, labels = text.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(text)
            loss = criterion(predictions, labels)
            acc = calc_accuracy(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        avg_loss = epoch_loss / len(loader)
        avg_acc = epoch_acc / len(loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.2f}%")

    model.eval()
    model.share_memory() 

    total_incercari = len(dataset) 
    print(f"\n--- 4. Start Atac Paralel (Threads: {NUM_WORKERS}) ---")
    
    tasks_data = []
    for i in range(total_incercari):
        t, l = dataset[i]
        tasks_data.append((i, t, l))

    results = Parallel(n_jobs=NUM_WORKERS, backend="loky", verbose=5, batch_size=2048, pre_dispatch='all')(
        delayed(run_single_attack)(
            idx, None, vocab, index_to_word, t_tensor, l_tensor
        ) for idx, t_tensor, l_tensor in tasks_data
    )

    atacuri_reusite_list = [] 
    adv_texts_for_eval = []
    true_labels_for_eval = []
    
    corecte_initial = 0
    succes_atac = 0

    for res in results:
        if res is None: continue
        orig, adv, label, is_initial_correct, is_success = res
        
        if is_initial_correct:
            corecte_initial += 1
            adv_texts_for_eval.append(adv) 
            true_labels_for_eval.append(label)
            
            if is_success:
                succes_atac += 1
                atacuri_reusite_list.append((orig, adv))
    
    print(f"\nRezultate Atac:")
    print(f"Exemple corecte inițial: {corecte_initial}")
    print(f"Atacuri reușite (flip): {succes_atac}")
    if corecte_initial > 0:
        rata = (succes_atac / corecte_initial) * 100
        print(f"Rata de succes a atacului (ASR): {rata:.2f}%")

    pd.DataFrame(atacuri_reusite_list, columns=['Original', 'Adversarial']).to_csv('cnn_hotflip_results.csv', index=False)

    print(f"\n--- 5. Evaluare Metrici POST-ATAC ---")
    print("Se procesează setul adversarial rezultat...")
    
    eval_batch_size = 100
    all_adv_preds = []
    all_adv_targets = []
    
    num_batches = math.ceil(len(adv_texts_for_eval) / eval_batch_size)
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_texts = adv_texts_for_eval[i*eval_batch_size : (i+1)*eval_batch_size]
            batch_labels = true_labels_for_eval[i*eval_batch_size : (i+1)*eval_batch_size]
            
            batch_indices = [text_to_indices(t, vocab) for t in batch_texts]
            tensor_text = torch.tensor(batch_indices, dtype=torch.long).to(device)
            tensor_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
            preds = model(tensor_text).argmax(1)
            
            all_adv_preds.extend(preds.cpu().numpy())
            all_adv_targets.extend(tensor_labels.cpu().numpy())

    acc = accuracy_score(all_adv_targets, all_adv_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_adv_targets, all_adv_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_adv_targets, all_adv_preds)
    
    print("-" * 60)
    print(f"REZULTATE DUPĂ ATAC (Pe subsetul inițial corect):")
    print("-" * 60)
    print(f"Acuratețe: {acc * 100:.2f}%")
    print(f"Recall (W): {r * 100:.2f}%")
    print(f"Precizie (W): {p * 100:.2f}%")
    print(f"F1 Score (W): {f1 * 100:.2f}%")
    print("\nMatricea de Confuzie:")
    print(cm)
    print("-" * 60)