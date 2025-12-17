import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import ast
import os
import sys
import math
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from joblib import Parallel, delayed
import warnings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    total_threads = os.cpu_count()
    NUM_WORKERS = max(1, total_threads - 2)
    torch.set_num_threads(NUM_WORKERS)
    device = torch.device('cpu')
else:
    NUM_WORKERS = 1
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

def run_textfooler_worker(idx, model, vocab, index_to_word, text_tensor, label_tensor):
    torch.set_num_threads(1)
    
    device = torch.device('cpu')
    text_tensor = text_tensor.to(device)
    true_label = label_tensor.to(device)
    label_val = true_label.item()
    
    model.eval()
    
    orig_cpu = text_tensor.cpu()
    orig_str = " ".join([index_to_word.get(x.item(),'') for x in orig_cpu if x.item()!=0])
    
    def get_synonyms(word_idx, embedding_matrix, k=20):
        target_vec = embedding_matrix[word_idx]
        norm_target = target_vec.norm(p=2)
        norm_matrix = embedding_matrix.norm(p=2, dim=1)
        sims = torch.matmul(embedding_matrix, target_vec) / (norm_matrix * norm_target + 1e-9)
        best_vals, best_idxs = torch.topk(sims, k + 1)
        return best_idxs[1:]

    with torch.no_grad():
        output_orig = model(text_tensor.unsqueeze(0))
        pred_orig = output_orig.argmax(1).item()
        prob_orig = F.softmax(output_orig, dim=1)[0][true_label].item()
    
    if pred_orig != label_val:
        return (None, None, None, False, False) 

    seq_len = text_tensor.shape[0]
    importance_scores = []
    embedding_matrix = model.embedding.weight.detach()

    for i in range(seq_len):
        token_id = text_tensor[i].item()
        if token_id == 0: continue 
        
        masked_text = text_tensor.clone().unsqueeze(0)
        masked_text[0, i] = 0 
        
        with torch.no_grad():
            output_masked = model(masked_text)
            prob_masked = F.softmax(output_masked, dim=1)[0][true_label].item()
        
        score = prob_orig - prob_masked
        importance_scores.append((score, i, token_id))
    
    importance_scores.sort(key=lambda x: x[0], reverse=True)
    
    current_text = text_tensor.clone().unsqueeze(0)
    
    for score, pos, original_token_id in importance_scores:
        candidates = get_synonyms(original_token_id, embedding_matrix, k=15) 
        
        for cand_id in candidates:
            cand_id = cand_id.item()
            if cand_id == 0: continue
            
            temp_text = current_text.clone()
            temp_text[0, pos] = cand_id
            
            with torch.no_grad():
                output_cand = model(temp_text)
                pred_cand = output_cand.argmax(1).item()
            
            if pred_cand != label_val:
                adv_cpu = temp_text[0].cpu()
                adv_str = " ".join([index_to_word.get(x.item(),'') for x in adv_cpu if x.item()!=0])
                return (orig_str, adv_str, label_val, True, True)
    
    return (orig_str, orig_str, label_val, True, False)


if __name__ == "__main__":
    print("--- 1. Încărcare Date ---")
    if not os.path.exists('ag_news_tokens.csv'):
        sys.exit("Lipsă fișier CSV!")
        
    df = pd.read_csv('ag_news_tokens.csv')
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    if df['label'].min() > 0:
        df['label'] = df['label'] - 1 

    print("--- 2. Construire Vocabular ---")
    all_words = [word for tokens in df['tokens'] for word in tokens]
    word_counts = Counter(all_words)
    vocab = {word: i+1 for i, (word, count) in enumerate(word_counts.most_common(10000))}
    vocab['<PAD>'] = 0
    index_to_word = {i: word for word, i in vocab.items()}
    final_vocab_size = len(vocab)

    dataset = NewsDataset(df, vocab)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)

    print("--- 3. Antrenare Model ---")
    model = TextCNN(final_vocab_size, 100, 100, [3,4,5], 4, 0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10): 
        epoch_loss = 0
        for text, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(text), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} done. Loss: {epoch_loss/len(loader):.4f}")

    model.eval()
    
    total_incercari = len(dataset) 
    print(f"\n--- 4. Start Atac TextFooler (Threads: {NUM_WORKERS}) ---")
    print(f"Număr știri de atacat: {total_incercari}")

    tasks_data = []
    for i in range(total_incercari):
        t, l = dataset[i]
        tasks_data.append((i, t, l))

    results = Parallel(
        n_jobs=NUM_WORKERS, 
        backend="loky", 
        verbose=5,
        batch_size=512,
        pre_dispatch='all'
    )(
        delayed(run_textfooler_worker)(
            idx, model, vocab, index_to_word, t_tensor, l_tensor
        ) for idx, t_tensor, l_tensor in tasks_data
    )

    
    atacuri_reusite_csv = []
    eval_texts = []
    eval_labels = []
    
    corecte_initial = 0
    succes_atac = 0

    for res in results:
        orig, adv, label, is_correct_init, is_success = res
        
        if not is_correct_init:
            continue
            
        corecte_initial += 1
        eval_texts.append(adv)
        eval_labels.append(label)
        
        if is_success:
            succes_atac += 1
            atacuri_reusite_csv.append((orig, adv))

    print(f"\n--- 5. Evaluare Metrici POST-ATAC ---")
    print("Se evaluează performanța modelului pe textele rezultate...")
    
    eval_batch_size = 512
    num_batches = math.ceil(len(eval_texts) / eval_batch_size)
    
    all_adv_preds = []
    all_adv_targets = []
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_texts = eval_texts[i*eval_batch_size : (i+1)*eval_batch_size]
            batch_labels = eval_labels[i*eval_batch_size : (i+1)*eval_batch_size]
            
            
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
    print(f"REZULTATE DUPĂ ATACUL TEXTFOOLER (Pe subsetul inițial corect):")
    print("-" * 60)
    print(f"Acuratețe (Robustness): {acc * 100:.2f}%")
    print(f"Precizie (W): {p * 100:.2f}%")
    print(f"Recall (W): {r * 100:.2f}%")
    print(f"F1 Score (W): {f1 * 100:.2f}%")
    print("\nMatricea de Confuzie:")
    print(cm)
    print("-" * 60)
    
    print(f"\nStatistici Atac:")
    print(f"Corecte inițial: {corecte_initial}")
    print(f"Atacuri reușite (flip): {succes_atac}")
    if corecte_initial > 0:
        print(f"Rata succes atac (ASR): {(succes_atac/corecte_initial)*100:.2f}%")

    pd.DataFrame(atacuri_reusite_csv, columns=['Original', 'Adversarial']).to_csv('cnn_textfooler_results.csv', index=False)