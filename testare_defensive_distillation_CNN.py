import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
import time
import ast
import sys
import os


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    total_threads = os.cpu_count()
    NUM_WORKERS = max(1, total_threads - 2)
    torch.set_num_threads(NUM_WORKERS)
    device = torch.device('cpu')
else:
    NUM_WORKERS = 1
    device = torch.device('cpu')

FILE_ORIGINAL = "ag_news_tokens.csv"
FILE_HOTFLIP = "cnn_hotflip_results.csv"      
FILE_TEXTFOOLER = "cnn_textfooler_results.csv"

BATCH_SIZE = 2048   
EPOCHS = 20         
LR = 0.001
EMBED_DIM = 100
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 100
MAX_SEQ_LEN = 100
PAD_IDX = 1 

DISTILLATION_TEMP = 20 
ALPHA = 1.0            

tokenizer = get_tokenizer('basic_english')

def load_and_align_data(orig_path, attack_path, attack_name="Attack"):
    print(f"\n--> Începe alinierea pentru {attack_name}...")
    
    if not os.path.exists(attack_path):
        print(f"   [Warning] Fișierul {attack_path} nu există! Se sare peste acest atac.")
        return None

    try:
        df_orig = pd.read_csv(orig_path)
        df_orig['label'] = pd.to_numeric(df_orig['label'], errors='coerce').fillna(0).astype(int)
        if df_orig['label'].min() > 0: df_orig['label'] -= 1
        
        def tokens_to_key(token_str):
            try: return " ".join(ast.literal_eval(token_str))
            except: return str(token_str)

        df_orig['key_text'] = df_orig['tokens'].apply(tokens_to_key)
        label_map = pd.Series(df_orig.label.values, index=df_orig.key_text).to_dict()
        
        df_mod = pd.read_csv(attack_path)
        
        valid_rows = []
        found = 0
        
        col_orig = 'Original' if 'Original' in df_mod.columns else df_mod.columns[0]
        col_adv = 'Adversarial' if 'Adversarial' in df_mod.columns else df_mod.columns[1]

        for _, row in df_mod.iterrows():
            orig_text = str(row[col_orig]).strip()
            adv_text = str(row[col_adv]).strip()
            
            if orig_text in label_map:
                valid_rows.append({
                    'label': label_map[orig_text],
                    'text': adv_text 
                })
                found += 1
        
        print(f"   Aliniat {found} exemple din {len(df_mod)}.")
        
        if found == 0: return None
        return pd.DataFrame(valid_rows)

    except Exception as e:
        print(f"EROARE la aliniere {attack_name}: {e}")
        return None


class TextPipeline:
    def __init__(self, vocab):
        self.vocab = vocab
        self.tokenizer = tokenizer 
    def __call__(self, text):
        return self.vocab(self.tokenizer(str(text)))

def yield_tokens(text_list):
    for text in text_list:
        yield tokenizer(str(text))

class CustomDataset(Dataset):
    def __init__(self, dataframe, text_pipeline_obj):
        self.df = dataframe.reset_index(drop=True)
        self.text_pipeline = text_pipeline_obj
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['label'])
        text = str(row['text'])
        token_ids = self.text_pipeline(text)
        if len(token_ids) > MAX_SEQ_LEN: token_ids = token_ids[:MAX_SEQ_LEN]
        return torch.tensor(label, dtype=torch.long), torch.tensor(token_ids, dtype=torch.long)

def collate_fn(batch):
    labels, texts = zip(*batch)
    labels = torch.stack(labels)
    texts = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    return labels.to(device), texts.to(device)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, output_dim, dropout=0.5, pad_idx=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=fs) 
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


def train_teacher(model, loader, epochs):
    print("\n--> [Faza 1] Antrenare TEACHER (Standard)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for labels, texts in loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        
        acc = 100 * correct / total
        print(f"   | Teacher Epoch {epoch+1} | Loss: {epoch_loss/len(loader):.4f} | Acc: {acc:.2f}%")
    return model

def train_student_distillation(student, teacher, loader, epochs, temp, alpha):
    print(f"\n--> [Faza 2] Antrenare STUDENT (Defensive Distillation T={temp})...")
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    
    
    distill_criterion = nn.KLDivLoss(reduction='batchmean')
    student_criterion = nn.CrossEntropyLoss()
    
    teacher.eval() 
    student.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for labels, texts in loader:
            optimizer.zero_grad()
            
            
            with torch.no_grad():
                teacher_logits = teacher(texts)
                teacher_probs = F.softmax(teacher_logits / temp, dim=1)
            
            
            student_logits = student(texts)
            
            
            loss_soft = distill_criterion(
                F.log_softmax(student_logits / temp, dim=1), 
                teacher_probs
            )
            
            
            loss = (alpha * (temp ** 2) * loss_soft) 
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            correct += (student_logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            
        acc = 100 * correct / total
        print(f"   | Student Epoch {epoch+1} | Loss: {epoch_loss/len(loader):.4f} | Acc: {acc:.2f}%")
    
    return student


def evaluate_model(model, df, pipeline_obj, title):
    if df is None or len(df) == 0:
        print(f"\n[SKIP] Nu există date pentru: {title}")
        return

    
    ds = CustomDataset(df, pipeline_obj)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for labels, texts in loader:
            preds = model(texts).argmax(1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            
    
    acc = accuracy_score(all_labels, all_preds) * 100
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("-" * 60)
    print(f"REZULTATE STUDENT vs {title}")
    print("-" * 60)
    print(f"Acuratețe:    {acc:.2f}%")
    print(f"Precizie (W): {p*100:.2f}%")
    print(f"Recall (W):   {r*100:.2f}%")
    print(f"F1 Score (W): {f1*100:.2f}%")
    print("\nMatricea de Confuzie:")
    print(cm)
    print("-" * 60 + "\n")


def main():
    print("--> Încărcare date antrenare (Original)...")
    df_train_full = pd.read_csv(FILE_ORIGINAL)
    df_train_full['label'] = pd.to_numeric(df_train_full['label'], errors='coerce').fillna(0).astype(int)
    if df_train_full['label'].min() > 0: df_train_full['label'] -= 1
    
    def clean_tok(x):
        try: return " ".join(ast.literal_eval(x))
        except: return str(x)
    df_train_full['text'] = df_train_full['tokens'].apply(clean_tok)
    
    df_train, _ = train_test_split(df_train_full, test_size=0.05, random_state=42)
    
    print("--> Construire Vocabular...")
    vocab = build_vocab_from_iterator(yield_tokens(df_train['text']), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    global PAD_IDX
    PAD_IDX = vocab["<pad>"]
    pipeline = TextPipeline(vocab)
    
    ds_train = CustomDataset(df_train, pipeline)
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    print(f"--> Inițializare CNN (Batch: {BATCH_SIZE}, Epochs: {EPOCHS})...")
    teacher_model = TextCNN(len(vocab), EMBED_DIM, NUM_FILTERS, FILTER_SIZES, 4, pad_idx=PAD_IDX).to(device)
    student_model = TextCNN(len(vocab), EMBED_DIM, NUM_FILTERS, FILTER_SIZES, 4, pad_idx=PAD_IDX).to(device)
    
    teacher_model = train_teacher(teacher_model, train_loader, epochs=EPOCHS)
    
    student_model = train_student_distillation(student_model, teacher_model, train_loader, 
                                               epochs=EPOCHS, temp=DISTILLATION_TEMP, alpha=ALPHA)
    
    print("\n--> Încărcare seturi de test (Atacuri)...")
    
    df_hotflip = load_and_align_data(FILE_ORIGINAL, FILE_HOTFLIP, "HotFlip")
    evaluate_model(student_model, df_hotflip, pipeline, "ATAC HOTFLIP")
    
    df_textfooler = load_and_align_data(FILE_ORIGINAL, FILE_TEXTFOOLER, "TextFooler")
    evaluate_model(student_model, df_textfooler, pipeline, "ATAC TEXTFOOLER")
    
    print("\nProces Defensive Distillation Finalizat.")

if __name__ == "__main__":
    main()