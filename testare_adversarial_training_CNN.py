import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
import time
import ast 
import sys


warnings.filterwarnings("ignore")
device = torch.device("cpu")
print(f"--> [System] Device: {device}")

FILE_ORIGINAL = "ag_news_tokens.csv"
FILE_MODIFICAT = "cnn_hotflip_results.csv"


BATCH_SIZE = 2048
EMBED_DIM = 100
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 100
EPOCHS = 20 
LR = 0.001
MAX_SEQ_LEN = 100
PAD_IDX = 1 

tokenizer = get_tokenizer('basic_english')


def load_and_align_data(orig_path, mod_path):
    print("--> Începe alinierea inteligentă a datelor...")
    
    try:
        df_orig = pd.read_csv(orig_path)
        df_orig['label'] = pd.to_numeric(df_orig['label'], errors='coerce').fillna(0).astype(int)
        if df_orig['label'].min() > 0:
            df_orig['label'] = df_orig['label'] - 1
            
        print("   Procesare chei originale...")
        
        def tokens_to_key(token_str):
            try:
                return " ".join(ast.literal_eval(token_str))
            except:
                return str(token_str)

        df_orig['key_text'] = df_orig['tokens'].apply(tokens_to_key)
        
        label_map = pd.Series(df_orig.label.values, index=df_orig.key_text).to_dict()
        print(f"   Dicționar creat: {len(label_map)} intrări unice din original.")
        
    except Exception as e:
        print(f"EROARE la original: {e}")
        return None, None

    try:
        df_mod = pd.read_csv(mod_path)
        print(f"   Modificat raw: {len(df_mod)} rânduri.")
        
        valid_rows = []
        found_count = 0
        not_found_count = 0
        
        for idx, row in df_mod.iterrows():
            orig_text_key = str(row['Original']).strip()
            adv_text = str(row['Adversarial']).strip()
            
            if orig_text_key in label_map:
                true_label = label_map[orig_text_key]
                valid_rows.append({
                    'label': true_label,
                    'text_clean': orig_text_key, 
                    'text_adv': adv_text         
                })
                found_count += 1
            else:
                not_found_count += 1
                
        print(f"   Aliniere finalizată: {found_count} potrivite | {not_found_count} pierdute.")
        
        if found_count == 0:
            print("CRITIC: Nu s-a putut potrivi niciun rând!")
            return None, None

        df_aligned = pd.DataFrame(valid_rows)
        
        df_clean_final = df_aligned[['label', 'text_clean']].rename(columns={'text_clean': 'text'})
        df_adv_final = df_aligned[['label', 'text_adv']].rename(columns={'text_adv': 'text'})
        
        return df_clean_final, df_adv_final

    except Exception as e:
        print(f"EROARE la procesare modificat: {e}")
        return None, None

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

def collate_fn(batch):
    labels, texts = zip(*batch)
    labels = torch.stack(labels)
    texts = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    return labels.to(device), texts.to(device)

def evaluate(model, loader, title):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for labels, texts in loader:
            preds = model(texts).argmax(1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
    
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("-" * 60)
    print(f"REZULTATE PE SETUL: {title}")
    print("-" * 60)
    print(f"Acuratețe:    {acc * 100:.2f}%")
    print(f"Recall (W):   {r * 100:.2f}%")
    print(f"Precizie (W): {p * 100:.2f}%")
    print(f"F1 Score:     {f1 * 100:.2f}%")
    print("\nMatricea de Confuzie:")
    print(cm)
    print("-" * 60 + "\n")

def main():
    global PAD_IDX
    num_workers = 0 
    
    df_clean, df_adv = load_and_align_data(FILE_ORIGINAL, FILE_MODIFICAT)
    
    if df_clean is None or df_adv is None:
        return

    print("--> Construire vocabular...")
    vocab = build_vocab_from_iterator(yield_tokens(df_clean['text']), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    PAD_IDX = vocab["<pad>"]
    pipeline_obj = TextPipeline(vocab)

    print("--> Împărțire Train/Test...")
    indices = list(range(len(df_clean)))
    train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    
    train_clean_df = df_clean.iloc[train_idx]
    train_adv_df = df_adv.iloc[train_idx]
    test_clean_df = df_clean.iloc[test_idx]
    test_adv_df = df_adv.iloc[test_idx]

    ds_train_combined = ConcatDataset([
        CustomDataset(train_clean_df, pipeline_obj), 
        CustomDataset(train_adv_df, pipeline_obj)
    ])
    
    ds_test_clean = CustomDataset(test_clean_df, pipeline_obj)
    ds_test_adv = CustomDataset(test_adv_df, pipeline_obj)

    train_loader = DataLoader(ds_train_combined, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    test_loader_clean = DataLoader(ds_test_clean, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader_adv = DataLoader(ds_test_adv, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    print(f"--> Total exemple antrenare (Mix): {len(ds_train_combined)}")
    print(f"--> Total exemple testare (Clean): {len(ds_test_clean)}")
    print(f"--> Total exemple testare (Adv):   {len(ds_test_adv)}")

    model = TextCNN(len(vocab), EMBED_DIM, NUM_FILTERS, FILTER_SIZES, 4, pad_idx=PAD_IDX).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\n--> Start Antrenare (Adversarial Defense) pe CPU...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        start = time.time()
        
        for i, (labels, texts) in enumerate(train_loader):
            optimizer.zero_grad()
            preds = model(texts)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)
            
        acc_percent = (correct / total) * 100
        print(f"| Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc_percent:.2f}% | Time: {time.time()-start:.1f}s |")

    print("\n=== RAPORT FINAL ===")
    evaluate(model, test_loader_clean, "ORIGINAL (CURAT)")
    evaluate(model, test_loader_adv, "MODIFICAT (ATACAT)")
    print("\nProces finalizat.")

if __name__ == '__main__':
    main()