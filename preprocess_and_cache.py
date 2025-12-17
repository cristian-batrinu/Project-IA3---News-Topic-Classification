import os
import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

MODEL_NAME = 'lucasresck/bert-base-cased-ag-news'
SUBSET_PERCENTAGE = 0.05
MAX_LENGTH = 128
BATCH_SIZE = 32
CACHE_DIR = 'cached_embeddings'

stop_words = set(stopwords.words('english'))
symbols = "!\"#$%&()*+-./:;<=>?@[]\\^_`{|}~\n"
stemmer = SnowballStemmer('english')

class AGNewsDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.raw_text = [item['text'] for item in dataset]
        self.labels = [item['label'] for item in dataset]
        self.preprocessed_text = [self.preprocess_text(text) for text in tqdm(self.raw_text, desc="Preprocessing")]
        self.tokenizer = tokenizer
        self.max_length = 512
    
    def __len__(self):
        return len(self.dataset)
    
    def preprocess_text(self, raw_text):
        lower_text = raw_text.lower()
        tokens = word_tokenize(lower_text)
        filtered_tokens = [word for word in tokens if word not in stop_words and word not in symbols]
        stems = [stemmer.stem(token) for token in filtered_tokens]
        preprocessed_text = ' '.join(stems)
        return preprocessed_text
    
    def __getitem__(self, idx):
        item = {
            "raw_text": self.raw_text[idx],
            "label": self.labels[idx],
            "preprocessed_text": self.preprocessed_text[idx]
        }
        return item

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def extract_embeddings(model, tokenizer, texts, device, batch_size=32, max_length=128):
    if hasattr(model, 'bert'):
        base_model = model.bert
    else:
        base_model = model
    
    base_model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_emb)
    
    return np.vstack(embeddings)

if __name__ == '__main__':
    print("Preprocessing...")
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = get_device()
    
    print("Loading dataset...")
    train_ds = load_dataset("ag_news", split="train")
    test_ds = load_dataset("ag_news", split="test")
    
    train_df = train_ds.to_pandas()
    test_df = test_ds.to_pandas()
    
    train_subset, _ = train_test_split(train_df, test_size=1-SUBSET_PERCENTAGE, stratify=train_df['label'], random_state=42)
    test_subset, _ = train_test_split(test_df, test_size=1-SUBSET_PERCENTAGE, stratify=test_df['label'], random_state=42)
    
    train_subset, val_subset = train_test_split(train_subset, test_size=0.2, stratify=train_subset['label'], random_state=42)
    
    print(f"Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    model.to(device)
    
    print("Preprocessing texts...")
    train_dataset_list = [{'text': text, 'label': label} for text, label in zip(train_subset['text'], train_subset['label'])]
    val_dataset_list = [{'text': text, 'label': label} for text, label in zip(val_subset['text'], val_subset['label'])]
    test_dataset_list = [{'text': text, 'label': label} for text, label in zip(test_subset['text'], test_subset['label'])]
    
    train_ag = AGNewsDataset(train_dataset_list, tokenizer)
    val_ag = AGNewsDataset(val_dataset_list, tokenizer)
    test_ag = AGNewsDataset(test_dataset_list, tokenizer)
    
    train_texts = [item['preprocessed_text'] for item in train_ag]
    train_labels = (train_subset['label'] - 1).values
    train_labels = np.clip(train_labels, 0, 3)
    
    val_texts = [item['preprocessed_text'] for item in val_ag]
    val_labels = (val_subset['label'] - 1).values
    val_labels = np.clip(val_labels, 0, 3)
    
    test_texts = [item['preprocessed_text'] for item in test_ag]
    test_labels = (test_subset['label'] - 1).values
    test_labels = np.clip(test_labels, 0, 3)
    
    dataset_size = int(SUBSET_PERCENTAGE * 10000)
    
    print("Caching train...")
    train_emb = extract_embeddings(model, tokenizer, train_texts, device, BATCH_SIZE, MAX_LENGTH)
    np.save(os.path.join(CACHE_DIR, f'train_embeddings_{dataset_size}.npy'), train_emb)
    np.save(os.path.join(CACHE_DIR, f'train_labels_{dataset_size}.npy'), train_labels)
    
    print("Caching val...")
    val_emb = extract_embeddings(model, tokenizer, val_texts, device, BATCH_SIZE, MAX_LENGTH)
    np.save(os.path.join(CACHE_DIR, f'val_embeddings_{dataset_size}.npy'), val_emb)
    np.save(os.path.join(CACHE_DIR, f'val_labels_{dataset_size}.npy'), val_labels)
    
    print("Caching test...")
    test_emb = extract_embeddings(model, tokenizer, test_texts, device, BATCH_SIZE, MAX_LENGTH)
    np.save(os.path.join(CACHE_DIR, f'test_embeddings_{dataset_size}.npy'), test_emb)
    np.save(os.path.join(CACHE_DIR, f'test_labels_{dataset_size}.npy'), test_labels)
    
    print("Done!")
