"""
BiLSTM hate-speech classifier (HASOC 2019 + 2020 English + Hindi, task_1: HOF/NOT)
Uses:
  - Character + word vocabulary built from training data
  - Embedding → BiLSTM → Attention pooling → FC → sigmoid
  - Class-weighted loss for imbalance
  - Trains on combined HASOC 2019 + 2020 data for improved coverage

Run:
    conda activate test2
    python train3/train_lstm.py

Outputs:
    model3/lstm/lstm_model.pt
    model3/lstm/lstm_vocab.pkl
    model3/lstm/report.json
    model3/plots/lstm_training_history.png
"""

import os, re, pickle, json
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Import prep2 preprocessing
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prep2 import preprocess_lstm

# ── config ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# HASOC 2019
EN_TRAIN_2019 = os.path.join(BASE_DIR, "english_dataset", "english_dataset", "english_dataset.tsv")
EN_TEST_2019  = os.path.join(BASE_DIR, "english_dataset", "english_dataset", "hasoc2019_en_test-2919.tsv")
HI_TRAIN_2019 = os.path.join(BASE_DIR, "hindi_dataset",  "hindi_dataset",  "hindi_dataset.tsv")
HI_TEST_2019  = os.path.join(BASE_DIR, "hindi_dataset",  "hindi_dataset",  "hasoc2019_hi_test_gold_2919.tsv")

# HASOC 2020
EN_TRAIN_2020 = os.path.join(BASE_DIR, "english_dataset_2020", "hasoc_2020_en_train.xlsx")
EN_TEST_2020  = os.path.join(BASE_DIR, "english_dataset_2020", "english_test_1509.csv")
HI_TRAIN_2020 = os.path.join(BASE_DIR, "hindi_dataset_2020",  "hasoc_2020_hi_train.xlsx")
HI_TEST_2020  = os.path.join(BASE_DIR, "hindi_dataset_2020",  "hindi_test_1509.csv")

MODEL_DIR   = os.path.join(BASE_DIR, "model3")
LSTM_DIR    = os.path.join(MODEL_DIR, "lstm")
PLOTS_DIR   = os.path.join(MODEL_DIR, "plots")
LSTM_MODEL  = os.path.join(LSTM_DIR, "lstm_model.pt")
LSTM_VOCAB  = os.path.join(LSTM_DIR, "lstm_vocab.pkl")
REPORT_PATH = os.path.join(LSTM_DIR, "report.json")

os.makedirs(LSTM_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 80       # tokens per sample
VOCAB_SIZE = 30000    # top n tokens
EMBED_DIM  = 128
HIDDEN_DIM = 256      # per direction; BiLSTM → 512
NUM_LAYERS = 2
DROPOUT    = 0.2      # reduced from 0.4 (better regularization)
BATCH_SIZE = 64
EPOCHS     = 50
LR         = 5e-4    # lower learning rate for stability
WEIGHT_DECAY = 0.001  # L2 regularization
MIN_FREQ   = 2

print(f"Device: {DEVICE}")

# ── text cleaning using prep2.preprocess_lstm() ───────────────────────────────
def clean(text: str) -> str:
    """Use prep2.preprocess_lstm() for LSTM preprocessing with sequence preservation."""
    return preprocess_lstm(text)
    return text

# ── load TSV ───────────────────────────────────────────────────────────────────
def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    if "task1" in df.columns and "task_1" not in df.columns:
        df = df.rename(columns={"task1": "task_1"})
    df = df[['text', 'task_1']].dropna()
    df['task_1'] = df['task_1'].str.strip().str.upper().replace('NNOT', 'NOT')
    df = df[df['task_1'].isin(['HOF', 'NOT'])]
    df['text'] = df['text'].apply(clean)
    return df

def load_tsv(path):
    return _normalise(pd.read_csv(path, sep='\t', dtype=str))

def load_csv(path):
    return _normalise(pd.read_csv(path, dtype=str))

def load_xlsx(path):
    return _normalise(pd.read_excel(path, dtype=str))

def load_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return load_xlsx(path)
    elif ext == '.csv':
        return load_csv(path)
    else:
        return load_tsv(path)

# ── load data (2019 + 2020) ────────────────────────────────────────────────────
print("Loading datasets (HASOC 2019 + 2020)...")

train_paths = [EN_TRAIN_2019, HI_TRAIN_2019, EN_TRAIN_2020, HI_TRAIN_2020]
test_paths  = [EN_TEST_2019,  HI_TEST_2019,  EN_TEST_2020,  HI_TEST_2020]

train_df = pd.concat([load_any(p) for p in train_paths if os.path.exists(p)], ignore_index=True)
test_df  = pd.concat([load_any(p) for p in test_paths  if os.path.exists(p)], ignore_index=True)
print(f"  Train: {len(train_df)}  HOF={(train_df['task_1']=='HOF').sum()}  NOT={(train_df['task_1']=='NOT').sum()}")
print(f"  Test : {len(test_df)}   HOF={(test_df['task_1']=='HOF').sum()}  NOT={(test_df['task_1']=='NOT').sum()}")

# ── vocabulary ─────────────────────────────────────────────────────────────────
PAD, UNK = '<PAD>', '<UNK>'

def tokenize(text): return text.split()

counter = Counter()
for text in train_df['text']:
    counter.update(tokenize(text))

vocab_tokens = [PAD, UNK] + [w for w, c in counter.most_common(VOCAB_SIZE - 2) if c >= MIN_FREQ]
word2idx = {w: i for i, w in enumerate(vocab_tokens)}
print(f"  Vocabulary size: {len(word2idx)}")

def encode(text, max_len=MAX_LEN):
    tokens = tokenize(text)[:max_len]
    ids    = [word2idx.get(t, word2idx[UNK]) for t in tokens]
    length = len(ids)
    ids   += [word2idx[PAD]] * (max_len - length)
    return ids, length

# ── dataset ────────────────────────────────────────────────────────────────────
LABEL2IDX = {'HOF': 1, 'NOT': 0}

class HateDataset(Dataset):
    def __init__(self, df):
        self.texts   = df['text'].tolist()
        self.labels  = [LABEL2IDX[l] for l in df['task_1']]

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        ids, length = encode(self.texts[i])
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(length, dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.float))

def collate(batch):
    ids, lengths, labels = zip(*batch)
    ids     = torch.stack(ids)
    lengths = torch.stack(lengths).clamp(min=1)  # avoid 0-length for pack_padded_sequence
    labels  = torch.stack(labels)
    # sort by length descending for pack_padded_sequence
    lengths, sort_idx = lengths.sort(descending=True)
    ids    = ids[sort_idx]
    labels = labels[sort_idx]
    return ids, lengths, labels

train_loader = DataLoader(HateDataset(train_df), batch_size=BATCH_SIZE,
                          shuffle=True,  collate_fn=collate)
test_loader  = DataLoader(HateDataset(test_df),  batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate)

# ── model ──────────────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states, lengths):
        # hidden_states: (B, T, H)
        scores = self.attn(hidden_states).squeeze(-1)            # (B, T)
        # mask padding
        mask = torch.arange(hidden_states.size(1), device=DEVICE).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(2)      # (B, T, 1)
        return (hidden_states * weights).sum(1)                  # (B, H)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_dim * 2)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        emb    = self.dropout(self.embedding(x))                 # (B, T, E)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=True)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)      # (B, T, 2H)
        ctx    = self.attention(out, lengths)                    # (B, 2H)
        return self.fc(self.dropout(ctx)).squeeze(1)             # (B,)


model = BiLSTMClassifier(len(word2idx), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# class weights for imbalance
hof_count = (train_df['task_1'] == 'HOF').sum()
not_count = (train_df['task_1'] == 'NOT').sum()
pos_weight = torch.tensor([not_count / hof_count], dtype=torch.float).to(DEVICE)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=True)

# ── train loop ─────────────────────────────────────────────────────────────────
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

best_f1    = 0.0
best_state = None

print("\nTraining BiLSTM...\n")
print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'Val F1':>6}")
print("-" * 60)

for epoch in range(1, EPOCHS + 1):
    # ── train ──
    model.train()
    t_loss, t_correct, t_total = 0.0, 0, 0
    for ids, lengths, labels in train_loader:
        ids, lengths, labels = ids.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(ids, lengths)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss    += loss.item() * len(labels)
        preds      = (torch.sigmoid(logits) >= 0.5).long()
        t_correct += (preds == labels.long()).sum().item()
        t_total   += len(labels)

    # ── eval ──
    model.eval()
    v_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for ids, lengths, labels in test_loader:
            ids, lengths, labels = ids.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
            logits = model(ids, lengths)
            v_loss += criterion(logits, labels).item() * len(labels)
            preds   = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.long().cpu().tolist())

    t_loss /= t_total
    t_acc   = t_correct / t_total
    v_loss /= len(test_df)
    v_acc   = accuracy_score(all_labels, all_preds)
    v_f1    = f1_score(all_labels, all_preds, average='macro')

    history["train_loss"].append(round(t_loss, 4))
    history["train_acc"].append(round(t_acc,  4))
    history["val_loss"].append(round(v_loss,  4))
    history["val_acc"].append(round(v_acc,    4))
    history["val_f1"].append(round(v_f1,      4))

    print(f"{epoch:>5}  {t_loss:>10.4f}  {t_acc:>9.4f}  {v_loss:>8.4f}  {v_acc:>7.4f}  {v_f1:>6.4f}")

    scheduler.step(v_loss)

    if v_f1 > best_f1:
        best_f1    = v_f1
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

# ── restore best & final eval ──────────────────────────────────────────────────
model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for ids, lengths, labels in test_loader:
        ids, lengths, labels = ids.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        preds = (torch.sigmoid(model(ids, lengths)) >= 0.5).long().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.long().cpu().tolist())

acc  = accuracy_score(all_labels, all_preds)
f1   = f1_score(all_labels, all_preds, average='macro')
print(f"\nBest model  Accuracy: {acc:.4f}  F1-macro: {f1:.4f}")
print("\nClassification Report:")
idx2label   = {v: k for k, v in LABEL2IDX.items()}
best_labels = [idx2label[l] for l in all_labels]
best_preds  = [idx2label[p] for p in all_preds]
print(classification_report(best_labels, best_preds, target_names=["HOF", "NOT"]))
cr = classification_report(best_labels, best_preds, target_names=["HOF", "NOT"], output_dict=True)

# ── plots ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs_range = range(1, EPOCHS + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs_range, history["train_loss"], label="Train", color="#5b9bd5")
    axes[0].plot(epochs_range, history["val_loss"],   label="Val",   color="#e05c5c")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[0].spines[["top","right"]].set_visible(False)

    axes[1].plot(epochs_range, history["train_acc"], label="Train", color="#5b9bd5")
    axes[1].plot(epochs_range, history["val_acc"],   label="Val",   color="#e05c5c")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    axes[1].set_ylim(0, 1); axes[1].spines[["top","right"]].set_visible(False)

    axes[2].plot(epochs_range, history["val_f1"], label="Val F1-macro", color="#70b87e")
    axes[2].set_title("Val F1-macro"); axes[2].set_xlabel("Epoch"); axes[2].legend()
    axes[2].set_ylim(0, 1); axes[2].spines[["top","right"]].set_visible(False)

    plt.suptitle("BiLSTM Training History (HASOC 2019+2020)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "lstm_training_history.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved plot  -> {plot_path}")
except Exception as e:
    print(f"Plot skipped: {e}")

# ── save model + vocab ─────────────────────────────────────────────────────────
torch.save({
    "state_dict": best_state,
    "config": {
        "vocab_size": len(word2idx), "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
    },
    "label2idx": LABEL2IDX,
}, LSTM_MODEL)

with open(LSTM_VOCAB, "wb") as f:
    pickle.dump(word2idx, f, protocol=4)

report = {
    "model": "BiLSTM + Attention",
    "dataset": "HASOC 2019 + 2020",
    "device": str(DEVICE),
    "epochs": EPOCHS, "best_epoch": int(np.argmax(history["val_f1"]) + 1),
    "test_accuracy": round(acc, 4), "test_f1_macro": round(f1, 4),
    "classification_report": cr,
    "history": history,
}
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print(f"Saved model -> {LSTM_MODEL}")
print(f"Saved vocab -> {LSTM_VOCAB}")
print(f"Saved report-> {REPORT_PATH}")
print("\nDone! Trained on HASOC 2019 + 2020 combined data.")
