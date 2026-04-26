"""
Full evaluation of all 4 models on the complete Hindi test set.
Loads hasoc2019_hi_test_gold_2919.tsv + hindi_test_1509.csv (combined: ~1981 samples).

Usage:
    conda activate test2
    cd C:\Documents\Projects\nlp_proj
    python train3/eval_hindi_full.py
"""

import os, sys, re, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prep2 import preprocess_lr, preprocess_lstm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── load Hindi test data ───────────────────────────────────────────────────────
HI_TEST_2019 = os.path.join(BASE_DIR, "hindi_dataset", "hindi_dataset", "hasoc2019_hi_test_gold_2919.tsv")
HI_TEST_2020 = os.path.join(BASE_DIR, "hindi_dataset_2020", "hindi_test_1509.csv")

def load_df(path):
    ext = os.path.splitext(path)[1].lower()
    df  = pd.read_csv(path, sep="\t" if ext == ".tsv" else ",", dtype=str)
    df.columns = df.columns.str.strip()
    if "task1" in df.columns and "task_1" not in df.columns:
        df = df.rename(columns={"task1": "task_1"})
    df = df[["text", "task_1"]].dropna()
    df["task_1"] = df["task_1"].str.strip().str.upper().replace("NNOT", "NOT")
    return df[df["task_1"].isin(["HOF", "NOT"])].reset_index(drop=True)

frames = []
for p in [HI_TEST_2019, HI_TEST_2020]:
    if os.path.exists(p):
        df = load_df(p)
        frames.append(df)
        print(f"  {os.path.basename(p)}: {len(df)} rows  HOF={(df['task_1']=='HOF').sum()}  NOT={(df['task_1']=='NOT').sum()}")
    else:
        print(f"  [SKIP] {p}")

test_df = pd.concat(frames, ignore_index=True)
print(f"\nTotal Hindi test: {len(test_df)}  HOF={(test_df['task_1']=='HOF').sum()}  NOT={(test_df['task_1']=='NOT').sum()}\n")

texts      = test_df["text"].tolist()
true_labels = test_df["task_1"].tolist()

# ── Devanagari extractor (must match train_model_v3.py for pickle to load) ────
import re as _re
from sklearn.base import BaseEstimator, TransformerMixin
_DEVA_RE_EXTRACTOR = _re.compile(r'[\u0900-\u097F]+')

class DevanagariExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return [' '.join(_DEVA_RE_EXTRACTOR.findall(t)) or ' ' for t in X]

# ── LR helpers ────────────────────────────────────────────────────────────────
def load_lr(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def batch_predict_lr(pipeline, texts):
    cleaned = [preprocess_lr(t) for t in texts]
    return list(pipeline.predict(cleaned))

# ── LSTM helpers ───────────────────────────────────────────────────────────────
class _AttentionV1(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.attn = nn.Linear(h, 1)
    def forward(self, hs, lengths):
        scores = self.attn(hs).squeeze(-1)
        mask   = torch.arange(hs.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
        return (hs * torch.softmax(scores, dim=1).unsqueeze(2)).sum(1)

class _AttentionV2(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.attn_hidden  = nn.Linear(h, h)
        self.attn_context = nn.Linear(h, 1, bias=False)
        self.attn_drop    = nn.Dropout(0.1)
    def forward(self, hs, lengths):
        energy = torch.tanh(self.attn_hidden(hs))
        scores = self.attn_context(energy).squeeze(-1)
        mask   = torch.arange(hs.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
        w      = self.attn_drop(torch.softmax(scores, dim=1)).unsqueeze(2)
        return (hs * w).sum(1)

class _BiLSTM(nn.Module):
    def __init__(self, vs, ed, hd, nl, dp, v1=False):
        super().__init__()
        self.embedding = nn.Embedding(vs, ed, padding_idx=0)
        self.lstm      = nn.LSTM(ed, hd, num_layers=nl, batch_first=True,
                                 bidirectional=True, dropout=dp if nl > 1 else 0)
        self.attention = _AttentionV1(hd * 2) if v1 else _AttentionV2(hd * 2)
        self.dropout   = nn.Dropout(dp)
        self.emb_drop  = nn.Identity()
        self.fc        = nn.Linear(hd * 2, 1)
    def forward(self, x, lengths):
        emb    = self.emb_drop(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=True)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.fc(self.dropout(self.attention(out, lengths))).squeeze(1)

def load_lstm(model_path, vocab_path):
    with open(vocab_path, "rb") as f:
        word2idx = pickle.load(f)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    is_v1 = "attention.attn.weight" in ckpt["state_dict"]
    net   = _BiLSTM(cfg["vocab_size"], cfg["embed_dim"],
                    cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"], v1=is_v1)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    label2idx = ckpt.get("label2idx", {"HOF": 1, "NOT": 0})
    threshold = cfg.get("hof_threshold", 0.5)
    idx2label = {v: k for k, v in label2idx.items()}
    return net, word2idx, idx2label, threshold

_TOK_RE = re.compile(r'[\u0900-\u097F]+|[a-zA-Z]+')
_NUKTA  = '\u093c'
_SW     = {"the","is","in","it","of","and","a","an","are","was","be"}

def _tokenize(text):
    word_toks = [t.replace(_NUKTA, '') for t in _TOK_RE.findall(text.lower())]
    word_toks = [t for t in word_toks if t and t not in _SW]
    result = list(word_toks)
    for w in word_toks:
        if len(w) >= 5:
            result.append(w[:3]); result.append(w[-3:])
    return result

def batch_predict_lstm(net, word2idx, idx2label, threshold, texts, max_len=120, batch_size=128):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = []
        for text in batch_texts:
            cleaned = preprocess_lstm(text)
            tokens  = _tokenize(cleaned)
            if len(tokens) > max_len:
                h = max_len // 2
                tokens = tokens[:h] + tokens[-h:]
            ids    = [word2idx.get(t, word2idx.get("<UNK>", 1)) for t in tokens]
            length = max(len(ids), 1)
            ids   += [0] * (max_len - len(ids))
            encoded.append((ids, length))
        # sort by length descending for pack_padded_sequence
        encoded_sorted = sorted(enumerate(encoded), key=lambda x: -x[1][1])
        orig_indices   = [e[0] for e in encoded_sorted]
        ids_batch      = torch.tensor([e[1][0] for e in encoded_sorted], dtype=torch.long)
        len_batch      = torch.tensor([e[1][1] for e in encoded_sorted], dtype=torch.long)
        with torch.no_grad():
            probs = torch.sigmoid(net(ids_batch, len_batch)).tolist()
        # unsort
        preds_sorted = [idx2label[1] if p >= threshold else idx2label[0] for p in probs]
        preds = [None] * len(batch_texts)
        for rank, orig_i in enumerate(orig_indices):
            preds[orig_i] = preds_sorted[rank]
        all_preds.extend(preds)
    return all_preds

def print_report(name, true, pred):
    acc = accuracy_score(true, pred)
    f1  = f1_score(true, pred, average="macro")
    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"{'═'*60}")
    print(f"  Accuracy : {acc:.4f}   F1-macro : {f1:.4f}")
    print(classification_report(true, pred, target_names=["HOF", "NOT"]))

# ── load models ────────────────────────────────────────────────────────────────
print("Loading models...")

lr1 = load_lr(os.path.join(BASE_DIR, "model3", "lr", "model.pkl"))
print("  LR v1   ✓")

lr2 = load_lr(os.path.join(BASE_DIR, "model3", "lr_v2", "model.pkl"))
print("  LR v2   ✓")

lr3 = load_lr(os.path.join(BASE_DIR, "model3", "lr_v3", "model.pkl"))
print("  LR v3   ✓")

lstm1_net, lstm1_w2i, lstm1_i2l, lstm1_thr = load_lstm(
    os.path.join(BASE_DIR, "model3", "lstm", "lstm_model.pt"),
    os.path.join(BASE_DIR, "model3", "lstm", "lstm_vocab.pkl"))
print(f"  LSTM v1 ✓  (threshold={lstm1_thr})")

lstm2_net, lstm2_w2i, lstm2_i2l, lstm2_thr = load_lstm(
    os.path.join(BASE_DIR, "model3", "lstm_v2", "lstm_model.pt"),
    os.path.join(BASE_DIR, "model3", "lstm_v2", "lstm_vocab.pkl"))
print(f"  LSTM v2 ✓  (threshold={lstm2_thr})")

# ── predict ────────────────────────────────────────────────────────────────────
print("\nRunning predictions on Hindi test set...")

lr1_preds   = batch_predict_lr(lr1, texts)
lr2_preds   = batch_predict_lr(lr2, texts)
lr3_preds   = batch_predict_lr(lr3, texts)
lstm1_preds = batch_predict_lstm(lstm1_net, lstm1_w2i, lstm1_i2l, lstm1_thr, texts)
lstm2_preds = batch_predict_lstm(lstm2_net, lstm2_w2i, lstm2_i2l, lstm2_thr, texts)

# ── results ────────────────────────────────────────────────────────────────────
print_report("LR v1   (strip_accents=unicode — destroys Devanagari)", true_labels, lr1_preds)
print_report("LR v2   (strip_accents removed — Devanagari preserved)", true_labels, lr2_preds)
print_report("LR v3   (+ dedicated Devanagari branch, min_df=1)",        true_labels, lr3_preds)
print_report("LSTM v1 (random embeddings, old attention)",              true_labels, lstm1_preds)
print_report("LSTM v2 (FastText embeddings, Bahdanau attention)",       true_labels, lstm2_preds)

# ── summary table ──────────────────────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"{'Model':<42}  {'Acc':>6}  {'F1':>6}")
print(f"{'─'*55}")
for name, pred in [
    ("LR v1  (strip_accents broken)", lr1_preds),
    ("LR v2  (fixed)",                lr2_preds),
    ("LR v3  (+ Devanagari branch)",  lr3_preds),
    ("LSTM v1",                       lstm1_preds),
    ("LSTM v2",                       lstm2_preds),
]:
    print(f"{name:<42}  {accuracy_score(true_labels, pred):.4f}  {f1_score(true_labels, pred, average='macro'):.4f}")
print(f"{'─'*55}")
