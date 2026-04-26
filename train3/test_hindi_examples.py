"""
Quick side-by-side test of all 4 models on real Hindi/Hinglish examples.
Runs standalone (no Flask app context needed).

Usage:
    conda activate test2
    python train3/test_hindi_examples.py
"""

import os, sys, re, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prep2 import preprocess_lr, preprocess_lstm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── test sentences ─────────────────────────────────────────────────────────────
EXAMPLES = [
    # clear HOF
    ("तुम बहुत बेवकूफ हो, चुप रहो",                           "HOF"),  # You are very stupid, keep quiet
    ("यह मुसलमान लोग देश के दुश्मन हैं",                      "HOF"),  # These Muslims are enemies of the country
    ("chutiye band kar apna munh",                              "HOF"),  # shut your mouth (Hinglish abuse)
    ("randi ki aulaad hai ye sab",                              "HOF"),  # abusive Hinglish
    ("इन हरामियों को देश से निकालो",                           "HOF"),  # throw these bastards out of the country
    ("tumhari maa ki aankh, sale kutte",                        "HOF"),  # strong Hinglish abuse
    # clear NOT
    ("आज मौसम बहुत अच्छा है",                                  "NOT"),  # weather is very good today
    ("मुझे क्रिकेट देखना बहुत पसंद है",                        "NOT"),  # I love watching cricket
    ("please help me with this problem",                        "NOT"),
    ("yaar kal movie dekhne chalein?",                          "NOT"),  # dude shall we watch a movie tomorrow?
    # ambiguous / tricky
    ("Modi ji ne bahut acha kaam kiya hai desh ke liye",        "NOT"),  # political but NOT
    ("ye log kuch bhi nahi karte sirf baat karte hain",         "NOT"),  # criticism but NOT
]

# ── LR model loader ────────────────────────────────────────────────────────────
def load_lr(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict_lr(pipeline, text):
    cleaned = preprocess_lr(text)
    prob    = pipeline.predict_proba([cleaned])[0]
    classes = list(pipeline.classes_)
    hof_p   = prob[classes.index("HOF")]
    label   = "HOF" if hof_p >= 0.5 else "NOT"
    conf    = hof_p if label == "HOF" else 1 - hof_p
    return label, round(conf, 3)

# ── LSTM model loader ──────────────────────────────────────────────────────────
class _Attention(nn.Module):
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

# v1 attention: single linear (old architecture)
class _AttentionV1(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.attn = nn.Linear(h, 1)
    def forward(self, hs, lengths):
        scores = self.attn(hs).squeeze(-1)
        mask   = torch.arange(hs.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
        w      = torch.softmax(scores, dim=1).unsqueeze(2)
        return (hs * w).sum(1)

class _BiLSTM(nn.Module):
    def __init__(self, vs, ed, hd, nl, dp, v1=False):
        super().__init__()
        self.embedding = nn.Embedding(vs, ed, padding_idx=0)
        self.lstm      = nn.LSTM(ed, hd, num_layers=nl, batch_first=True,
                                 bidirectional=True, dropout=dp if nl > 1 else 0)
        self.attention = _AttentionV1(hd * 2) if v1 else _Attention(hd * 2)
        self.dropout   = nn.Dropout(dp)
        self.emb_drop  = nn.Identity()
        self.fc        = nn.Linear(hd * 2, 1)
    def forward(self, x, lengths):
        emb    = self.emb_drop(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=True)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        ctx    = self.attention(out, lengths)
        return self.fc(self.dropout(ctx)).squeeze(1)

def load_lstm(model_path, vocab_path):
    with open(vocab_path, "rb") as f:
        word2idx = pickle.load(f)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    # auto-detect architecture from checkpoint keys
    is_v1 = "attention.attn.weight" in ckpt["state_dict"]
    net   = _BiLSTM(cfg["vocab_size"], cfg["embed_dim"],
                    cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"], v1=is_v1)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    label2idx = ckpt.get("label2idx", {"HOF": 1, "NOT": 0})
    threshold = cfg.get("hof_threshold", 0.5)
    return net, word2idx, {v: k for k, v in label2idx.items()}, threshold

_TOK_RE = re.compile(r'[\u0900-\u097F]+|[a-zA-Z]+')
_NUKTA  = '\u093c'
_SW     = {"the","is","in","it","of","and","a","an","are","was","be"}

def _tokenize_v2(text):
    word_toks = [t.replace(_NUKTA, '') for t in _TOK_RE.findall(text.lower())]
    word_toks = [t for t in word_toks if t and t not in _SW]
    result    = list(word_toks)
    for w in word_toks:
        if len(w) >= 5:
            result.append(w[:3]); result.append(w[-3:])
    return result

def predict_lstm(net, word2idx, idx2label, threshold, text, max_len=120):
    cleaned = preprocess_lstm(text)
    tokens  = _tokenize_v2(cleaned)
    if len(tokens) > max_len:
        h = max_len // 2
        tokens = tokens[:h] + tokens[-h:]
    ids    = [word2idx.get(t, word2idx.get("<UNK>", 1)) for t in tokens]
    length = max(len(ids), 1)
    ids   += [0] * (max_len - len(ids))
    with torch.no_grad():
        x   = torch.tensor([ids],    dtype=torch.long)
        l   = torch.tensor([length], dtype=torch.long)
        prob = torch.sigmoid(net(x, l)).item()
    label = idx2label[1] if prob >= threshold else idx2label[0]
    conf  = prob if prob >= 0.5 else 1 - prob
    return label, round(conf, 3)

# ── load all 4 models ──────────────────────────────────────────────────────────
print("Loading models...\n")

# LR v1
lr1_path = os.path.join(BASE_DIR, "model3", "lr", "model.pkl")
lr1 = load_lr(lr1_path) if os.path.exists(lr1_path) else None
print(f"LR v1   : {'✓ loaded' if lr1 else '✗ not found  (' + lr1_path + ')'}")

# LR v2
lr2_path = os.path.join(BASE_DIR, "model3", "lr_v2", "model.pkl")
lr2 = load_lr(lr2_path) if os.path.exists(lr2_path) else None
print(f"LR v2   : {'✓ loaded' if lr2 else '✗ not found  (' + lr2_path + ')'}")

# LSTM v1
lstm1_model = os.path.join(BASE_DIR, "model3", "lstm", "lstm_model.pt")
lstm1_vocab = os.path.join(BASE_DIR, "model3", "lstm", "lstm_vocab.pkl")
if os.path.exists(lstm1_model) and os.path.exists(lstm1_vocab):
    try:
        lstm1_net, lstm1_w2i, lstm1_i2l, lstm1_thr = load_lstm(lstm1_model, lstm1_vocab)
        print(f"LSTM v1 : ✓ loaded  (threshold={lstm1_thr})")
    except Exception as e:
        lstm1_net = None; print(f"LSTM v1 : ✗ load failed — {e}")
else:
    lstm1_net = None; print(f"LSTM v1 : ✗ not found  ({lstm1_model})")

# LSTM v2
lstm2_model = os.path.join(BASE_DIR, "model3", "lstm_v2", "lstm_model.pt")
lstm2_vocab = os.path.join(BASE_DIR, "model3", "lstm_v2", "lstm_vocab.pkl")
if os.path.exists(lstm2_model) and os.path.exists(lstm2_vocab):
    try:
        lstm2_net, lstm2_w2i, lstm2_i2l, lstm2_thr = load_lstm(lstm2_model, lstm2_vocab)
        print(f"LSTM v2 : ✓ loaded  (threshold={lstm2_thr})")
    except Exception as e:
        lstm2_net = None; print(f"LSTM v2 : ✗ load failed — {e}")
else:
    lstm2_net = None; print(f"LSTM v2 : ✗ not found  ({lstm2_model})")

# ── run predictions ────────────────────────────────────────────────────────────
print()
print("─" * 120)
print(f"{'Text':<52}  {'True':<4}  {'LR-v1':<12}  {'LR-v2':<12}  {'LSTM-v1':<12}  {'LSTM-v2':<12}")
print("─" * 120)

for text, true_label in EXAMPLES:
    display = text[:48] + ".." if len(text) > 50 else text

    lr1_res  = f"{predict_lr(lr1,  text)[0]} ({predict_lr(lr1,  text)[1]:.2f})"   if lr1       else "N/A"
    lr2_res  = f"{predict_lr(lr2,  text)[0]} ({predict_lr(lr2,  text)[1]:.2f})"   if lr2       else "N/A"
    l1_res   = f"{predict_lstm(lstm1_net, lstm1_w2i, lstm1_i2l, lstm1_thr, text)[0]} ({predict_lstm(lstm1_net, lstm1_w2i, lstm1_i2l, lstm1_thr, text)[1]:.2f})" if lstm1_net else "N/A"
    l2_res   = f"{predict_lstm(lstm2_net, lstm2_w2i, lstm2_i2l, lstm2_thr, text)[0]} ({predict_lstm(lstm2_net, lstm2_w2i, lstm2_i2l, lstm2_thr, text)[1]:.2f})" if lstm2_net else "N/A"

    print(f"{display:<52}  {true_label:<4}  {lr1_res:<12}  {lr2_res:<12}  {l1_res:<12}  {l2_res:<12}")

print("─" * 120)
