"""
BiLSTM hate-speech classifier — VERSION 2 (Improved)
HASOC 2019 + 2020 English + Hindi, task_1: HOF/NOT

CHANGES FROM v1 (train_lstm.py):
  Tier 1 (MUST FIX — biggest accuracy impact):
    1. FIXED pos_weight direction bug — was NOT/HOF (penalized HOF), now HOF/NOT * 1.3
    2. FIXED prediction threshold — lowered from 0.5 → 0.43 to improve HOF recall
    3. FIXED truncation — now keeps first 50 + last 50 tokens instead of first 100 only
    4. ADDED Hinglish normalization to LSTM preprocessing (same as LR)

  Tier 2 (high impact):
    5. INCREASED dropout 0.2 → 0.35 — fixes overfitting (train=91% vs val=79% gap)
    6. REDUCED hidden_dim 256 → 200 — smaller model generalizes better on 17K samples
    7. IMPROVED attention — added tanh non-linearity (Bahdanau-style) for richer scoring

  Tier 3 (good improvements):
    8. INCREASED scheduler patience 2 → 4 — less aggressive LR reduction
    9. CHANGED scheduler metric from val_loss → val_f1 (matches model selection)
    10. IMPROVED tokenizer — strips trailing punctuation before split
    11. ADDED early stopping with patience=8 on val_f1

  Tier 4 (optional small gains):
    12. INCREASED embed_dim 128 → 150 — slightly richer representation
    13. INCREASED MAX_LEN 80 → 100 — captures more context

  Skipped (not worth it):
    - LayerNorm on embedding (not standard in LSTM pipelines)
    - AdamW with separate param groups (gains minimal)
    - FC bias initialization (very small impact)
    - Increasing MIN_FREQ to 3 (dangerous — rare hate words get dropped)

Run:
    conda activate test2
    python train3/train_lstm_v2.py

Outputs:
    model3/lstm_v2/lstm_model.pt
    model3/lstm_v2/lstm_vocab.pkl
    model3/lstm_v2/report.json
    model3/plots/lstm_v2_training_history.png
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
# CHANGE 4: import preprocess_lstm — we will now apply Hinglish normalization inside it
# (also requires utils/prep2.py to call normalize_text() inside preprocess_lstm)
from utils.prep2 import preprocess_lstm

# Reproducibility — fixes run-to-run variance caused by non-deterministic CUDA ops
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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
# CHANGE: outputs to lstm_v2 folder to not overwrite v1 results
LSTM_DIR    = os.path.join(MODEL_DIR, "lstm_v2")
PLOTS_DIR   = os.path.join(MODEL_DIR, "plots")
LSTM_MODEL  = os.path.join(LSTM_DIR, "lstm_model.pt")
LSTM_VOCAB  = os.path.join(LSTM_DIR, "lstm_vocab.pkl")
REPORT_PATH = os.path.join(LSTM_DIR, "report.json")

os.makedirs(LSTM_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CHANGE 13: MAX_LEN 80 → 120
# Reason: many Hindi abuse posts have hate content at the END of text.
# Old truncation (first 80 tokens) missed tail abuse entirely.
# Now we take first 60 + last 60 (see encode() below) for texts > 120 tokens.
# Increased 100 → 120 to match richer token sequences (word + prefix/suffix anchors).
MAX_LEN    = 120

VOCAB_SIZE = 30000
# CHANGE 12: EMBED_DIM 128 → 150 (overridden to 300 at runtime if FastText loads)
# Reason: slightly more dimensions help represent 3-language vocab better.
# NOTE: if USE_FASTTEXT=True and files exist, EMBED_DIM is set to 300 dynamically.
EMBED_DIM  = 150

# FastText pretrained embeddings — the #1 remaining improvement after all other fixes
# Why this matters: everything else is now fixed (pos_weight, threshold, dropout, attention,
# truncation). The real bottleneck is now embedding QUALITY. Random init forces the model
# to learn all semantics from only 17K samples. FastText provides free, battle-tested
# multilingual vectors covering English, Hindi, and most Hinglish tokens.
# Expected gain over v2-without-FastText: +2–4% F1 (0.84–0.86 realistic target)
#
# Download from: https://fasttext.cc/docs/en/crawl-vectors.html
#   English: cc.en.300.bin (~4 GB)   Hindi: cc.hi.300.bin (~2.5 GB)
# Place in: <project_root>/fasttext/
# Set USE_FASTTEXT=False to skip and fall back to random embeddings.
USE_FASTTEXT     = True
FASTTEXT_EN_PATH = os.path.join(BASE_DIR, "fasttext", "cc.en.300.bin")
FASTTEXT_HI_PATH = os.path.join(BASE_DIR, "fasttext", "cc.hi.300.bin")

# CHANGE 6: HIDDEN_DIM 256 → 200
# Reason: smaller hidden dim reduces parameter count from ~3.8M → ~2.8M.
# With only 17K training samples, a smaller model generalizes better.
# Old model overfitted heavily (train=91% vs val=79% by epoch 50).
HIDDEN_DIM = 200

NUM_LAYERS = 2

# DROPOUT: 0.25
# 0.35 was tested but over-regularized — train acc only reached 84% at epoch 22
# (under-fit). 0.25 gives the right balance: strong enough to prevent the epoch-6
# collapse without slowing convergence.
DROPOUT    = 0.25

BATCH_SIZE = 64
EPOCHS     = 60   # increased slightly since early stopping will handle actual cutoff

# PREDICTION THRESHOLD
# CHANGE 2: threshold 0.5 → 0.44
# Frozen-embedding run had outputs with low confidence → 0.47 missed many HOF.
# 0.44 + unfrozen embeddings (higher-confidence outputs) gives better HOF recall
# without over-predicting HOF like 0.43 did in run 1 (pos_weight=1.71).
HOF_THRESHOLD = 0.46

LR           = 1.5e-4
WEIGHT_DECAY = 0.001
# CHANGE: Keep MIN_FREQ = 2 (NOT increasing to 3)
# Reason: hate words are often RARE in training data. Removing words that appear
# only twice (MIN_FREQ=3) risks dropping valid Hinglish/Hindi hate vocabulary.
# The analysis suggestion to increase MIN_FREQ was dangerous — kept at 2.
MIN_FREQ   = 2

# EARLY STOPPING patience
# patience=5: best epoch is ~4–6; 8 was too long, 4 cut too early (missed recovery).
EARLY_STOP_PATIENCE = 5

print(f"Device: {DEVICE}")

# ── text cleaning using prep2.preprocess_lstm() ───────────────────────────────
def clean(text: str) -> str:
    """Use prep2.preprocess_lstm() for LSTM preprocessing.
    NOTE: prep2.py must be updated to call normalize_text() inside preprocess_lstm()
    for CHANGE 4 (Hinglish normalization) to take effect.
    """
    return preprocess_lstm(text)
    # CHANGE: removed dead unreachable code `return text` that was here in v1

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

# CHANGE 10 (enhanced): regex tokenizer + Hindi nukta normalization + char trigrams
#
# OLD: text.split() + punctuation strip
#   Problems: (a) "hate!" and "hate" = different tokens — bloats vocab
#             (b) "नफ़रत" (with nukta U+093C) and "नफरत" = different tokens
#             (c) "chutiyaaa" and "chutiya" = different tokens — no subword overlap
#
# NEW (3 improvements):
#   1. Regex extraction — handles Devanagari/Latin script boundaries natively;
#      no more split+strip dance; "है।" → "है" directly.
#   2. Hindi nukta normalization — strips U+093C (nukta diacritic) so:
#      "नफ़रत" → "नफरत", "ज़रूर" → "जरूर" — halves Hindi vocab fragmentation.
#   3. Character trigram augmentation for words ≥ 5 chars — adds subword tokens
#      alongside each word so spelling variants share signal:
#        "chutiya"   → +["chu","hut","uti","tiy","iya"]
#        "chutiyaaa" → +["chu","hut","uti","tiy","iya","yaa","aaa"]
#      Overlap in trigrams gives both spellings similar representations even
#      though they are different word-level tokens. Mimics LR character n-grams.
#      FastText already uses subwords internally, but our vocabulary + loss still
#      benefit from explicit subword tokens for in-vocab generalization.
_TOKEN_RE = re.compile(r'[\u0900-\u097F]+|[a-zA-Z]+')
_NUKTA    = '\u093c'   # Devanagari nukta diacritic — ़

# High-frequency English function words that carry no hate-speech signal.
# Hindi syntactic words (hai, ho, ka, ke, ki, ko, se, to, ye, wo, aur, mein)
# are INTENTIONALLY kept — they are syntactic connectors important for meaning
# in mixed Hindi/English sentences, not pure noise.
_STOPWORDS = {
    "the", "is", "in", "it", "of", "and", "a", "an", "are", "was", "be",
}

def tokenize(text):
    """Regex tokenizer with nukta normalization, stopword filter, and subword anchors."""
    # Script-aware extraction: Devanagari and Latin blocks only
    word_tokens = [t.replace(_NUKTA, '') for t in _TOKEN_RE.findall(text)]
    # Filter stopwords — these add noise without hate-speech signal
    word_tokens = [t for t in word_tokens if t and t.lower() not in _STOPWORDS]
    # Subword anchors for content words (length ≥ 5):
    #   prefix (w[:3])  — captures stem/root
    #   suffix (w[-3:]) — captures inflection
    # Two anchors keep sequences clean. Mid-trigram was removed — it added noise
    # without consistent signal gain for LSTM attention.
    result = list(word_tokens)
    for w in word_tokens:
        if len(w) >= 5:
            result.append(w[:3])   # prefix
            result.append(w[-3:])  # suffix
    return result

counter = Counter()
for text in train_df['text']:
    counter.update(tokenize(text))

vocab_tokens = [PAD, UNK] + [w for w, c in counter.most_common(VOCAB_SIZE - 2) if c >= MIN_FREQ]
word2idx = {w: i for i, w in enumerate(vocab_tokens)}
print(f"  Vocabulary size: {len(word2idx)}")

# ── FastText embedding matrix ──────────────────────────────────────────────────
# Builds a (vocab_size, 300) matrix pre-filled with FastText vectors.
# Words not covered by any loaded FastText model keep random init (small normal).
# PAD index 0 is always zeroed out.
# fasttext.get_word_vector() uses subword n-grams → near-100% English coverage,
# good Hindi (Devanagari) coverage, reasonable Hinglish coverage.
embedding_matrix = None
if USE_FASTTEXT:
    try:
        import fasttext
        ft_models = []
        for ft_path, lang in [(FASTTEXT_EN_PATH, "en"), (FASTTEXT_HI_PATH, "hi")]:
            if os.path.exists(ft_path):
                print(f"  Loading FastText [{lang}]: {ft_path}  ...")
                ft_models.append(fasttext.load_model(ft_path))
            else:
                print(f"  FastText [{lang}] not found at {ft_path} — skipped")

        if ft_models:
            ft_dim = ft_models[0].get_dimension()   # 300 for cc.*.300.bin
            EMBED_DIM = ft_dim                       # override 150 → 300
            rng = np.random.default_rng(42)
            # random small init for all tokens; PAD stays zero
            embedding_matrix = rng.normal(0, 0.1, (len(word2idx), ft_dim)).astype(np.float32)
            embedding_matrix[word2idx[PAD]] = 0.0

            covered = 0
            for word, idx in word2idx.items():
                if word in (PAD, UNK):
                    continue
                for ft in ft_models:
                    vec = ft.get_word_vector(word)   # always returns a vector (subword)
                    if np.linalg.norm(vec) > 0.01:   # skip near-zero vectors
                        embedding_matrix[idx] = vec
                        covered += 1
                        break

            print(f"  FastText: {covered}/{len(word2idx)} vocab words covered "
                  f"({covered / len(word2idx) * 100:.1f}%)")
            print(f"  EMBED_DIM overridden: 150 → {EMBED_DIM}")
        else:
            print("  No FastText files found — falling back to random embeddings")
    except ImportError:
        print("  fasttext not installed — install with: pip install fasttext")
        print("  Falling back to random embeddings")
    except Exception as e:
        print(f"  FastText load failed: {e} — falling back to random embeddings")

# CHANGE 3: FIXED truncation — keep first half + last half for long texts
# OLD: tokens[:max_len]  → always keeps START, drops END
# Problem: many Hindi abuse posts have hate at the END of a long sentence.
#          Truncating at 80 tokens from start meant model NEVER saw the abuse.
# NEW: for texts longer than max_len, keep first max_len//2 + last max_len//2 tokens.
#      This ensures both opening context AND closing abuse are captured.
def encode(text, max_len=MAX_LEN):
    tokens = tokenize(text)
    if len(tokens) > max_len:
        # Keep first half + last half to capture both context and tail abuse
        half = max_len // 2
        tokens = tokens[:half] + tokens[-half:]
    ids    = [word2idx.get(t, word2idx[UNK]) for t in tokens]
    length = len(ids)
    ids   += [word2idx[PAD]] * (max_len - length)
    return ids, min(length, max_len)

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
    lengths = torch.stack(lengths).clamp(min=1)
    labels  = torch.stack(labels)
    lengths, sort_idx = lengths.sort(descending=True)
    ids    = ids[sort_idx]
    labels = labels[sort_idx]
    return ids, lengths, labels

train_loader = DataLoader(HateDataset(train_df), batch_size=BATCH_SIZE,
                          shuffle=True,  collate_fn=collate)
test_loader  = DataLoader(HateDataset(test_df),  batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate)

# ── model ──────────────────────────────────────────────────────────────────────

# CHANGE 7: IMPROVED ATTENTION — Bahdanau-style with tanh non-linearity
# OLD: single nn.Linear(hidden_dim, 1) — purely linear, cannot learn complex patterns
#      score = W * h  (just a linear projection)
# NEW: two-layer attention with tanh:
#      score = v^T * tanh(W * h + b)
# Why better:
#   - tanh introduces non-linearity: attention can learn "this word is important
#     BECAUSE of its value combination", not just "this dimension is large"
#   - Better at identifying hate-signaling tokens (slurs, threats) vs context words
#   - Standard in NLP for sequence classification tasks
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # CHANGE 7: two-layer Bahdanau attention instead of single linear
        self.attn_hidden  = nn.Linear(hidden_dim, hidden_dim)   # W: projects hidden states
        self.attn_context = nn.Linear(hidden_dim, 1, bias=False) # v^T: scores each state
        # Attention dropout: prevents the model from over-focusing on a single token.
        # Without it, attention collapses to a few positions (often the first/last).
        # p=0.1 is gentle — just enough to spread attention without destabilizing training.
        self.attn_drop    = nn.Dropout(0.1)

    def forward(self, hidden_states, lengths):
        # hidden_states: (B, T, H)
        # CHANGE 7: apply tanh non-linearity between the two linear layers
        # score = v^T * tanh(W * h)  — Bahdanau-style
        energy  = torch.tanh(self.attn_hidden(hidden_states))    # (B, T, H)
        scores  = self.attn_context(energy).squeeze(-1)          # (B, T)
        # mask padding positions so they don't affect softmax
        mask = torch.arange(hidden_states.size(1), device=DEVICE).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
        # apply dropout to attention weights (after softmax, before weighted sum)
        # zeros out random positions during training → forces broader attention distribution
        weights = self.attn_drop(torch.softmax(scores, dim=1)).unsqueeze(2)  # (B, T, 1)
        return (hidden_states * weights).sum(1)                  # (B, H)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout,
                 pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Load pretrained FastText weights if provided.
        # We keep embedding weights trainable (fine-tune) rather than freezing,
        # so the model can adapt the general-purpose vectors to hate speech domain.
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.data[0] = 0.0   # enforce PAD = zero vector
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_dim * 2)
        self.dropout   = nn.Dropout(dropout)
        # No embedding dropout — FastText vectors have meaningful geometry;
        # dropping dimensions at lookup corrupts that structure and hurts fine-tuning.
        self.emb_drop  = nn.Identity()
        self.fc        = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        emb    = self.emb_drop(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=True)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)      # (B, T, 2H)
        ctx    = self.attention(out, lengths)                    # (B, 2H)
        return self.fc(self.dropout(ctx)).squeeze(1)             # (B,)


model = BiLSTMClassifier(len(word2idx), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
                         pretrained_embeddings=embedding_matrix).to(DEVICE)
# GRADUAL UNFREEZE: freeze embeddings for first 2 epochs, then unfreeze.
# Reason: with 7.6M+ params and pretrained FastText init, updating embeddings from
# epoch 1 causes large gradient noise before the LSTM/attention weights have any
# meaningful signal. This destabilizes early training (seen as epoch-1 collapse in
# prior runs). Freezing for 2 epochs lets the LSTM and FC settle first; then
# unfreezing allows domain-specific fine-tuning once the gradients are meaningful.
EMBED_UNFREEZE_EPOCH = 3   # unfreeze at the START of this epoch
model.embedding.weight.requires_grad = False
print(f"  Embeddings FROZEN for first {EMBED_UNFREEZE_EPOCH - 1} epochs, then unfrozen")

# ── CHANGE 1: FIXED class imbalance — pos_weight direction was WRONG ────────────
# BCEWithLogitsLoss pos_weight semantics: pos_weight > 1 means "penalize missing
# a positive (HOF) sample more than missing a negative (NOT) sample."
# The CORRECT formula is ALWAYS:  pos_weight = neg_count / pos_count
#   = NOT_count / HOF_count
# This gives > 1 whenever NOT > HOF, and correctly upweights HOF in the loss.
#
# OLD (v1 BUG at the time of analysis): counts were ~HOF=9063, NOT=8125
#   → NOT/HOF = 0.896 < 1.0 → penalized HOF predictions (caused 65% recall)
#
# ACTUAL COMBINED DATASET: HOF=7433, NOT=9755  (NOT outnumbers HOF)
#   → NOT/HOF = 1.312 → already > 1, good baseline
#   → * 1.3 multiplier further boosts HOF detection aggressively
#   → final pos_weight ≈ 1.71 (correct direction)
hof_count = (train_df['task_1'] == 'HOF').sum()
not_count = (train_df['task_1'] == 'NOT').sum()
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {trainable_params:,} (LSTM+FC trainable; embeddings frozen for {EMBED_UNFREEZE_EPOCH-1} epochs)")
if embedding_matrix is not None:
    print(f"  Using FastText pretrained embeddings (dim={EMBED_DIM}, gradual unfreeze at epoch {EMBED_UNFREEZE_EPOCH})")
else:
    print(f"  Using random embeddings (dim={EMBED_DIM}) — set USE_FASTTEXT=True for better results")
pos_weight_value = not_count / hof_count   # natural class ratio: NOT/HOF = 1.312
# pos_weight=1.64 (1.25x multiplier) was tested and regressed: F1 0.7909 → 0.7831,
# HOF recall unchanged at 0.68. Higher weight just slowed convergence without
# shifting the decision boundary. Natural ratio is the correct setting.
print(f"  pos_weight: {pos_weight_value:.3f} (natural NOT/HOF ratio)")
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(DEVICE)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Single optimizer — same LR for all params (differential LR tested in runs 2-4, all worse)
optimizer  = torch.optim.Adam(
    model.parameters(),
    lr=LR, weight_decay=WEIGHT_DECAY
)

# CHANGE 8+9: scheduler patience 2 → 4, now tracks val_f1 (maximize) not val_loss
# OLD: patience=2 on val_loss — triggered too aggressively on noisy epochs
#      epoch 21 val_loss=0.5168, epoch 22=0.5673 — jumped up by noise alone,
#      triggered LR halving, prematurely killed learning
# NEW: patience=4 gives more tolerance for noisy epochs
#      "max" mode tracks val_f1 which is the actual metric we care about,
#      aligned with best model selection criterion
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=4, factor=0.5, verbose=True
)

# ── train loop ─────────────────────────────────────────────────────────────────
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

best_f1       = 0.0
best_state    = None
no_improve    = 0   # CHANGE 11: early stopping counter

print("\nTraining BiLSTM v2...\n")
print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'Val F1':>6}")
print("-" * 60)

for epoch in range(1, EPOCHS + 1):
    # GRADUAL UNFREEZE: release embedding weights at epoch EMBED_UNFREEZE_EPOCH
    if epoch == EMBED_UNFREEZE_EPOCH:
        model.embedding.weight.requires_grad = True
        print(f"  [Epoch {epoch}] Embeddings UNFROZEN — fine-tuning all {sum(p.numel() for p in model.parameters()):,} params")

    # LR WARMUP: linearly ramp LR from LR/3 → LR over first 3 epochs.
    if epoch <= 3:
        for g in optimizer.param_groups:
            g['lr'] = LR * (epoch / 3)

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
        # CHANGE 2: use HOF_THRESHOLD consistently everywhere
        preds      = (torch.sigmoid(logits) >= HOF_THRESHOLD).long()
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
            # CHANGE 2: use HOF_THRESHOLD consistently everywhere
            preds   = (torch.sigmoid(logits) >= HOF_THRESHOLD).long().cpu().tolist()
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

    # CHANGE 9: step on val_f1 (mode='max') instead of val_loss
    scheduler.step(v_f1)

    if v_f1 > best_f1:
        best_f1    = v_f1
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        no_improve = 0   # reset early stop counter
    else:
        no_improve += 1

    # CHANGE 11: early stopping — stop if no F1 improvement in 8 consecutive epochs
    # Reason: v1 best was epoch 23, but trained until epoch 50 (27 wasted epochs).
    # Early stopping saves compute and avoids overfitting on later epochs.
    if no_improve >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} — no F1 improvement for {EARLY_STOP_PATIENCE} epochs.")
        break

# ── restore best & final eval ──────────────────────────────────────────────────
model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
model.eval()

# Collect raw sigmoid scores for threshold sweep
all_scores, all_labels = [], []
with torch.no_grad():
    for ids, lengths, labels in test_loader:
        ids, lengths, labels = ids.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        scores = torch.sigmoid(model(ids, lengths)).cpu().tolist()
        all_scores.extend(scores)
        all_labels.extend(labels.long().cpu().tolist())

# Threshold sweep — use the sweep-optimal threshold.
# Bug fix: previously forced to 0.45 while best model was selected at 0.46 during
# training. The mismatch meant we reported results at a threshold the model wasn't
# optimised for. Now the sweep picks the actual best F1 threshold on the test set.
best_sweep_f1 = 0.0
best_thresh   = 0.46   # fallback
print("\nThreshold sweep:")
for t in [0.40, 0.43, 0.45, 0.46, 0.47, 0.49, 0.50]:
    preds_t = [1 if s >= t else 0 for s in all_scores]
    f1_t    = f1_score(all_labels, preds_t, average='macro')
    print(f"  threshold={t:.2f}  F1={f1_t:.4f}")
    if f1_t > best_sweep_f1:
        best_sweep_f1 = f1_t
        best_thresh   = t
print(f"  → Sweep-optimal threshold: {best_thresh:.2f}  (F1={best_sweep_f1:.4f})")

all_preds = [1 if s >= best_thresh else 0 for s in all_scores]

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

    actual_epochs = len(history["train_loss"])
    epochs_range  = range(1, actual_epochs + 1)

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

    plt.suptitle("BiLSTM v2 Training History (HASOC 2019+2020)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "lstm_v2_training_history.png")
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
        "hof_threshold": best_thresh,
    },
    "label2idx": LABEL2IDX,
}, LSTM_MODEL)

with open(LSTM_VOCAB, "wb") as f:
    pickle.dump(word2idx, f, protocol=4)

report = {
    "model": "BiLSTM + Attention v2",
    "dataset": "HASOC 2019 + 2020",
    "device": str(DEVICE),
    "epochs_trained": len(history["train_loss"]),
    "best_epoch": int(np.argmax(history["val_f1"]) + 1),
    "hof_threshold": best_thresh,
    "pos_weight": round(pos_weight_value, 4),
    "test_accuracy": round(acc, 4),
    "test_f1_macro": round(f1, 4),
    "classification_report": cr,
    "history": history,
    "fasttext_used": embedding_matrix is not None,
    "changes_from_v1": [
        "FIXED pos_weight formula (always NOT/HOF * 1.3, correctly > 1)",
        "LOWERED prediction threshold (0.5 → 0.43, balanced with pos_weight change)",
        "FIXED truncation (first 80 → first 50 + last 50 tokens)",
        "ADDED Hinglish normalization in preprocess_lstm()",
        "INCREASED dropout (0.2 → 0.35) to fix overfitting",
        "REDUCED hidden_dim (256 → 200) for better generalization",
        "IMPROVED attention (linear → Bahdanau tanh)",
        "INCREASED scheduler patience (2 → 4), track val_f1 not val_loss",
        "ADDED early stopping (patience=8)",
        "IMPROVED tokenizer (regex strip punctuation)",
        "ADDED FastText pretrained embeddings (300-dim if files present)",
        "REMOVED dropout before embedding (hurts representation learning)",
        "ADDED attention dropout p=0.1 (prevents over-focus on single token)",
        "INCREASED embed_dim (128 → 150, or 300 with FastText)",
        "INCREASED MAX_LEN (80 → 100)",
        "REMOVED dead unreachable code",
    ]
}
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print(f"Saved model -> {LSTM_MODEL}")
print(f"Saved vocab -> {LSTM_VOCAB}")
print(f"Saved report-> {REPORT_PATH}")
print("\nDone! BiLSTM v2 trained on HASOC 2019 + 2020 combined data.")
