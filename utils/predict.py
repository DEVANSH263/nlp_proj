"""
predict.py – Wraps the pre-trained model and vectorizer.

Supports three backends:
  - 'lr'    : TF-IDF + Logistic Regression (sklearn Pipeline)
  - 'lstm'  : BiLSTM + Attention (PyTorch)
  - 'muril' : MuRIL fine-tuned transformer (HuggingFace)

Falls back to keyword heuristic when model files are missing.
"""

import os
import re
import pickle
import logging
from flask import current_app

log = logging.getLogger(__name__)

# Import model-specific preprocessing
from utils.prep2 import preprocess_lr, preprocess_lstm, preprocess_muril

# ---------------------------------------------------------------------------
# Offensive keyword list used by the basic fallback predictor
# ---------------------------------------------------------------------------
OFFENSIVE_KEYWORDS = {
    # English
    "hate", "kill", "stupid", "idiot", "loser", "dumb", "ugly", "trash",
    "disgusting", "pathetic", "worthless", "moron", "freak", "scum",
    "horrible", "awful", "bastard", "asshole", "fuck", "shit", "damn",
    # Transliterated / Hinglish (normalised forms already in English after normalize_text)
    "dirty", "nonsense", "fool", "shameless", "donkey", "bastard",
    "scoundrel", "abuse", "crazy", "coward", "thief",
}

# Phrase boost: rules for multi-word expressions
OFFENSIVE_PHRASES = {
    "get lost": 0.15,
    "shut up": 0.12,
    "go to hell": 0.15,
    "fuck off": 0.20,
    "damn you": 0.10,
    "kill yourself": 0.25,
    "you suck": 0.12,
    "go die": 0.20,
    "drop dead": 0.18,
    "i hate": 0.10,
}

# Optimized threshold for better HOF recall (was 0.5)
CONFIDENCE_THRESHOLD = 0.42


def _load_artifacts():
    """
    Attempt to load model.pkl (and optionally vectorizer.pkl).
    Returns (model, vectorizer) where vectorizer may be None when model is
    a full sklearn Pipeline that handles its own vectorisation.
    Returns (None, None) if model file is missing or corrupted.
    """
    model_path      = current_app.config.get('MODEL_PATH', '')  # Path to saved trained LR model file
    vectorizer_path = current_app.config.get('VECTORIZER_PATH', '')  # Path to saved TF-IDF vectorizer file

    if not os.path.exists(model_path):
        return None, None

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)  # Loaded trained model (sklearn classifier)
    except Exception as e:
        log.error('LR model load error: %s', e)
        return None, None

    # If model is a full Pipeline it handles vectorisation internally
    if hasattr(model, 'named_steps'):
        return model, None

    # Legacy: separate vectorizer needed
    if os.path.exists(vectorizer_path):
        try:
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)  # Loaded TF-IDF vectorizer (converts text to numbers)
            return model, vectorizer
        except Exception:
            pass

    return None, None


def _heuristic_predict(text: str):
    """
    Simple keyword-presence heuristic with phrase boosting.
    Returns (label, confidence) where label is 'HOF' or 'NOT'.
    """
    words = set(text.lower().split())  # All words in text (lowercase)
    matches = words & OFFENSIVE_KEYWORDS  # Offensive words found in text (intersection)
    
    # Start with keyword match confidence
    if matches:
        confidence = min(0.60 + 0.08 * len(matches), 0.98)  # Higher confidence if more offensive words found
    else:
        confidence = 0.85  # Default confidence if no offensive words
    
    # Apply phrase boost if matched
    text_lower = text.lower()  # Lowercase version of original text
    for phrase, boost in OFFENSIVE_PHRASES.items():  # phrase=hateful phrase, boost=how much to increase confidence
        if phrase in text_lower:
            confidence = min(confidence + boost, 1.0)  # Add boost to confidence (capped at 1.0)
            break
    
    # Use optimized threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        return 'HOF', round(confidence, 4)
    return 'NOT', round(1 - confidence if confidence < 0.5 else confidence, 4)


def predict(text: str, model_type: str = 'lr'):
    """
    Full prediction pipeline with model-specific preprocessing.

    Parameters
    ----------
    text       : raw user input
    model_type : 'lr' (Logistic Regression) | 'lstm' (BiLSTM) | 'muril' (MuRIL transformer)
    
    Returns
    -------
    dict with keys:
      - 'cleaned_text'   : preprocessed text
      - 'prediction'     : 'HOF' or 'NOT'
      - 'confidence'     : confidence score (0-1)
      - 'model_used'     : model name
    """
    # Step 1 – Apply model-specific preprocessing
    if model_type == 'lstm':
        cleaned_text = preprocess_lstm(text)  # Cleaned text for LSTM model
    elif model_type == 'muril':
        cleaned_text = preprocess_muril(text)  # Cleaned text for MuRIL model
    else:  # default to 'lr'
        cleaned_text = preprocess_lr(text)  # Cleaned text for Logistic Regression model

    # Step 2 – route to chosen backend
    if model_type == 'lstm':
        label, confidence = _lstm_predict(cleaned_text)  # label='HOF' or 'NOT', confidence=0-1
    elif model_type == 'muril':
        label, confidence = _muril_predict(cleaned_text)  # label='HOF' or 'NOT', confidence=0-1
    else:
        label, confidence = _lr_predict(cleaned_text)  # label='HOF' or 'NOT', confidence=0-1

    model_names = {'lr': 'Logistic Regression', 'lstm': 'BiLSTM', 'muril': 'MuRIL'}  # Display names for models
    return {
        'cleaned_text'   : cleaned_text,
        'prediction'     : str(label).upper(),
        'confidence'     : round(confidence, 4),
        'model_used'     : model_names.get(model_type, model_type),
    }


# ── LR backend ────────────────────────────────────────────────────────────────
def _lr_predict(text: str):
    model, vectorizer = _load_artifacts()  # model=trained classifier, vectorizer=TF-IDF converter
    if model is not None:
        try:
            if vectorizer is None:
                label_raw = model.predict([text])[0]  # Predicted label (0 or 1)
                proba = model.predict_proba([text])[0]  # Probability for each class [P(NOT), P(HOF)]
            else:
                vec   = vectorizer.transform([text])  # vec=text converted to TF-IDF numbers
                label_raw = model.predict(vec)[0]  # Predicted label (0 or 1)
                proba = model.predict_proba(vec)[0]  # Probability for each class [P(NOT), P(HOF)]
            
            # proba[0] = HOF confidence, proba[1] = NOT confidence
            hof_prob = proba[0]  # Probability that text is hateful (HOF)
            
            # Apply optimized threshold instead of default 0.5
            if hof_prob >= CONFIDENCE_THRESHOLD:  # CONFIDENCE_THRESHOLD = 0.42
                label = 'HOF'  # Final prediction: hateful
                confidence = hof_prob  # Use HOF probability as confidence score
            else:
                label = 'NOT'  # Final prediction: not hateful
                confidence = proba[1]  # NOT confidence = probability it's not hateful
            
            return label, float(confidence)
        except Exception as e:
            log.error('LR predict error: %s', e, exc_info=True)
    return _heuristic_predict(text)


# ── LSTM backend ───────────────────────────────────────────────────────────────
_lstm_cache = {}   # module-level cache so we load once per process

def _lstm_predict(text: str):
    """Run inference with the BiLSTM model. Falls back to heuristic on any error."""
    print('[LSTM] _lstm_predict called', flush=True)
    try:
        import torch
        import torch.nn as nn
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        if 'model' not in _lstm_cache:
            model_path = current_app.config.get('LSTM_MODEL_PATH', '')
            vocab_path = current_app.config.get('LSTM_VOCAB_PATH', '')
            if not (os.path.exists(model_path) and os.path.exists(vocab_path)):
                return _heuristic_predict(text)

            with open(vocab_path, 'rb') as f:
                word2idx = pickle.load(f)

            ckpt = torch.load(model_path, map_location='cpu')
            cfg  = ckpt['config']

            # ── v2 architecture: Bahdanau attention + Identity embedding dropout ──
            # Must exactly mirror train_lstm_v2.py or load_state_dict will fail with
            # key mismatches (attn_hidden / attn_context / emb_drop keys).
            class _Attention(nn.Module):
                def __init__(self, h):
                    super().__init__()
                    self.attn_hidden  = nn.Linear(h, h)        # projects hidden states
                    self.attn_context = nn.Linear(h, 1, bias=False)  # scalar score per state
                    self.attn_drop    = nn.Dropout(0.1)
                def forward(self, hs, lengths):
                    energy = torch.tanh(self.attn_hidden(hs))      # (B, T, H)
                    scores = self.attn_context(energy).squeeze(-1)  # (B, T)
                    mask   = torch.arange(hs.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
                    scores = scores.masked_fill(mask, -1e9)
                    w      = self.attn_drop(torch.softmax(scores, dim=1)).unsqueeze(2)
                    return (hs * w).sum(1)

            class _BiLSTM(nn.Module):
                def __init__(self, vs, ed, hd, nl, dp):
                    super().__init__()
                    self.embedding = nn.Embedding(vs, ed, padding_idx=0)
                    self.lstm      = nn.LSTM(ed, hd, num_layers=nl, batch_first=True,
                                            bidirectional=True, dropout=dp if nl > 1 else 0)
                    self.attention = _Attention(hd * 2)
                    self.dropout   = nn.Dropout(dp)
                    self.emb_drop  = nn.Identity()   # v2: no embedding dropout
                    self.fc        = nn.Linear(hd * 2, 1)
                def forward(self, x, lengths):
                    emb    = self.emb_drop(self.embedding(x))
                    packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=True)
                    out, _ = self.lstm(packed)
                    out, _ = pad_packed_sequence(out, batch_first=True)
                    ctx    = self.attention(out, lengths)
                    return self.fc(self.dropout(ctx)).squeeze(1)

            net = _BiLSTM(cfg['vocab_size'], cfg['embed_dim'],
                          cfg['hidden_dim'], cfg['num_layers'], cfg['dropout'])
            net.load_state_dict(ckpt['state_dict'])
            net.eval()

            _lstm_cache['model']     = net
            _lstm_cache['word2idx']  = word2idx
            _lstm_cache['label2idx'] = ckpt.get('label2idx', {'HOF': 1, 'NOT': 0})
            # Use the threshold selected during training (threshold sweep picks best value)
            _lstm_cache['threshold'] = cfg.get('hof_threshold', 0.45)

        net       = _lstm_cache['model']
        word2idx  = _lstm_cache['word2idx']
        label2idx = _lstm_cache['label2idx']
        idx2label = {v: k for k, v in label2idx.items()}
        threshold = _lstm_cache['threshold']
        MAX_LEN   = 120   # must match train_lstm_v2.py MAX_LEN

        # ── v2 tokenizer: regex extraction + nukta normalization + prefix/suffix anchors ──
        _TOK_RE  = re.compile(r'[\u0900-\u097F]+|[a-zA-Z]+')
        _NUKTA   = '\u093c'
        _SW      = {"the","is","in","it","of","and","a","an","are","was","be"}
        word_toks = [t.replace(_NUKTA, '') for t in _TOK_RE.findall(text.lower())]
        word_toks = [t for t in word_toks if t and t not in _SW]
        result    = list(word_toks)
        for w in word_toks:
            if len(w) >= 5:
                result.append(w[:3])
                result.append(w[-3:])
        tokens = result

        # first half + last half truncation (same as train_lstm_v2.py encode())
        if len(tokens) > MAX_LEN:
            half   = MAX_LEN // 2
            tokens = tokens[:half] + tokens[-half:]
        ids    = [word2idx.get(t, word2idx.get('<UNK>', 1)) for t in tokens]
        length = max(len(ids), 1)
        ids   += [0] * (MAX_LEN - len(ids))

        with torch.no_grad():
            x     = torch.tensor([ids],    dtype=torch.long)
            l     = torch.tensor([length], dtype=torch.long)
            prob  = torch.sigmoid(net(x, l)).item()

        label      = idx2label[1] if prob >= threshold else idx2label[0]
        confidence = prob if prob >= 0.5 else 1 - prob
        return label, round(confidence, 4)

    except Exception as e:
        import traceback
        print('[LSTM] ERROR:', e, flush=True)
        traceback.print_exc()
        log.error('LSTM predict error: %s', e, exc_info=True)
        return _heuristic_predict(text)


# ── MuRIL backend ──────────────────────────────────────────────────────────────
_muril_cache = {}

def _muril_predict(text: str):
    """Run inference with fine-tuned MuRIL. Falls back to heuristic on any error."""
    print('[MuRIL] _muril_predict called', flush=True)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if 'model' not in _muril_cache:  # Check if model is already loaded in cache
            model_path = current_app.config.get('MURIL_MODEL_PATH', '')  # Path to MuRIL model directory
            
            if not model_path or not os.path.isdir(model_path):
                print(f'[MuRIL] Model path invalid: {model_path}', flush=True)
                return _heuristic_predict(text)  # Fall back if path invalid
            
            print(f'[MuRIL] Loading from local: {model_path}', flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path)  # tokenizer = converts text to token IDs
            model     = AutoModelForSequenceClassification.from_pretrained(model_path)  # model = fine-tuned transformer
            model.eval()  # Switch to evaluation mode
            _muril_cache['model']     = model  # Store model in cache
            _muril_cache['tokenizer'] = tokenizer  # Store tokenizer in cache
            print('[MuRIL] Model loaded successfully', flush=True)

        model     = _muril_cache['model']  # Retrieved cached model
        tokenizer = _muril_cache['tokenizer']  # Retrieved cached tokenizer
        # run on CPU in Flask (GPU not available in venv)
        enc = tokenizer(text, max_length=128, padding='max_length',
                        truncation=True, return_tensors='pt')  # enc = encoded text as tokens (tensor)
        with torch.no_grad():  # Disable gradient calculation
            logits = model(**enc).logits  # logits = raw prediction scores from model
            proba  = torch.softmax(logits, dim=1)[0]  # proba = probabilities [P(NOT), P(HOF)]

        hof_prob = proba[1].item()  # hof_prob = probability that text is hateful (0-1)
        label      = 'HOF' if hof_prob >= 0.5 else 'NOT'  # label = final prediction
        confidence = hof_prob if hof_prob >= 0.5 else 1 - hof_prob  # confidence = how sure we are
        print(f'[MuRIL] Prediction: {label} ({confidence})', flush=True)
        return label, round(confidence, 4)

    except Exception as e:
        import traceback
        print('[MuRIL] ERROR:', e, flush=True)
        traceback.print_exc()
        log.error('MuRIL predict error: %s', e, exc_info=True)
        return _heuristic_predict(text)

