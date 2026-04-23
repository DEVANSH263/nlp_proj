# prep2.py - Model-Specific Preprocessing Guide

## Overview
`prep2.py` provides three preprocessing functions tailored to each model in your hate speech detection system.

## Quick Usage

```python
from utils.prep2 import preprocess_lr, preprocess_lstm, preprocess_muril

text = "I hate these people!!! 😡 @user #hate www.bad.com"

# For Logistic Regression
lr_text = preprocess_lr(text)
# Result: "i hate these people hate"

# For LSTM
lstm_text = preprocess_lstm(text)
# Result: "i hate these people user hate"

# For MuRIL
muril_text = preprocess_muril(text)
# Result: "i hate these people!!! user #hate"
```

## Why Different Preprocessing?

### 🔴 Logistic Regression (`preprocess_lr`)
- **Most Aggressive**
- Removes: URLs, emojis, numbers, punctuation, light stopwords
- Normalizes repeated chars (goooood → good)
- **Why?** TF-IDF + bag-of-words benefits from reduced vocabulary and noise removal
- **Use case**: Fast, dimensionality reduction important for linear models

### 🟡 LSTM (`preprocess_lstm`)
- **Moderate**
- Removes: URLs, emojis, numbers, punctuation
- Normalizes repeated chars
- **Keeps**: Word order, context, all words (no stopword removal)
- **Why?** RNNs need sequence information and context to understand relationships
- **Use case**: Sequence models where "not good" ≠ "good"

### 🟢 MuRIL (`preprocess_muril`)
- **Minimal**
- Removes: URLs, emojis only
- **Keeps**: Punctuation, numbers, hashtags, repeated chars
- **Why?** Pre-trained transformers are designed to handle raw text; over-processing loses semantic signals
- **Use case**: Transformer models already understand context and can handle noise

## Key Features

✅ **Preserves Critical Words**: "not", "no", "don't", "hate", "love" always kept  
✅ **Handles Devanagari**: Hindi characters preserved in all models  
✅ **Safe Hinglish Handling**: Mixed Hindi-English text stays intact  
✅ **No Heavy Dependencies**: Uses only `re` and `string` modules  
✅ **Batch Processing**: Use `preprocess_batch(texts, model_type='muril')`  

## Batch Processing

```python
from utils.prep2 import preprocess_batch

texts = ["I hate...", "I love..."]

# Process for specific model
processed = preprocess_batch(texts, model_type='lr')  # or 'lstm', 'muril'
```

## What Each Preprocessing Removes

| Element | LR | LSTM | MuRIL |
|---------|----|----|-------|
| URLs | ❌ | ❌ | ❌ |
| Emojis | ❌ | ❌ | ❌ |
| Numbers | ❌ | ❌ | ✅ |
| Punctuation | ❌ | ❌ | ✅ |
| Repeated chars | Normalize | Normalize | ✅ |
| Hashtags | ❌ | ❌ | ✅ |
| Stopwords | Limited | ✅ | ✅ |
| Devanagari | ✅ | ✅ | ✅ |

## Example Comparison

**Original:** `"I NO like them!!! 😠 www.bad.com @user"`

- **LR:** `i no like` (most compact)
- **LSTM:** `i no like them user` (preserves sequence)
- **MuRIL:** `i no like them!!! user` (keeps punctuation)

## Integration with Your App

Update your route or inference code:

```python
from utils.prep2 import preprocess_lr, preprocess_lstm, preprocess_muril

def predict_hate(text, model_type='muril'):
    if model_type == 'lr':
        text = preprocess_lr(text)
    elif model_type == 'lstm':
        text = preprocess_lstm(text)
    elif model_type == 'muril':
        text = preprocess_muril(text)
    
    # ... rest of prediction logic
```

## Testing
Run the file directly to see comparison:
```bash
python utils/prep2.py
```
