import re
import string

# Import Hinglish normalization (for LR only)
try:
    from utils.normalize import normalize_text
    NORMALIZE_AVAILABLE = True
except ImportError:
    NORMALIZE_AVAILABLE = False

"""
Model-Specific Preprocessing Pipeline for Multilingual Hate Speech Detection

KEY DESIGN DECISIONS:
======================

1. LOGISTIC REGRESSION (preprocess_lr):
   - MOST aggressive: light stopword removal + char normalization + Hinglish → English
   - WHY: TF-IDF + bag-of-words model benefits from reduced vocab + standardized words
   - Includes fuzzy Hinglish normalization (ganda→dirty, chutiya→idiot, etc.)
   - Fast, dimensionality reduction important for linear models
   - Aggressive but safe (preserves "not", "no", "don't")
   
2. LSTM (preprocess_lstm):
   - MODERATE: minimal stopword removal, preserve sequence
   - WHY: RNN models need context + word order matters
   - Repeated char normalization helps with hateful typos
   - Avoid aggressive stopword removal that loses meaning
   
3. MuRIL (preprocess_muril):
   - MINIMAL: almost no cleaning, keep text natural
   - WHY: Pre-trained transformers handle raw text better
   - Tokenizer + subword pieces handle most issues
   - Over-preprocessing loses semantic information

GENERAL HANDLING:
- Emojis: Remove (low signal for hate speech detection)
- Numbers: Remove (mostly noise in this task)
- URLs: Remove (not hate speech in URL)
- Mentions: Remove @ but keep username
- Hashtags: Keep (often indicate sentiment)
- Repeated chars: Normalize for LR/LSTM only
- Devanagari: PRESERVE in all models
- Hinglish: PRESERVE (common in Indian context)
"""

# ============================================================================
# LIGHTWEIGHT CUSTOM STOPWORDS (not using NLTK, avoiding critical words)
# ============================================================================

# Words to remove ONLY for LR (aggressive cleanup)
LR_STOPWORDS = {
    'the', 'a', 'an', 'is', 'it', 'in', 'on', 'and', 'or', 'to',
    'of', 'for', 'this', 'that', 'with', 'are', 'was', 'be', 'at',
    'by', 'from', 'but', 'so', 'if', 'as', 'up', 'do', 'have', 'has',
    'he', 'she', 'you', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'am', 'been', 'being', 'can', 'could', 'would', 'should', 'will',
}

# Words CRITICAL to hate speech detection - NEVER remove
CRITICAL_WORDS = {
    'not', 'no', 'don', 'didn', 'won', 'isn', 'aren', 'shouldn',
    'wouldn', 'couldn', 'hate', 'love', 'bad', 'good', 'like', 'dislike',
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def remove_urls(text: str) -> str:
    """Remove URLs and domains."""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\S+\.(com|org|net|in|io)\S*', '', text)
    return text


def remove_emojis(text: str) -> str:
    """Remove emojis and emoticons."""
    # Unicode emoji ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def remove_numbers(text: str) -> str:
    """Remove numbers (mostly noise for hate speech detection)."""
    text = re.sub(r'\d+', '', text)
    return text


def normalize_repeated_chars(text: str, threshold: int = 3) -> str:
    """
    Normalize repeated characters: 'goooood' -> 'good'
    Only normalizes if 3+ consecutive identical chars (configurable).
    """
    # Pattern: any character repeated 3+ times
    pattern = r'(\w)\1{' + str(threshold - 1) + ',}'
    text = re.sub(pattern, r'\1\1', text)  # Keep only 2
    return text


def clean_mentions(text: str) -> str:
    """Remove @ symbol but keep the username text."""
    text = re.sub(r'@', '', text)
    return text


def clean_hashtags(text: str) -> str:
    """Keep hashtags but remove # symbol."""
    text = re.sub(r'#', '', text)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces to single space."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_punctuation_safe(text: str, keep_chars: str = "") -> str:
    """
    Remove punctuation, but optionally keep certain chars.
    By default keeps spaces and Devanagari punctuation.
    """
    punct_to_remove = string.punctuation
    for char in keep_chars:
        punct_to_remove = punct_to_remove.replace(char, '')
    
    text = text.translate(str.maketrans('', '', punct_to_remove))
    return text


def remove_stopwords(text: str, stopwords: set) -> str:
    """Remove stopwords from text."""
    words = text.split()
    words = [w for w in words if w not in stopwords]
    return ' '.join(words)


# ============================================================================
# MODEL-SPECIFIC PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_lr(text: str) -> str:
    """
    LOGISTIC REGRESSION PREPROCESSING (Most Aggressive)
    
    Strategy: Reduce dimensionality while keeping semantic meaning.
    TF-IDF works best with normalized vocab and reduced noise.
    
    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove emojis
      4. Remove numbers
      5. Normalize repeated chars (e.g., "goooood" -> "good")
      6. Remove mentions @ and clean hashtags #
      7. Remove punctuation (safe for Devanagari)
      8. Normalize Hinglish words (e.g., "ganda" -> "dirty")
      9. Remove light stopwords (keep critical words)
      10. Collapse whitespace
    
    Returns:
      Cleaned, deduplicated text ready for TF-IDF vectorization.
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = remove_urls(text)
    
    # 3. Remove emojis
    text = remove_emojis(text)
    
    # 4. Remove numbers
    text = remove_numbers(text)
    
    # 5. Normalize repeated chars
    text = normalize_repeated_chars(text, threshold=3)
    
    # 6. Clean mentions and hashtags
    text = clean_mentions(text)
    text = clean_hashtags(text)
    
    # 7. Remove punctuation (except space)
    text = remove_punctuation_safe(text, keep_chars=' ')
    
    # 8. Normalize Hinglish words to English equivalents
    if NORMALIZE_AVAILABLE:
        text = normalize_text(text)
    
    # 9. Remove light stopwords (but keep critical ones)
    words = text.split()
    words = [
        w for w in words 
        if w not in LR_STOPWORDS or w in CRITICAL_WORDS
    ]
    text = ' '.join(words)
    
    # 10. Collapse whitespace
    text = collapse_whitespace(text)
    
    return text


def preprocess_lstm(text: str) -> str:
    """
    LSTM PREPROCESSING (Moderate)
    
    Strategy: Preserve word order and context while cleaning noise.
    BiLSTM needs sequence information and context for good performance.
    
    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove emojis
      4. Remove numbers
      5. Normalize repeated chars (typos matter for sentiment)
      6. Remove mentions @ and clean hashtags #
      7. Remove punctuation (safe for Devanagari)
      8. Collapse whitespace
    
    NOTE: NO aggressive stopword removal. Context and order matter.
    
    Returns:
      Cleaned text preserving sequence and semantic context.
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = remove_urls(text)
    
    # 3. Remove emojis
    text = remove_emojis(text)
    
    # 4. Remove numbers
    text = remove_numbers(text)
    
    # 5. Normalize repeated chars (but less aggressively than LR)
    text = normalize_repeated_chars(text, threshold=3)
    
    # 6. Clean mentions and hashtags
    text = clean_mentions(text)
    text = clean_hashtags(text)
    
    # 7. Remove punctuation
    text = remove_punctuation_safe(text, keep_chars=' ')
    
    # 8. Collapse whitespace
    text = collapse_whitespace(text)
    
    return text


def preprocess_muril(text: str) -> str:
    """
    MuRIL PREPROCESSING (Minimal)
    
    Strategy: Keep text as natural as possible.
    Pre-trained transformers (MURIL, BERT, etc.) have learned to handle
    raw text, punctuation, and noise. Over-cleaning removes semantic signals.
    
    Steps:
      1. Lowercase (slight normalization)
      2. Remove URLs (not hate speech)
      3. Remove emojis (low signal)
      4. Remove mentions @ (keep username)
      5. Keep punctuation (adds emotional signal)
      6. Keep numbers (some context)
      7. Keep hashtags (adds context)
      8. Keep repeated chars (transformer can handle)
      9. Collapse extreme whitespace only
    
    NOTE: Minimal cleaning. Tokenizer will handle most issues.
    NO stopword removal. Transformer handles context.
    
    Returns:
      Lightly cleaned text with natural characteristics preserved.
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = remove_urls(text)
    
    # 3. Remove emojis only
    text = remove_emojis(text)
    
    # 4. Clean mentions (remove @ but keep word)
    text = clean_mentions(text)
    
    # NOTE: Keep punctuation, numbers, hashtags, repeated chars for transformer
    
    # 5. Collapse extreme whitespace only
    text = re.sub(r'\s{2,}', ' ', text).strip()
    
    return text


# ============================================================================
# BATCH PROCESSING UTILITY
# ============================================================================

def preprocess_batch(texts: list, model_type: str = 'muril') -> list:
    """
    Preprocess a batch of texts using specified model's preprocessing.
    
    Args:
        texts: List of text strings
        model_type: 'lr', 'lstm', or 'muril'
    
    Returns:
        List of preprocessed texts
    """
    if model_type == 'lr':
        return [preprocess_lr(text) for text in texts]
    elif model_type == 'lstm':
        return [preprocess_lstm(text) for text in texts]
    elif model_type == 'muril':
        return [preprocess_muril(text) for text in texts]
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'lr', 'lstm', or 'muril'")


# ============================================================================
# TESTING / EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example text with various challenging elements
    test_texts = [
        "I hate these people!!! 😡😡😡 Check: www.example.com #hate",
        "This is goooood and I love it!! 💯 @user mention here",
        "आपको नहीं पसंद है??? और मुझे भी नहीं! 😠🔥",
        "Don't listen to them, they're not good people",
        "I NO LIKE THEM111 🤦‍♂️🤦‍♂️ https://badsite.com",
    ]
    
    print("=" * 80)
    print("MODEL-SPECIFIC PREPROCESSING COMPARISON")
    print("=" * 80)
    
    for text in test_texts:
        print(f"\n📝 Original: {text}")
        print(f"🔴 LR:      {preprocess_lr(text)}")
        print(f"🟡 LSTM:    {preprocess_lstm(text)}")
        print(f"🟢 MuRIL:   {preprocess_muril(text)}")
        print("-" * 80)
