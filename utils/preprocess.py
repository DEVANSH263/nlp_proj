import re
import string

# Common stopwords to remove (lightweight, no NLTK needed)
STOPWORDS = {
    'the', 'a', 'an', 'is', 'it', 'in', 'on', 'and', 'or', 'to',
    'of', 'for', 'this', 'that', 'with', 'are', 'was', 'be', 'at',
    'by', 'from', 'but', 'not', 'no', 'so', 'if', 'as', 'up', 'do',
}


def preprocess_text(text: str) -> str:
    """
    Clean raw input text:
      1. Lowercase
      2. Remove URLs
      3. Remove @mentions and #hashtags symbols (keep word)
      4. Remove punctuation
      5. Collapse extra whitespace
    Returns cleaned string.
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Remove @mentions (keep username text) and strip # from hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)

    # 4. Remove punctuation (keep spaces)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 5. Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
