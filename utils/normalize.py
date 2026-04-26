"""
Transliteration Normalization for Hinglish / code-mixed text.

Maps common Romanised Hindi words → English equivalents using
fuzzy string matching (thefuzz / python-levenshtein).

If thefuzz is not installed the module falls back gracefully
(normalization returns the original text unchanged).
"""

FUZZY_THRESHOLD = 80  # minimum similarity score (0-100)
# Lowered from 85: catches common spelling variants (randii, haraami, etc.)
# without meaningfully increasing false positives on English text

# Strong Hinglish → English / offensive mappings (100+ words)
# Optimized for LR: Using stronger semantic equivalents for better weight
HINGLISH_DICT = {
    # ===== STRONG ABUSE / OFFENSIVE TERMS =====
    "bewakoof": "idiot",          # was "fool" - stronger
    "pagal": "insane",            # was "crazy" - stronger
    "chutiya": "asshole",         # strong abuse
    "bakchod": "asshole",         # strong abuse
    "haraami": "bastard",         # strong
    "harami": "bastard",
    "kamina": "scoundrel",
    "gadha": "stupid",            # was "donkey" - direct meaning
    "ullu": "idiot",              # was "idiot" - keep
    "kutta": "dog scum",          # animal insult
    "kutiya": "bitch",            # strong insult
    "saale": "bastard",           # was "damn" - stronger
    "saala": "bastard",
    "besharam": "shameless",
    "nikamma": "useless",
    "jhatu": "asshole",
    "chodu": "idiot",
    "bhondu": "stupid",
    "randi": "whore",           # FIXED: was "derogatory" — zero weight in embeddings
    "randii": "whore",          # spelling variant
    "raandi": "whore",          # spelling variant
    "raand": "whore",
    "bhikari": "beggar trash",
    "budha": "old fool",
    "darpok": "coward",
    "chor": "thief",
    "ganda": "filthy",
    "ghatiya": "despicable",
    "badmash": "criminal",
    "bakwas": "bullshit",
    "nikal": "get lost",
    "jhatka": "stupid",
    "khara": "dishonest",
    "bekar": "useless",
    "bekaar": "useless",
    "bhad": "damn",
    "bhaad": "hell",
    "nalayak": "incompetent",
    "ganwar": "uncivilized",
    "jhooth": "liar",
    "dhokha": "betrayal",
    "chamcha": "ass kisser",
    # Neutral→neutral replacements (dost→friend, bhai→brother, acha→okay) add noise:
    #   - short tokens ("ja", "na") false-positive fuzzy-match English words
    #   - neutral replacements don't help HOF detection, dilute signal
    #   - duplicate "acha" key (was "okay" then "good") silently overwrote itself
}

try:
    from thefuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False


def normalize_text(text: str) -> str:
    """
    Iterate over each token in *text*.
    If a token fuzzy-matches a Hinglish dict key (score ≥ FUZZY_THRESHOLD),
    replace it with the English equivalent.

    Returns the normalised string.
    Falls back to returning *text* unchanged if thefuzz is unavailable.
    """
    if not FUZZY_AVAILABLE:
        return text  # graceful degradation

    tokens = text.split()
    normalised = []

    for token in tokens:
        best_match = None
        best_score = 0

        # Skip short tokens (≤ 4 chars): fuzzy matching on short strings produces
        # too many false positives. E.g. "gal" → "gali" (86%), "hat" → "hato" (86%),
        # "mar" → "maar" (86%) — all legitimate English words corrupted.
        if len(token) <= 4:
            normalised.append(token)
            continue

        for hin_word in HINGLISH_DICT:
            score = fuzz.ratio(token.lower(), hin_word)
            if score > best_score:
                best_score = score
                best_match = hin_word

        if best_score >= FUZZY_THRESHOLD and best_match:
            normalised.append(HINGLISH_DICT[best_match])
        else:
            normalised.append(token)

    return ' '.join(normalised)
