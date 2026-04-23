"""
Transliteration Normalization for Hinglish / code-mixed text.

Maps common Romanised Hindi words → English equivalents using
fuzzy string matching (thefuzz / python-levenshtein).

If thefuzz is not installed the module falls back gracefully
(normalization returns the original text unchanged).
"""

FUZZY_THRESHOLD = 85  # minimum similarity score (0-100)

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
    "randi": "derogatory",
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
    "masoom": "innocent",
    "sasta": "cheap",
    "jhooth": "liar",
    "dhokha": "betrayal",
    "chamcha": "ass kisser",
    "thullu": "idiot",
    "gandoo": "dirty bastard",
    "pant": "coward",
    "hilaa": "shake off",
    "susti": "lazy",
    "dhongi": "fraud",
    "chalak": "cunning",
    "chichu": "sissy",
    "budhapa": "old age shame",
    "beksufi": "shameless",
    "beshwami": "shameless",
    "chamkila": "slut",
    "randa": "whore",
    "thand": "cold shoulder",
    "thandia": "betrayer",
    "khichdi": "mixture mess",
    "mungeya": "poor trash",
    "chikna": "slick fraud",
    "chipku": "sticky leech",
    "khechar": "vagrant",
    "naukar": "servant trash",
    
    # ===== MILD ABUSIVE / NEGATIVE =====
    "maro": "hit",
    "maar": "hit",
    "maar pakad": "arrest",
    "jhappad": "slap",
    "chakka": "faggot",
    "gali": "abuse",
    "chup": "shut up",
    "shup": "shut up",
    "choop": "shut",
    "band karo": "stop it",
    "niklo": "get out",
    "ja": "go away",
    "jao": "go away",
    "bhag": "run away",
    "bhaag": "flee",
    "hato": "move aside",
    "shor": "noise",
    "chilla": "yell",
    "chillao": "yell loud",
    "seena": "chest beat",
    "dhokha": "betrayal",
    "bewahaal": "ruined",
    "barbaadi": "destruction",
    "tabahi": "disaster",
    
    # ===== POSITIVE / NEUTRAL (for context) =====
    "badhiya": "good",
    "mast": "excellent",
    "jhakaas": "awesome",
    "shabash": "well done",
    "acha": "okay",
    "accha": "okay",
    "theek": "fine",
    "shukriya": "thanks",
    "acha": "good",
    "pyaar": "love",
    "dost": "friend",
    "bhai": "brother",
    "behan": "sister",
    "mummy": "mom",
    "papa": "dad",
    "nana": "grandpa",
    "nani": "grandma",
    "shaadi": "wedding",
    "khushi": "happiness",
    "sukh": "peace",
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
