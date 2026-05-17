"""
normalization.py — Arabic & Egyptian text normalization utilities.

Covers:
  - Diacritic / tatweel removal
  - Alef / Ya / Hamza / Ta-marbuta normalization
  - Case folding (Arabic has no case; folding targets common variant characters)
  - Soundex encoding for Arabic transliterations
  - Thesaurus lookup via the Egyptian-Arabic dictionary
  - Lemmatization (dictionary-based for Arabic; WordNet-based for English with fallback)
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# ─────────────────────────────────────────────
# Arabic Unicode ranges / constants
# ─────────────────────────────────────────────

AR_DIACRITICS_PATTERN = re.compile(
    r"[\u0617-\u061A\u064B-\u0652\u0657-\u065F\u0670\u06D6-\u06ED]"
)
AR_TATWEEL = "\u0640"

# Character normalization maps
AR_ALEF_MAP = {"\u0622": "\u0627", "\u0623": "\u0627", "\u0625": "\u0627"}
AR_YA_MAP = {"\u0649": "\u064A"}
AR_HAMZA_MAP = {"\u0624": "\u0648", "\u0626": "\u064A"}
AR_TA_MARBUTA_MAP = {"\u0629": "\u0647"}

_FULL_NORM_MAP = str.maketrans(
    {**AR_ALEF_MAP, **AR_YA_MAP, **AR_HAMZA_MAP, **AR_TA_MARBUTA_MAP}
)

# ─────────────────────────────────────────────
# Core normalization
# ─────────────────────────────────────────────

def normalize_arabic(
    text: str,
    *,
    remove_tatweel: bool = True,
    remove_diacritics: bool = True,
    normalize_alef: bool = True,
    normalize_ya: bool = True,
    normalize_hamza: bool = True,
    normalize_ta_marbuta: bool = True,
) -> str:
    """
    Normalize Arabic text by removing tatweel, diacritics, and standardizing
    character variants.

    Parameters
    ----------
    text : str
        Input Arabic text.
    remove_tatweel : bool
        Remove the kashida / tatweel elongation character (U+0640).
    remove_diacritics : bool
        Remove all Arabic diacritical marks (tashkeel).
    normalize_alef : bool
        Map Alef variants (أ إ آ) → plain Alef (ا).
    normalize_ya : bool
        Map Alef-Maqsura (ى) → Ya (ي).
    normalize_hamza : bool
        Map Hamza-on-Waw (ؤ) → Waw, Hamza-on-Ya (ئ) → Ya.
    normalize_ta_marbuta : bool
        Map Ta-Marbuta (ة) → Ha (ه).

    Returns
    -------
    str
        Normalized text.

    Example
    -------
    >>> normalize_arabic("مَدْرَسَةٌ")
    'مدرسه'
    """
    if remove_tatweel:
        text = text.replace(AR_TATWEEL, "")
    if remove_diacritics:
        text = AR_DIACRITICS_PATTERN.sub("", text)

    translation_map: Dict[str, Optional[str]] = {}
    if normalize_alef:
        translation_map.update(AR_ALEF_MAP)
    if normalize_ya:
        translation_map.update(AR_YA_MAP)
    if normalize_hamza:
        translation_map.update(AR_HAMZA_MAP)
    if normalize_ta_marbuta:
        translation_map.update(AR_TA_MARBUTA_MAP)

    if translation_map:
        text = text.translate(str.maketrans(translation_map))

    return text


# ─────────────────────────────────────────────
# Case folding
# ─────────────────────────────────────────────

def case_fold(text: str, language: str = "arabic") -> str:
    """
    Apply case folding to text.

    For Arabic, "case folding" means normalizing all common character variants
    to a canonical base form (calls :func:`normalize_arabic` under the hood).
    For Latin / English text it applies Unicode case folding (equivalent to
    aggressive lowercasing across scripts).

    Parameters
    ----------
    text : str
        Input text.
    language : str
        ``"arabic"`` (default) or ``"english"`` / ``"latin"``.

    Returns
    -------
    str
        Case-folded text.

    Example
    -------
    >>> case_fold("Hello World", language="english")
    'hello world'
    >>> case_fold("أَحْمَد", language="arabic")
    'احمد'
    """
    if language == "arabic":
        return normalize_arabic(text)
    # Unicode case-folding for Latin scripts
    return unicodedata.normalize("NFKC", text).casefold()


# ─────────────────────────────────────────────
# Soundex
# ─────────────────────────────────────────────

# Mapping of Arabic letters to Soundex codes (based on phonetic proximity)
_AR_SOUNDEX_MAP: Dict[str, str] = {
    # Group 1 – labials / labio-dentals
    "ب": "1", "ف": "1", "م": "1",
    # Group 2 – sibilants / dentals
    "ث": "2", "ذ": "2", "ظ": "2",
    # Group 3 – back velars / uvulars
    "ج": "3", "ش": "3", "ي": "3",
    # Group 4 – alveolar stops
    "د": "4", "ت": "4", "ط": "4",
    # Group 5 – sibilant fricatives
    "ز": "5", "س": "5", "ص": "5",
    # Group 6 – pharyngeals / laryngeals
    "ح": "6", "خ": "6", "ه": "6", "ع": "6", "غ": "6",
    # Group 7 – liquids / nasals
    "ل": "7", "ر": "7", "ن": "7",
    # Vowel-like / ignored
    "ا": "0", "و": "0", "ي": "0", "ء": "0", "ق": "8", "ك": "8",
}

# English Soundex table (standard)
_EN_SOUNDEX_MAP: Dict[str, str] = {
    c: code
    for code, chars in [
        ("1", "BFPV"),
        ("2", "CGJKQSXYZ"),
        ("3", "DT"),
        ("4", "L"),
        ("5", "MN"),
        ("6", "R"),
    ]
    for c in chars
}


def soundex_arabic(word: str) -> str:
    """
    Compute a Soundex-style code for an Arabic word.

    The first Arabic letter is kept as-is; subsequent letters are replaced
    with phonetic group codes and consecutive duplicates are collapsed.

    Parameters
    ----------
    word : str
        Arabic word (should be a single token without diacritics).

    Returns
    -------
    str
        A 4-character code like ``"ك600"`` (padded with zeros).

    Example
    -------
    >>> soundex_arabic("كتب")
    'ك300'
    """
    word = normalize_arabic(word)
    # Keep only Arabic letters
    letters = [c for c in word if "\u0600" <= c <= "\u06FF"]
    if not letters:
        return "0000"

    first = letters[0]
    codes: List[str] = []
    prev_code = _AR_SOUNDEX_MAP.get(first, "0")

    for letter in letters[1:]:
        code = _AR_SOUNDEX_MAP.get(letter, "0")
        if code != "0" and code != prev_code:
            codes.append(code)
        prev_code = code

    result = first + "".join(codes)
    result = (result + "000")[:4]
    return result


def soundex_english(word: str) -> str:
    """
    Standard American Soundex for English words.

    Parameters
    ----------
    word : str
        English word.

    Returns
    -------
    str
        4-character Soundex code (e.g., ``"R163"``).

    Example
    -------
    >>> soundex_english("Robert")
    'R163'
    """
    word = word.upper()
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return "0000"

    first = letters[0]
    codes: List[str] = []
    prev_code = _EN_SOUNDEX_MAP.get(first, "")

    for letter in letters[1:]:
        code = _EN_SOUNDEX_MAP.get(letter, "")
        if code and code != prev_code:
            codes.append(code)
        prev_code = code

    result = first + "".join(codes)
    result = (result + "000")[:4]
    return result


# ─────────────────────────────────────────────
# Dictionary-backed thesaurus & lemmatizer
# ─────────────────────────────────────────────

_DICT_PATH = Path(__file__).parent / "egyptian_arabic_dictionary_v2.csv"


_BIDI_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")


def _clean_dict_text(text: str) -> str:
    """
    Normalise a raw dictionary cell:
    1. NFKC – converts Arabic Presentation Forms (FExx) to base characters
    2. Strip bidi control characters
    3. Remove spurious spaces that NFKC may insert between ligature components
    4. Run normalize_arabic (diacritics, tatweel, Alef, etc.)
    """
    text = unicodedata.normalize("NFKC", text)
    text = _BIDI_PATTERN.sub("", text)
    # Remove spaces between Arabic letters (artifact of presentation-form decomposition)
    text = re.sub(r"(?<=[\u0600-\u06FF])\s+(?=[\u0600-\u06FF])", "", text)
    text = normalize_arabic(text.strip())
    return text


@lru_cache(maxsize=1)
def _load_dictionary() -> pd.DataFrame:
    """Load and cache the Egyptian Arabic dictionary."""
    df = pd.read_csv(_DICT_PATH, low_memory=False)
    # Drop rows with no Arabic script
    df = df.dropna(subset=["arabic_script"]).copy()
    df["arabic_script_norm"] = df["arabic_script"].apply(
        lambda x: _clean_dict_text(str(x))
    )
    return df


def thesaurus_lookup(word: str, source: str = "arabic") -> List[str]:
    """
    Look up synonyms / alternate forms for a word using the Egyptian Arabic
    dictionary.

    Parameters
    ----------
    word : str
        The word to look up.
    source : str
        ``"arabic"`` (default) – look up by Arabic script and return alternates.
        ``"english"`` – look up by English meaning and return Arabic equivalents.

    Returns
    -------
    List[str]
        List of synonym / alternate tokens. Empty list if not found.

    Example
    -------
    >>> thesaurus_lookup("ترك", source="arabic")
    ['يترك', 'تارك', ...]
    """
    df = _load_dictionary()
    results: Set[str] = set()

    if source == "arabic":
        norm_word = normalize_arabic(word.strip())
        matches = df[df["arabic_script_norm"] == norm_word]
        for _, row in matches.iterrows():
            alts = str(row.get("arabic_alternates", "") or "")
            for alt in alts.split("|"):
                alt = alt.strip()
                if alt and alt != "nan":
                    results.add(_clean_dict_text(alt))
    else:
        query = word.strip().lower()
        matches = df[
            df["english_meanings"].fillna("").str.lower().str.contains(
                r"\b" + re.escape(query) + r"\b", regex=True
            )
        ]
        for _, row in matches.iterrows():
            script = str(row.get("arabic_script", "") or "").strip()
            if script and script != "nan":
                results.add(_clean_dict_text(script))

    return sorted(results)


def lemmatize_arabic(word: str) -> str:
    """
    Dictionary-based lemmatizer for Arabic / Egyptian words.

    Looks up the word in the Egyptian Arabic dictionary. If found, returns the
    canonical (dictionary headword) form. Otherwise returns the word unchanged.

    Parameters
    ----------
    word : str
        Arabic word (with or without diacritics).

    Returns
    -------
    str
        Lemma (canonical form) or original word if not found.

    Example
    -------
    >>> lemmatize_arabic("مدارس")
    'مدرسة'
    """
    df = _load_dictionary()
    norm_word = normalize_arabic(word.strip())

    # Direct match on normalized arabic_script
    direct = df[df["arabic_script_norm"] == norm_word]
    if not direct.empty:
        return _clean_dict_text(str(direct.iloc[0]["arabic_script"]).strip())

    # Search in arabic_alternates (inflected forms)
    for _, row in df.iterrows():
        alts = str(row.get("arabic_alternates", "") or "")
        for alt in alts.split("|"):
            if _clean_dict_text(alt.strip()) == norm_word:
                return _clean_dict_text(str(row["arabic_script"]).strip())

    return norm_word


def lemmatize_english(word: str) -> str:
    """
    Lemmatize an English word using WordNet (NLTK) if available,
    otherwise fall back to a simple rule-based approach.

    Parameters
    ----------
    word : str
        English word.

    Returns
    -------
    str
        Lemma of the word.

    Example
    -------
    >>> lemmatize_english("running")
    'run'
    """
    try:
        from nltk.stem import WordNetLemmatizer
        _lemmatizer = WordNetLemmatizer()
        return _lemmatizer.lemmatize(word.lower())
    except Exception:
        pass

    # Lightweight rule-based fallback
    w = word.lower()
    rules = [
        (r"ies$", "y"),
        (r"ves$", "f"),
        (r"oes$", "o"),
        (r"sses$", "ss"),
        (r"xes$|zes$|ches$|shes$", lambda m: m.string[: m.start()]),
        (r"(?<=[aeiou][^aeiou])ing$", ""),
        (r"(?<=[^aeiou])ing$", ""),
        (r"(?<=[aeiou][^aeiou])ed$", ""),
        (r"(?<=[^aeiou])ed$", ""),
        (r"s$", ""),
    ]
    for pattern, repl in rules:
        new_w = re.sub(pattern, repl if isinstance(repl, str) else repl, w)
        if new_w != w and len(new_w) >= 3:
            return new_w
    return w
