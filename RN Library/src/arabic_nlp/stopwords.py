"""
stopwords.py — Arabic and English stop-word utilities.

Provides:
  - A built-in list of common Egyptian / Modern Standard Arabic stop words
  - A built-in list of common English stop words
  - A loader for custom stop-word files
  - A filter function
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Set

# ─────────────────────────────────────────────
# Built-in Arabic stop words (Egyptian + MSA)
# ─────────────────────────────────────────────

_ARABIC_STOPWORDS: Set[str] = {
    # Pronouns
    "انا", "احنا", "انت", "انتي", "انتو", "هو", "هي", "هم", "هن",
    "ده", "دي", "دول", "هاده", "هادي", "هادول",
    # Common prepositions
    "في", "على", "من", "الى", "عن", "مع", "ب", "ل", "ك",
    "عند", "لو", "لما", "اما",
    # Articles & conjunctions
    "ال", "و", "ف", "ثم", "او", "لكن", "لان", "اذا",
    # Verbs (to be / auxiliary)
    "كان", "كانت", "كانوا", "يكون", "تكون", "هيكون",
    # Negation
    "لا", "لم", "لن", "ما", "مش", "مو",
    # Question words
    "ايه", "مين", "فين", "امتى", "ازاي", "ليه", "كام",
    "ما", "هل", "كيف", "متى", "اين", "من", "لماذا",
    # Numbers (common)
    "واحد", "اتنين", "تلاتة", "اربعة", "خمسة",
    # Misc high-frequency
    "زي", "برضو", "كمان", "بس", "يعني", "طب", "عشان",
    "دلوقتي", "هنا", "هناك", "اهو", "اهي",
    "جدا", "كثيرا", "قليلا", "كل", "بعض",
    # MSA common stop words
    "هذا", "هذه", "هؤلاء", "ذلك", "تلك", "اولئك",
    "التي", "الذي", "الذين", "اللواتي",
    "قد", "قدم", "منذ", "حتى", "عبر", "خلال",
    "بين", "فوق", "تحت", "امام", "وراء", "بعد", "قبل",
    "نفس", "ذات", "كلا", "كلتا",
}

# ─────────────────────────────────────────────
# Built-in English stop words
# ─────────────────────────────────────────────

_ENGLISH_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "as", "is", "was",
    "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "not", "no", "nor", "so", "yet",
    "both", "either", "neither", "each", "every", "all", "any",
    "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "it", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "what", "which",
    "who", "whom", "when", "where", "why", "how", "there", "here",
    "than", "more", "most", "also", "just", "only", "own", "same",
    "up", "out", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "such", "while",
}


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def get_arabic_stopwords() -> Set[str]:
    """
    Return the built-in set of Egyptian / Modern Standard Arabic stop words.

    Returns
    -------
    Set[str]
        Stop words (already normalised — no diacritics, unified Alef, etc.).

    Example
    -------
    >>> sw = get_arabic_stopwords()
    >>> "في" in sw
    True
    """
    return set(_ARABIC_STOPWORDS)


def get_english_stopwords() -> Set[str]:
    """
    Return the built-in set of English stop words.

    Returns
    -------
    Set[str]
        Lower-cased English stop words.

    Example
    -------
    >>> sw = get_english_stopwords()
    >>> "the" in sw
    True
    """
    return set(_ENGLISH_STOPWORDS)


def load_stopwords(path: Optional[str]) -> Set[str]:
    """
    Load stop words from a plain-text file (one word per line).

    Parameters
    ----------
    path : str or None
        Path to the stop-word file.  Returns an empty set if ``None``.

    Returns
    -------
    Set[str]
        Stop words read from the file.

    Example
    -------
    >>> sw = load_stopwords("/path/to/my_stopwords.txt")
    """
    if not path:
        return set()

    stopwords: Set[str] = set()
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            word = line.strip()
            if word and not word.startswith("#"):
                stopwords.add(word)
    return stopwords


def remove_stopwords(
    tokens: Iterable[str],
    stopwords: Optional[Set[str]] = None,
    language: str = "arabic",
) -> List[str]:
    """
    Remove stop words from a token list.

    Parameters
    ----------
    tokens : Iterable[str]
        List of word tokens.
    stopwords : Set[str] or None
        Custom stop-word set.  If ``None``, the built-in set for
        *language* is used.
    language : str
        ``"arabic"`` (default) or ``"english"`` — selects which
        built-in stop-word set to use when *stopwords* is ``None``.

    Returns
    -------
    List[str]
        Tokens with stop words removed.

    Example
    -------
    >>> remove_stopwords(["في", "البيت", "جميل"])
    ['البيت', 'جميل']
    """
    if stopwords is None:
        stopwords = (
            get_arabic_stopwords()
            if language == "arabic"
            else get_english_stopwords()
        )
    return [t for t in tokens if t not in stopwords]
