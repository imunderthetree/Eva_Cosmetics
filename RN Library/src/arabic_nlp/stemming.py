"""
stemming.py — Stemming for Arabic and English.

Provides:
  - Light Arabic stemmer (prefix/suffix stripping)
  - ISRI stemmer wrapper (via NLTK)
  - Porter Stemmer wrapper (via NLTK) for English
  - Snowball stemmer wrapper for Arabic transliterations / English
  - Convenience batch function
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional

# ─────────────────────────────────────────────
# Prefix / suffix tables (Arabic)
# ─────────────────────────────────────────────

PREFIXES = (
    "وال",   # wa-al  (and the)
    "بال",   # bi-al  (with the)
    "كال",   # ka-al  (like the)
    "فال",   # fa-al  (so the)
    "لل",    # li-l   (for the)
    "ال",    # al     (the)
    "و",     # wa     (and)
    "ف",     # fa     (so/then)
    "ب",     # bi     (in/with)
    "ك",     # ka     (like)
    "ل",     # li     (for/to)
    "س",     # sa     (will)
)

SUFFIXES = (
    "ها",    # her / it (fem)
    "هم",    # them (masc)
    "هن",    # them (fem)
    "كما",   # you two
    "كم",    # you (plural masc)
    "نا",    # us / our
    "ية",    # -iya (nisba fem)
    "ات",    # -āt  (fem plural)
    "ون",    # -ūn  (masc plural nom)
    "ين",    # -īn  (masc plural acc/gen)
    "ان",    # -ān  (dual)
    "ه",     # his / it
    "ة",     # ta-marbuta
    "ي",     # my / -ī
    "ك",     # your (sg)
    "ت",     # 2nd/3rd person verb suffix
    "ن",     # -n   (fem verb suffix)
    "ا",     # -ā   (dual / verb suffix)
)


# ─────────────────────────────────────────────
# Light Arabic stemmer
# ─────────────────────────────────────────────

def light_stem(token: str, min_length: int = 3) -> str:
    """
    Light Arabic stemmer: strips one prefix and one suffix.

    Stripping only happens when the remaining stem is at least
    *min_length* characters.

    Parameters
    ----------
    token : str
        A single Arabic word (already normalized / diacritic-free).
    min_length : int
        Minimum stem length to allow stripping (default 3).

    Returns
    -------
    str
        Stemmed token.

    Example
    -------
    >>> light_stem("المدرسة")
    'مدرس'
    >>> light_stem("وكتبوا")
    'كتب'
    """
    if len(token) <= min_length:
        return token

    for prefix in PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= min_length:
            token = token[len(prefix):]
            break

    for suffix in SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= min_length:
            token = token[: -len(suffix)]
            break

    return token


# ─────────────────────────────────────────────
# ISRI stemmer (NLTK)
# ─────────────────────────────────────────────

def get_isri_stemmer() -> Callable[[str], str]:
    """
    Return an ISRI (Information Science Research Institute) Arabic stemmer.

    Requires NLTK::

        pip install nltk

    Returns
    -------
    Callable[[str], str]
        A callable ``stemmer(word) -> stem``.

    Raises
    ------
    RuntimeError
        If NLTK is not installed.

    Example
    -------
    >>> stemmer = get_isri_stemmer()
    >>> stemmer("الكتب")
    'كتب'
    """
    try:
        from nltk.stem import ISRIStemmer
    except ImportError as exc:
        raise RuntimeError(
            "nltk is required for ISRIStemmer.\n"
            "Install with: pip install nltk"
        ) from exc
    _stemmer = ISRIStemmer()
    return _stemmer.stem


# ─────────────────────────────────────────────
# Porter Stemmer (NLTK) — English
# ─────────────────────────────────────────────

def get_porter_stemmer() -> Callable[[str], str]:
    """
    Return a Porter Stemmer for English.

    Requires NLTK::

        pip install nltk

    Returns
    -------
    Callable[[str], str]
        A callable ``stemmer(word) -> stem``.

    Raises
    ------
    RuntimeError
        If NLTK is not installed.

    Example
    -------
    >>> stemmer = get_porter_stemmer()
    >>> stemmer("running")
    'run'
    >>> stemmer("happiness")
    'happi'
    """
    try:
        from nltk.stem import PorterStemmer
    except ImportError as exc:
        raise RuntimeError(
            "nltk is required for PorterStemmer.\n"
            "Install with: pip install nltk"
        ) from exc
    _stemmer = PorterStemmer()
    return _stemmer.stem


# ─────────────────────────────────────────────
# Snowball Stemmer (NLTK) — multi-language
# ─────────────────────────────────────────────

def get_snowball_stemmer(language: str = "english") -> Callable[[str], str]:
    """
    Return a Snowball stemmer for the given language.

    Requires NLTK::

        pip install nltk

    Parameters
    ----------
    language : str
        NLTK Snowball language string, e.g. ``"english"``, ``"arabic"``,
        ``"french"``.  Run ``SnowballStemmer.languages`` to see all options.

    Returns
    -------
    Callable[[str], str]
        A callable ``stemmer(word) -> stem``.

    Example
    -------
    >>> stemmer = get_snowball_stemmer("english")
    >>> stemmer("generously")
    'generous'
    """
    try:
        from nltk.stem import SnowballStemmer
    except ImportError as exc:
        raise RuntimeError(
            "nltk is required for SnowballStemmer.\n"
            "Install with: pip install nltk"
        ) from exc
    _stemmer = SnowballStemmer(language)
    return _stemmer.stem


# ─────────────────────────────────────────────
# Batch stemming
# ─────────────────────────────────────────────

def stem_tokens(
    tokens: Iterable[str],
    stemmer: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """
    Apply a stemmer to a list of tokens.

    Parameters
    ----------
    tokens : Iterable[str]
        Words to stem.
    stemmer : Callable or None
        A function ``(word) -> stem``.  If ``None``, uses
        :func:`light_stem` (Arabic).

    Returns
    -------
    List[str]
        Stemmed tokens.

    Example
    -------
    >>> stem_tokens(["الكتاب", "المدارس"], stemmer=light_stem)
    ['كتاب', 'مدار']
    """
    if stemmer is None:
        stemmer = light_stem
    return [stemmer(token) for token in tokens]
