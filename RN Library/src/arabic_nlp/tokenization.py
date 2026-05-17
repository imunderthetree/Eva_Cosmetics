"""
tokenization.py — Arabic / Egyptian text tokenization.

Provides:
  - Arabic-script tokenizer using a Unicode regex
  - Sentence tokenizer (punctuation-based)
  - Clitic (prefix/suffix) splitter built on the stemming prefix/suffix tables
  - Mixed Arabic-English tokenizer
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from .normalization import normalize_arabic
from .stemming import PREFIXES, SUFFIXES

# ─────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────

# Matches runs of Arabic letters and Arabic-Indic digits
AR_TOKEN_PATTERN = re.compile(r"[\u0621-\u064A\u0660-\u0669\u0671-\u06D3]+")

# Matches Latin words
EN_TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")

# Sentence boundary punctuation (Arabic + Latin)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟।\n])\s+")


# ─────────────────────────────────────────────
# Clitic splitting
# ─────────────────────────────────────────────

def _split_token_clitics(token: str, min_length: int = 2) -> List[str]:
    """
    Split a single token into prefix, stem, and suffix parts.

    Only one prefix and one suffix are stripped per token to avoid
    over-segmentation.
    """
    remaining = token
    parts: List[str] = []

    for prefix in PREFIXES:
        if remaining.startswith(prefix) and len(remaining) - len(prefix) >= min_length:
            parts.append(prefix)
            remaining = remaining[len(prefix):]
            break

    for suffix in SUFFIXES:
        if remaining.endswith(suffix) and len(remaining) - len(suffix) >= min_length:
            core = remaining[: -len(suffix)]
            if core:
                parts.append(core)
            parts.append(suffix)
            return parts

    if parts:
        parts.append(remaining)
        return parts

    return [token]


def split_clitic_tokens(tokens: Sequence[str], min_length: int = 2) -> List[str]:
    """
    Apply clitic splitting to a list of tokens.

    Parameters
    ----------
    tokens : Sequence[str]
        Pre-tokenized words.
    min_length : int
        Minimum stem length after stripping a clitic.

    Returns
    -------
    List[str]
        Tokens with clitics separated.

    Example
    -------
    >>> split_clitic_tokens(["وللمدرسة"])
    ['و', 'للمدرسة']   # prefix 'و' stripped first
    """
    result: List[str] = []
    for token in tokens:
        result.extend(_split_token_clitics(token, min_length=min_length))
    return result


# ─────────────────────────────────────────────
# Arabic tokenizer
# ─────────────────────────────────────────────

def tokenize_arabic(
    text: str,
    normalize_text: bool = True,
    split_clitics: bool = False,
    min_clitic_length: int = 2,
) -> List[str]:
    """
    Tokenize Arabic text into a list of word tokens.

    Parameters
    ----------
    text : str
        Raw Arabic text.
    normalize_text : bool
        Apply :func:`~arabic_nlp.normalization.normalize_arabic` before
        tokenization.
    split_clitics : bool
        Further split each token into prefix/stem/suffix parts.
    min_clitic_length : int
        Minimum stem length when splitting clitics.

    Returns
    -------
    List[str]
        List of tokens.

    Example
    -------
    >>> tokenize_arabic("ذهب الولد إلى المدرسة")
    ['ذهب', 'الولد', 'الى', 'المدرسه']
    """
    if normalize_text:
        text = normalize_arabic(text)
    tokens = AR_TOKEN_PATTERN.findall(text)
    if split_clitics:
        tokens = split_clitic_tokens(tokens, min_length=min_clitic_length)
    return tokens


# ─────────────────────────────────────────────
# Sentence tokenizer
# ─────────────────────────────────────────────

def tokenize_sentences(text: str) -> List[str]:
    """
    Split text into sentences using punctuation boundaries.

    Handles Arabic (؟ .) and Latin (. ! ?) sentence-ending characters.

    Parameters
    ----------
    text : str
        Multi-sentence text.

    Returns
    -------
    List[str]
        List of sentence strings.

    Example
    -------
    >>> tokenize_sentences("الجو جميل. هل تحب الطقس؟ أنا أحبه.")
    ['الجو جميل.', 'هل تحب الطقس؟', 'أنا أحبه.']
    """
    sentences = SENTENCE_BOUNDARY.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ─────────────────────────────────────────────
# Mixed Arabic-English tokenizer
# ─────────────────────────────────────────────

def tokenize_mixed(
    text: str,
    normalize_arabic_tokens: bool = True,
    split_clitics: bool = False,
) -> List[str]:
    """
    Tokenize text that contains both Arabic and English words.

    Arabic runs are tokenized with :func:`tokenize_arabic`; Latin runs
    with a simple word-regex. The original left-to-right order is preserved.

    Parameters
    ----------
    text : str
        Text that may contain Arabic and/or Latin words.
    normalize_arabic_tokens : bool
        Normalize Arabic tokens.
    split_clitics : bool
        Split clitics in Arabic tokens.

    Returns
    -------
    List[str]
        Interleaved list of Arabic and English tokens in document order.

    Example
    -------
    >>> tokenize_mixed("هذا phone جديد")
    ['هذا', 'phone', 'جديد']
    """
    # Split on whitespace to preserve order
    tokens: List[str] = []
    for chunk in text.split():
        arabic_tokens = AR_TOKEN_PATTERN.findall(chunk)
        english_tokens = EN_TOKEN_PATTERN.findall(chunk)
        if arabic_tokens:
            if normalize_arabic_tokens:
                arabic_tokens = [normalize_arabic(t) for t in arabic_tokens]
            if split_clitics:
                arabic_tokens = split_clitic_tokens(arabic_tokens)
            tokens.extend(arabic_tokens)
        elif english_tokens:
            tokens.extend([t.lower() for t in english_tokens])
    return tokens
