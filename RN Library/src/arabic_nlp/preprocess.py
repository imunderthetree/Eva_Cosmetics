"""
preprocess.py — High-level Arabic NLP preprocessing pipeline.

Combines tokenization → stop-word removal → normalization →
stemming / lemmatization into a single configurable callable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set

from .normalization import lemmatize_arabic, normalize_arabic
from .stemming import light_stem
from .stopwords import get_arabic_stopwords, remove_stopwords
from .tokenization import tokenize_arabic


# ─────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class PreprocessConfig:
    """
    Immutable configuration for the Arabic preprocessing pipeline.

    Attributes
    ----------
    normalize_text : bool
        Apply Arabic normalization (diacritics removal, Alef unification, …).
    split_clitics : bool
        Split clitics (prefixes / suffixes) from each token.
    min_clitic_length : int
        Minimum stem length for clitic splitting.
    use_stopwords : bool
        Remove stop words.
    stopwords : Set[str] or None
        Custom stop-word set; built-in Arabic stop words used when ``None``
        and *use_stopwords* is ``True``.
    use_light_stemmer : bool
        Apply the light Arabic stemmer (prefix/suffix stripping).
    use_lemmatizer : bool
        Apply dictionary-based Arabic lemmatization (overrides stemmer when
        both are ``True``).
    stemmer : Callable or None
        Custom stemmer function ``(word) -> stem``.  When provided, takes
        precedence over *use_light_stemmer*.
    """

    normalize_text: bool = True
    split_clitics: bool = False
    min_clitic_length: int = 2
    use_stopwords: bool = True
    stopwords: Optional[Set[str]] = None
    use_light_stemmer: bool = True
    use_lemmatizer: bool = False
    stemmer: Optional[Callable[[str], str]] = None


# ─────────────────────────────────────────────
# Functional API
# ─────────────────────────────────────────────

def preprocess_text(
    text: str,
    stopwords: Optional[Set[str]] = None,
    stemmer: Optional[Callable[[str], str]] = None,
    normalize_text: bool = True,
    split_clitics: bool = False,
    min_clitic_length: int = 2,
    use_stopwords: bool = True,
    use_lemmatizer: bool = False,
    tokenizer: Optional[Callable[[str], Sequence[str]]] = None,
) -> List[str]:
    """
    Full preprocessing pipeline for a single Arabic text string.

    Steps (in order):

    1. **Tokenize** – split text into Arabic word tokens.
    2. **Stop-word removal** – filter common function words.
    3. **Normalize** – remove diacritics, unify character variants.
    4. **Lemmatize or stem** – reduce words to base forms.

    Parameters
    ----------
    text : str
        Raw Arabic text.
    stopwords : Set[str] or None
        Custom stop-word set (built-in Arabic set used when ``None``).
    stemmer : Callable or None
        Custom stemmer; defaults to :func:`~arabic_nlp.stemming.light_stem`.
    normalize_text : bool
        Normalize tokens after tokenization.
    split_clitics : bool
        Split clitics before stop-word removal.
    min_clitic_length : int
        Minimum stem length for clitic splitting.
    use_stopwords : bool
        Whether to apply stop-word removal (default ``True``).
    use_lemmatizer : bool
        Use dictionary lemmatization instead of (or in addition to) stemming.
    tokenizer : Callable or None
        Custom tokenizer; defaults to :func:`~arabic_nlp.tokenization.tokenize_arabic`.

    Returns
    -------
    List[str]
        Preprocessed token list.

    Example
    -------
    >>> preprocess_text("ذهب الولد إلى المدرسة في الصباح")
    ['ذهب', 'ولد', 'مدرس', 'صباح']
    """
    # 1. Tokenize
    if tokenizer is not None:
        tokens = list(tokenizer(text))
    else:
        tokens = tokenize_arabic(
            text,
            normalize_text=normalize_text,
            split_clitics=split_clitics,
            min_clitic_length=min_clitic_length,
        )

    # 2. Stop-word removal
    if use_stopwords:
        sw = stopwords if stopwords is not None else get_arabic_stopwords()
        tokens = remove_stopwords(tokens, stopwords=sw)

    # 3. Lemmatize or stem
    if use_lemmatizer:
        tokens = [lemmatize_arabic(t) for t in tokens]
    elif stemmer is not None:
        tokens = [stemmer(t) for t in tokens]
    else:
        tokens = [light_stem(t) for t in tokens]

    return tokens


# ─────────────────────────────────────────────
# Class-based pipeline
# ─────────────────────────────────────────────

class ArabicPreprocessor:
    """
    Reusable Arabic text preprocessing pipeline.

    Instantiate once, then call it on any string.

    Parameters
    ----------
    stopwords : Set[str] or None
        Custom stop-word set.
    stemmer : Callable or None
        Custom stemmer function.
    normalize_text : bool
        Normalize tokens (default ``True``).
    use_light_stemmer : bool
        Use light Arabic stemmer when no custom stemmer is given.
    use_lemmatizer : bool
        Use dictionary-based lemmatizer.
    use_stopwords : bool
        Remove stop words.
    split_clitics : bool
        Split clitics from tokens.
    min_clitic_length : int
        Minimum stem length for clitic splitting.
    tokenizer : Callable or None
        Custom tokenizer.

    Example
    -------
    >>> preprocessor = ArabicPreprocessor(use_lemmatizer=True)
    >>> preprocessor("الأولاد يذهبون إلى المدارس")
    ['ولد', 'ذهب', 'مدرسة']
    """

    def __init__(
        self,
        stopwords: Optional[Set[str]] = None,
        stemmer: Optional[Callable[[str], str]] = None,
        normalize_text: bool = True,
        use_light_stemmer: bool = True,
        use_lemmatizer: bool = False,
        use_stopwords: bool = True,
        split_clitics: bool = False,
        min_clitic_length: int = 2,
        tokenizer: Optional[Callable[[str], Sequence[str]]] = None,
    ) -> None:
        self.stopwords = stopwords
        if stemmer is None and use_light_stemmer and not use_lemmatizer:
            stemmer = light_stem
        self.stemmer = stemmer
        self.normalize_text = normalize_text
        self.use_lemmatizer = use_lemmatizer
        self.use_stopwords = use_stopwords
        self.split_clitics = split_clitics
        self.min_clitic_length = min_clitic_length
        self.tokenizer = tokenizer

    def __call__(self, text: str) -> List[str]:
        return preprocess_text(
            text,
            stopwords=self.stopwords,
            stemmer=self.stemmer,
            normalize_text=self.normalize_text,
            split_clitics=self.split_clitics,
            min_clitic_length=self.min_clitic_length,
            use_stopwords=self.use_stopwords,
            use_lemmatizer=self.use_lemmatizer,
            tokenizer=self.tokenizer,
        )

    @classmethod
    def from_config(cls, config: PreprocessConfig) -> "ArabicPreprocessor":
        """Create a preprocessor from a :class:`PreprocessConfig`."""
        return cls(
            stopwords=config.stopwords,
            stemmer=config.stemmer,
            normalize_text=config.normalize_text,
            use_light_stemmer=config.use_light_stemmer,
            use_lemmatizer=config.use_lemmatizer,
            use_stopwords=config.use_stopwords,
            split_clitics=config.split_clitics,
            min_clitic_length=config.min_clitic_length,
        )
