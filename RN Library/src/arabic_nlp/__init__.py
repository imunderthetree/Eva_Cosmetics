"""
arabic_nlp — Egyptian Arabic NLP toolkit.

Quick-start
-----------
>>> from arabic_nlp import ArabicPreprocessor
>>> preprocessor = ArabicPreprocessor()
>>> preprocessor("ذهب الولد إلى المدرسة")
['ذهب', 'ولد', 'مدرس']

Modules
-------
normalization
    normalize_arabic, case_fold, soundex_arabic, soundex_english,
    thesaurus_lookup, lemmatize_arabic, lemmatize_english

tokenization
    tokenize_arabic, tokenize_sentences, tokenize_mixed, split_clitic_tokens

stemming
    light_stem, get_isri_stemmer, get_porter_stemmer, get_snowball_stemmer,
    stem_tokens

stopwords
    get_arabic_stopwords, get_english_stopwords, load_stopwords,
    remove_stopwords

preprocess
    ArabicPreprocessor, PreprocessConfig, preprocess_text

ir
    CountVectorizer, TfidfVectorizer, BM25Vectorizer, HashingVectorizer
"""

from .normalization import (
    case_fold,
    lemmatize_arabic,
    lemmatize_english,
    normalize_arabic,
    soundex_arabic,
    soundex_english,
    thesaurus_lookup,
)
from .preprocess import ArabicPreprocessor, PreprocessConfig, preprocess_text
from .stemming import (
    get_isri_stemmer,
    get_porter_stemmer,
    get_snowball_stemmer,
    light_stem,
    stem_tokens,
)
from .stopwords import (
    get_arabic_stopwords,
    get_english_stopwords,
    load_stopwords,
    remove_stopwords,
)
from .tokenization import (
    split_clitic_tokens,
    tokenize_arabic,
    tokenize_mixed,
    tokenize_sentences,
)
from .ir import BM25Vectorizer, CountVectorizer, HashingVectorizer, TfidfVectorizer

__all__ = [
    # normalization
    "normalize_arabic",
    "case_fold",
    "soundex_arabic",
    "soundex_english",
    "thesaurus_lookup",
    "lemmatize_arabic",
    "lemmatize_english",
    # tokenization
    "tokenize_arabic",
    "tokenize_sentences",
    "tokenize_mixed",
    "split_clitic_tokens",
    # stemming
    "light_stem",
    "get_isri_stemmer",
    "get_porter_stemmer",
    "get_snowball_stemmer",
    "stem_tokens",
    # stopwords
    "get_arabic_stopwords",
    "get_english_stopwords",
    "load_stopwords",
    "remove_stopwords",
    # pipeline
    "ArabicPreprocessor",
    "PreprocessConfig",
    "preprocess_text",
    # ir
    "CountVectorizer",
    "TfidfVectorizer",
    "BM25Vectorizer",
    "HashingVectorizer",
]

__version__ = "1.0.0"
