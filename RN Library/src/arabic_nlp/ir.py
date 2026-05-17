from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def term_frequencies(tokens: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def document_frequencies(corpus_tokens: Iterable[Sequence[str]]) -> Dict[str, int]:
    doc_freqs: Dict[str, int] = {}
    for tokens in corpus_tokens:
        for token in set(tokens):
            doc_freqs[token] = doc_freqs.get(token, 0) + 1
    return doc_freqs


def inverse_document_frequencies(
    doc_freqs: Dict[str, int],
    doc_count: int,
    smooth: bool = True,
) -> Dict[str, float]:
    idf: Dict[str, float] = {}
    for token, df in doc_freqs.items():
        if smooth:
            idf[token] = math.log((1.0 + doc_count) / (1.0 + df)) + 1.0
        else:
            idf[token] = math.log(doc_count / df)
    return idf


def l2_normalize(values: Dict[int, float]) -> Dict[int, float]:
    norm = math.sqrt(sum(score * score for score in values.values()))
    if norm == 0:
        return values
    return {idx: score / norm for idx, score in values.items()}


def word_ngrams(
    tokens: Sequence[str],
    ngram_range: Tuple[int, int] = (1, 1),
    joiner: str = "_",
) -> List[str]:
    min_n, max_n = ngram_range
    if min_n <= 0 or max_n < min_n:
        raise ValueError("Invalid ngram_range")

    ngrams: List[str] = []
    for n in range(min_n, max_n + 1):
        if n == 1:
            ngrams.extend(tokens)
            continue
        for i in range(len(tokens) - n + 1):
            ngrams.append(joiner.join(tokens[i : i + n]))
    return ngrams


def char_ngrams(
    text: str,
    ngram_range: Tuple[int, int] = (3, 5),
    keep_whitespace: bool = False,
) -> List[str]:
    if not keep_whitespace:
        text = " ".join(text.split())

    min_n, max_n = ngram_range
    if min_n <= 0 or max_n < min_n:
        raise ValueError("Invalid ngram_range")

    ngrams: List[str] = []
    for n in range(min_n, max_n + 1):
        for i in range(len(text) - n + 1):
            ngrams.append(text[i : i + n])
    return ngrams


def document_stats(tokens: Sequence[str]) -> Dict[str, int]:
    return {
        "length": len(tokens),
        "unique_terms": len(set(tokens)),
    }


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]
    id_to_token: List[str]

    @classmethod
    def from_counts(cls, counts: Dict[str, int], min_count: int = 1) -> "Vocabulary":
        items = [token for token, count in counts.items() if count >= min_count]
        items.sort()
        token_to_id = {token: idx for idx, token in enumerate(items)}
        return cls(token_to_id=token_to_id, id_to_token=items)


class CountVectorizer:
    def __init__(
        self,
        min_count: int = 1,
        ngram_range: Tuple[int, int] = (1, 1),
        token_joiner: str = "_",
        lowercase: bool = False,
    ) -> None:
        self.min_count = min_count
        self.ngram_range = ngram_range
        self.token_joiner = token_joiner
        self.lowercase = lowercase
        self.vocab_: Optional[Vocabulary] = None

    def _prepare_tokens(self, tokens: Sequence[str]) -> List[str]:
        if self.lowercase:
            return [token.lower() for token in tokens]
        return list(tokens)

    def _ngrams(self, tokens: Sequence[str]) -> List[str]:
        return word_ngrams(tokens, ngram_range=self.ngram_range, joiner=self.token_joiner)

    def fit(self, corpus_tokens: Iterable[Sequence[str]]) -> "CountVectorizer":
        counts: Dict[str, int] = {}
        for tokens in corpus_tokens:
            tokens = self._prepare_tokens(tokens)
            for ngram in self._ngrams(tokens):
                counts[ngram] = counts.get(ngram, 0) + 1
        self.vocab_ = Vocabulary.from_counts(counts, min_count=self.min_count)
        return self

    def transform(self, tokens: Sequence[str], return_sparse: bool = True):
        if self.vocab_ is None:
            raise RuntimeError("Vectorizer has not been fitted")
        tokens = self._prepare_tokens(tokens)
        counts: Dict[int, float] = {}
        for ngram in self._ngrams(tokens):
            idx = self.vocab_.token_to_id.get(ngram)
            if idx is None:
                continue
            counts[idx] = counts.get(idx, 0.0) + 1.0

        if return_sparse:
            return counts

        dense = [0.0] * len(self.vocab_.id_to_token)
        for idx, value in counts.items():
            dense[idx] = value
        return dense

    def fit_transform(self, corpus_tokens: Iterable[Sequence[str]], return_sparse: bool = True):
        self.fit(corpus_tokens)
        return [self.transform(tokens, return_sparse=return_sparse) for tokens in corpus_tokens]


class TfidfVectorizer(CountVectorizer):
    def __init__(
        self,
        min_count: int = 1,
        ngram_range: Tuple[int, int] = (1, 1),
        token_joiner: str = "_",
        lowercase: bool = False,
        smooth_idf: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__(
            min_count=min_count,
            ngram_range=ngram_range,
            token_joiner=token_joiner,
            lowercase=lowercase,
        )
        self.smooth_idf = smooth_idf
        self.normalize = normalize
        self.idf_: Optional[Dict[int, float]] = None

    def fit(self, corpus_tokens: Iterable[Sequence[str]]) -> "TfidfVectorizer":
        corpus_list = [self._prepare_tokens(tokens) for tokens in corpus_tokens]
        counts: Dict[str, int] = {}
        for tokens in corpus_list:
            for ngram in self._ngrams(tokens):
                counts[ngram] = counts.get(ngram, 0) + 1
        self.vocab_ = Vocabulary.from_counts(counts, min_count=self.min_count)

        doc_freqs: Dict[str, int] = {}
        for tokens in corpus_list:
            seen = set(self._ngrams(tokens))
            for ngram in seen:
                doc_freqs[ngram] = doc_freqs.get(ngram, 0) + 1

        doc_count = len(corpus_list)
        idf_map = inverse_document_frequencies(doc_freqs, doc_count, smooth=self.smooth_idf)
        self.idf_ = {
            idx: idf_map[token]
            for token, idx in self.vocab_.token_to_id.items()
            if token in idf_map
        }
        return self

    def transform(self, tokens: Sequence[str], return_sparse: bool = True):
        if self.vocab_ is None or self.idf_ is None:
            raise RuntimeError("Vectorizer has not been fitted")
        tokens = self._prepare_tokens(tokens)
        counts: Dict[int, float] = {}
        for ngram in self._ngrams(tokens):
            idx = self.vocab_.token_to_id.get(ngram)
            if idx is None:
                continue
            counts[idx] = counts.get(idx, 0.0) + 1.0

        tfidf: Dict[int, float] = {}
        for idx, tf in counts.items():
            tfidf[idx] = tf * self.idf_.get(idx, 0.0)

        if self.normalize:
            tfidf = l2_normalize(tfidf)

        if return_sparse:
            return tfidf

        dense = [0.0] * len(self.vocab_.id_to_token)
        for idx, value in tfidf.items():
            dense[idx] = value
        return dense


class BM25Vectorizer(CountVectorizer):
    def __init__(
        self,
        min_count: int = 1,
        ngram_range: Tuple[int, int] = (1, 1),
        token_joiner: str = "_",
        lowercase: bool = False,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        super().__init__(
            min_count=min_count,
            ngram_range=ngram_range,
            token_joiner=token_joiner,
            lowercase=lowercase,
        )
        self.k1 = k1
        self.b = b
        self.avgdl_: Optional[float] = None
        self.idf_: Optional[Dict[int, float]] = None

    def fit(self, corpus_tokens: Iterable[Sequence[str]]) -> "BM25Vectorizer":
        corpus_list = [self._prepare_tokens(tokens) for tokens in corpus_tokens]
        counts: Dict[str, int] = {}
        doc_freqs: Dict[str, int] = {}
        total_len = 0

        for tokens in corpus_list:
            total_len += len(tokens)
            for ngram in self._ngrams(tokens):
                counts[ngram] = counts.get(ngram, 0) + 1
            for ngram in set(self._ngrams(tokens)):
                doc_freqs[ngram] = doc_freqs.get(ngram, 0) + 1

        self.vocab_ = Vocabulary.from_counts(counts, min_count=self.min_count)
        doc_count = len(corpus_list)
        self.avgdl_ = (total_len / doc_count) if doc_count else 0.0

        idf: Dict[int, float] = {}
        for token, idx in self.vocab_.token_to_id.items():
            df = doc_freqs.get(token, 0)
            idf[idx] = math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
        self.idf_ = idf
        return self

    def transform(self, tokens: Sequence[str], return_sparse: bool = True):
        if self.vocab_ is None or self.idf_ is None or self.avgdl_ is None:
            raise RuntimeError("Vectorizer has not been fitted")
        tokens = self._prepare_tokens(tokens)
        doc_len = len(tokens)
        counts: Dict[int, float] = {}
        for ngram in self._ngrams(tokens):
            idx = self.vocab_.token_to_id.get(ngram)
            if idx is None:
                continue
            counts[idx] = counts.get(idx, 0.0) + 1.0

        scores: Dict[int, float] = {}
        for idx, tf in counts.items():
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl_ or 1.0)))
            scores[idx] = self.idf_.get(idx, 0.0) * (numerator / denominator)

        if return_sparse:
            return scores

        dense = [0.0] * len(self.vocab_.id_to_token)
        for idx, value in scores.items():
            dense[idx] = value
        return dense


class HashingVectorizer:
    def __init__(
        self,
        n_features: int = 2**18,
        ngram_range: Tuple[int, int] = (1, 1),
        token_joiner: str = "_",
        lowercase: bool = False,
        use_sign: bool = True,
    ) -> None:
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.token_joiner = token_joiner
        self.lowercase = lowercase
        self.use_sign = use_sign

    def _prepare_tokens(self, tokens: Sequence[str]) -> List[str]:
        if self.lowercase:
            return [token.lower() for token in tokens]
        return list(tokens)

    def _ngrams(self, tokens: Sequence[str]) -> List[str]:
        return word_ngrams(tokens, ngram_range=self.ngram_range, joiner=self.token_joiner)

    def _hash(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16)

    def transform(self, tokens: Sequence[str], return_sparse: bool = True):
        tokens = self._prepare_tokens(tokens)
        counts: Dict[int, float] = {}
        for ngram in self._ngrams(tokens):
            raw = self._hash(ngram)
            idx = raw % self.n_features
            value = 1.0
            if self.use_sign and (raw % 2 == 1):
                value = -1.0
            counts[idx] = counts.get(idx, 0.0) + value

        if return_sparse:
            return counts

        dense = [0.0] * self.n_features
        for idx, value in counts.items():
            dense[idx] = value
        return dense
