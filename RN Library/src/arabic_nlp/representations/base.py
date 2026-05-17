from __future__ import annotations

from typing import Iterable, List, Sequence


class BaseEncoder:
    def encode(self, text: str | None = None, tokens: Sequence[str] | None = None, **kwargs):
        raise NotImplementedError

    def encode_batch(
        self,
        texts: Iterable[str] | None = None,
        tokens_batch: Iterable[Sequence[str]] | None = None,
        **kwargs,
    ):
        if texts is None and tokens_batch is None:
            raise ValueError("Either texts or tokens_batch must be provided")

        if texts is not None:
            return [self.encode(text=text, **kwargs) for text in texts]
        return [self.encode(tokens=tokens, **kwargs) for tokens in tokens_batch or []]

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError


class TrainableEncoder(BaseEncoder):
    def fit(self, corpus_tokens: Iterable[Sequence[str]]):
        raise NotImplementedError


def mean_pool(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    length = len(vectors[0])
    summed = [0.0] * length
    for vector in vectors:
        for idx, value in enumerate(vector):
            summed[idx] += value
    return [value / len(vectors) for value in summed]


def sum_pool(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    length = len(vectors[0])
    summed = [0.0] * length
    for vector in vectors:
        for idx, value in enumerate(vector):
            summed[idx] += value
    return summed


def select_pooling(pooling: str):
    if pooling == "mean":
        return mean_pool
    if pooling == "sum":
        return sum_pool
    raise ValueError(f"Unsupported pooling: {pooling}")
