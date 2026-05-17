from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .base import TrainableEncoder, select_pooling


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required for CBOW/Skip-gram. Install: pip install RNArabic[ml]"
        ) from exc
    return np


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


@dataclass
class Word2VecConfig:
    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    negative: int = 5
    epochs: int = 5
    learning_rate: float = 0.025
    seed: int = 0


class _Word2VecBase(TrainableEncoder):
    def __init__(self, config: Optional[Word2VecConfig] = None) -> None:
        self.config = config or Word2VecConfig()
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        self.counts_: Dict[str, int] = {}
        self.in_vectors_ = None
        self.out_vectors_ = None
        self.neg_sampling_probs_ = None

    def _build_vocab(self, corpus_tokens: Iterable[Sequence[str]]) -> List[List[str]]:
        counts: Dict[str, int] = {}
        corpus_list: List[List[str]] = []
        for tokens in corpus_tokens:
            tokens_list = list(tokens)
            corpus_list.append(tokens_list)
            for token in tokens_list:
                counts[token] = counts.get(token, 0) + 1

        items = [token for token, count in counts.items() if count >= self.config.min_count]
        items.sort()
        self.id_to_token = items
        self.token_to_id = {token: idx for idx, token in enumerate(items)}
        self.counts_ = counts
        return corpus_list

    def _init_weights(self):
        np = _require_numpy()
        vocab_size = len(self.id_to_token)
        limit = 0.5 / max(1, self.config.vector_size)
        rng = np.random.default_rng(self.config.seed)
        self.in_vectors_ = rng.uniform(-limit, limit, size=(vocab_size, self.config.vector_size))
        self.out_vectors_ = rng.uniform(-limit, limit, size=(vocab_size, self.config.vector_size))

        freqs = np.array(
            [max(1, self.counts_.get(token, 1)) for token in self.id_to_token],
            dtype=np.float64,
        )
        freqs = freqs ** 0.75
        self.neg_sampling_probs_ = freqs / freqs.sum()

    def _negative_samples(self, rng, positive_idx: int) -> List[int]:
        np = _require_numpy()
        if self.config.negative <= 0:
            return []
        negatives = rng.choice(
            len(self.id_to_token),
            size=self.config.negative,
            replace=True,
            p=self.neg_sampling_probs_,
        )
        return [int(idx) for idx in negatives if int(idx) != positive_idx]

    def _tokens_to_indices(self, tokens: Sequence[str]) -> List[int]:
        return [self.token_to_id[token] for token in tokens if token in self.token_to_id]

    def encode(
        self,
        text: str | None = None,
        tokens: Sequence[str] | None = None,
        pooling: str = "mean",
    ) -> List[float] | List[List[float]]:
        if tokens is None:
            if text is None:
                raise ValueError("Either text or tokens must be provided")
            tokens = text.split()

        if self.in_vectors_ is None:
            raise RuntimeError("Model is not trained")

        vectors: List[List[float]] = []
        for token in tokens:
            idx = self.token_to_id.get(token)
            if idx is None:
                continue
            vectors.append(self.in_vectors_[idx].tolist())

        if pooling == "none":
            return vectors
        pooler = select_pooling(pooling)
        return pooler(vectors)

    def encode_batch(
        self,
        texts: Iterable[str] | None = None,
        tokens_batch: Iterable[Sequence[str]] | None = None,
        pooling: str = "mean",
    ):
        if texts is not None:
            return [self.encode(text=text, pooling=pooling) for text in texts]
        return [self.encode(tokens=tokens, pooling=pooling) for tokens in tokens_batch or []]

    def save(self, path: str) -> None:
        if self.in_vectors_ is None or self.out_vectors_ is None:
            raise RuntimeError("Model is not trained")
        np = _require_numpy()
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)

        np.savez(
            target / "vectors.npz",
            in_vectors=self.in_vectors_,
            out_vectors=self.out_vectors_,
        )
        metadata = {
            "model_type": self.model_type,
            "config": {
                "vector_size": self.config.vector_size,
                "window": self.config.window,
                "min_count": self.config.min_count,
                "negative": self.config.negative,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "seed": self.config.seed,
            },
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
        }
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def _load_common(cls, path: str, expected_type: str):
        np = _require_numpy()
        source = Path(path)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("model_type") != expected_type:
            raise RuntimeError("Model type does not match")

        config = metadata.get("config", {})
        model = cls(config=Word2VecConfig(**config))
        model.token_to_id = {k: int(v) for k, v in metadata.get("token_to_id", {}).items()}
        model.id_to_token = list(metadata.get("id_to_token", []))
        vectors = np.load(source / "vectors.npz")
        model.in_vectors_ = vectors["in_vectors"]
        model.out_vectors_ = vectors["out_vectors"]
        return model


class CBOW(_Word2VecBase):
    model_type = "cbow"

    def fit(self, corpus_tokens: Iterable[Sequence[str]]):
        np = _require_numpy()
        corpus_list = self._build_vocab(corpus_tokens)
        self._init_weights()
        rng = np.random.default_rng(self.config.seed)

        for _ in range(self.config.epochs):
            for tokens in corpus_list:
                indices = self._tokens_to_indices(tokens)
                for idx, target_idx in enumerate(indices):
                    start = max(0, idx - self.config.window)
                    end = min(len(indices), idx + self.config.window + 1)
                    context = [indices[i] for i in range(start, end) if i != idx]
                    if not context:
                        continue

                    context_vectors = self.in_vectors_[context]
                    context_vector = context_vectors.mean(axis=0)

                    output_vector = self.out_vectors_[target_idx]
                    score = _sigmoid(float(context_vector.dot(output_vector)))
                    grad = self.config.learning_rate * (1.0 - score)
                    self.out_vectors_[target_idx] += grad * context_vector
                    self.in_vectors_[context] += (grad * output_vector) / len(context)

                    for neg_idx in self._negative_samples(rng, target_idx):
                        neg_vector = self.out_vectors_[neg_idx]
                        score = _sigmoid(float(context_vector.dot(neg_vector)))
                        grad = self.config.learning_rate * (0.0 - score)
                        self.out_vectors_[neg_idx] += grad * context_vector
                        self.in_vectors_[context] += (grad * neg_vector) / len(context)

        return self

    @classmethod
    def load(cls, path: str) -> "CBOW":
        return cls._load_common(path, expected_type="cbow")


class SkipGram(_Word2VecBase):
    model_type = "skipgram"

    def fit(self, corpus_tokens: Iterable[Sequence[str]]):
        np = _require_numpy()
        corpus_list = self._build_vocab(corpus_tokens)
        self._init_weights()
        rng = np.random.default_rng(self.config.seed)

        for _ in range(self.config.epochs):
            for tokens in corpus_list:
                indices = self._tokens_to_indices(tokens)
                for idx, target_idx in enumerate(indices):
                    start = max(0, idx - self.config.window)
                    end = min(len(indices), idx + self.config.window + 1)
                    for ctx_pos in range(start, end):
                        if ctx_pos == idx:
                            continue
                        context_idx = indices[ctx_pos]
                        target_vector = self.in_vectors_[target_idx]
                        context_vector = self.out_vectors_[context_idx]
                        score = _sigmoid(float(target_vector.dot(context_vector)))
                        grad = self.config.learning_rate * (1.0 - score)
                        self.out_vectors_[context_idx] += grad * target_vector
                        self.in_vectors_[target_idx] += grad * context_vector

                        for neg_idx in self._negative_samples(rng, context_idx):
                            neg_vector = self.out_vectors_[neg_idx]
                            score = _sigmoid(float(target_vector.dot(neg_vector)))
                            grad = self.config.learning_rate * (0.0 - score)
                            self.out_vectors_[neg_idx] += grad * target_vector
                            self.in_vectors_[target_idx] += grad * neg_vector

        return self

    @classmethod
    def load(cls, path: str) -> "SkipGram":
        return cls._load_common(path, expected_type="skipgram")
