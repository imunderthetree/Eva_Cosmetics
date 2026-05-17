from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from .base import BaseEncoder


def _require_allennlp():
    try:
        import torch
        from allennlp.modules.elmo import Elmo, batch_to_ids
    except ImportError as exc:
        raise RuntimeError(
            "allennlp is required for ELMo. Install: pip install RNArabic[elmo]"
        ) from exc
    return torch, Elmo, batch_to_ids


def _default_tokenizer():
    from ..tokenization import tokenize_arabic

    return tokenize_arabic


class ElmoEncoder(BaseEncoder):
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        pooling: str = "mean",
        representation_index: int = 0,
        device: str | None = None,
        tokenizer=None,
    ) -> None:
        torch, Elmo, batch_to_ids = _require_allennlp()
        self.torch = torch
        self.batch_to_ids = batch_to_ids
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0.0)
        if device:
            self.elmo.to(device)
        self.elmo.eval()
        self.options_file = options_file
        self.weight_file = weight_file
        self.pooling = pooling
        self.representation_index = representation_index
        self.device = device
        self.tokenizer = tokenizer or _default_tokenizer()

    def _pool(self, vectors):
        if self.pooling == "mean":
            return vectors.mean(dim=0)
        if self.pooling == "max":
            return vectors.max(dim=0).values
        if self.pooling == "first":
            return vectors[0]
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def encode(self, text: str | None = None, tokens: Sequence[str] | None = None):
        if tokens is None:
            if text is None:
                raise ValueError("Either text or tokens must be provided")
            tokens = self.tokenizer(text)
        tokens_list = list(tokens)

        char_ids = self.batch_to_ids([tokens_list])
        if self.device:
            char_ids = char_ids.to(self.device)

        with self.torch.no_grad():
            outputs = self.elmo(char_ids)
        representations = outputs["elmo_representations"][self.representation_index]
        token_vectors = representations[0, : len(tokens_list), :]
        pooled = self._pool(token_vectors)
        return pooled.tolist()

    def encode_batch(
        self,
        texts: Iterable[str] | None = None,
        tokens_batch: Iterable[Sequence[str]] | None = None,
    ):
        if texts is None and tokens_batch is None:
            raise ValueError("Either texts or tokens_batch must be provided")
        if tokens_batch is None:
            tokens_batch = [self.tokenizer(text) for text in texts or []]
        tokens_list = [list(tokens) for tokens in tokens_batch]

        char_ids = self.batch_to_ids(tokens_list)
        if self.device:
            char_ids = char_ids.to(self.device)

        with self.torch.no_grad():
            outputs = self.elmo(char_ids)
        representations = outputs["elmo_representations"][self.representation_index]

        pooled = []
        for row, tokens in zip(representations, tokens_list):
            token_vectors = row[: len(tokens), :]
            pooled.append(self._pool(token_vectors).tolist())
        return pooled

    def save(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        metadata = {
            "options_file": self.options_file,
            "weight_file": self.weight_file,
            "pooling": self.pooling,
            "representation_index": self.representation_index,
            "device": self.device,
        }
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "ElmoEncoder":
        source = Path(path)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        return cls(
            options_file=metadata["options_file"],
            weight_file=metadata["weight_file"],
            pooling=metadata.get("pooling", "mean"),
            representation_index=metadata.get("representation_index", 0),
            device=metadata.get("device"),
        )
