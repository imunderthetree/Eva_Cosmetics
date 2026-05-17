from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from .base import BaseEncoder


def _require_transformers():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers and torch are required. Install: pip install RNArabic[transformers]"
        ) from exc
    return torch, AutoModel, AutoTokenizer


def _mean_pool(hidden_states, attention_mask, torch):
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def _max_pool(hidden_states, attention_mask, torch):
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    masked = hidden_states.masked_fill(mask == 0, float("-inf"))
    return masked.max(dim=1).values


class BertEncoder(BaseEncoder):
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        pooling: str = "cls",
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        torch, AutoModel, AutoTokenizer = _require_transformers()
        self.torch = torch
        self.model_name = model_name
        self.pooling = pooling
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if device:
            self.model.to(device)

    def _pool(self, hidden_states, attention_mask):
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        if self.pooling == "mean":
            return _mean_pool(hidden_states, attention_mask, self.torch)
        if self.pooling == "max":
            return _max_pool(hidden_states, attention_mask, self.torch)
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def encode(self, text: str | None = None, tokens: Sequence[str] | None = None):
        if tokens is not None:
            text = " ".join(tokens)
        if text is None:
            raise ValueError("Either text or tokens must be provided")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        if self.device:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)
        pooled = self._pool(outputs.last_hidden_state, inputs["attention_mask"])
        return pooled.squeeze(0).tolist()

    def encode_batch(
        self,
        texts: Iterable[str] | None = None,
        tokens_batch: Iterable[Sequence[str]] | None = None,
    ):
        if texts is None and tokens_batch is None:
            raise ValueError("Either texts or tokens_batch must be provided")
        if tokens_batch is not None:
            texts = [" ".join(tokens) for tokens in tokens_batch]

        inputs = self.tokenizer(
            list(texts or []),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        if self.device:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)
        pooled = self._pool(outputs.last_hidden_state, inputs["attention_mask"])
        return pooled.tolist()

    def save(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_name": self.model_name,
            "pooling": self.pooling,
            "device": self.device,
            "max_length": self.max_length,
        }
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BertEncoder":
        source = Path(path)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        return cls(
            model_name=metadata.get("model_name", "bert-base-multilingual-cased"),
            pooling=metadata.get("pooling", "cls"),
            device=metadata.get("device"),
            max_length=metadata.get("max_length", 512),
        )
