"""
Reranker interface with optional cross-encoder backend.

Falls back to a no-op reranker (preserves dense retrieval order) when
sentence-transformers is not installed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Reranker(ABC):
    """Interface for reranking query-document pairs."""

    @abstractmethod
    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score each (query, document) pair.

        Returns:
            1-D float array of scores (higher = more relevant).
        """
        ...


class CrossEncoderReranker(Reranker):
    """Precision reranking via a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder  # noqa: F811
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking: "
                "pip install memx[local]"
            ) from None

        self._model = CrossEncoder(model_name)

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        scores = self._model.predict(pairs)
        return np.asarray(scores, dtype="float32")


class NoReranker(Reranker):
    """Pass-through reranker that preserves the input ordering."""

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.arange(len(pairs), 0, -1, dtype="float32")


def auto_reranker(model: str | Reranker | None = None) -> Reranker:
    """Create a reranker, auto-detecting the best available backend.

    Args:
        model: One of:
            - A ``Reranker`` instance (returned as-is).
            - A model name string → ``CrossEncoderReranker``.
            - ``None`` → try CrossEncoder, fall back to ``NoReranker``.
    """
    if isinstance(model, Reranker):
        return model

    if model is not None:
        return CrossEncoderReranker(model_name=model)

    try:
        return CrossEncoderReranker()
    except ImportError:
        return NoReranker()
