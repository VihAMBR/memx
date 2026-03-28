"""
Embedding interface with pluggable backends.

Auto-detection priority (when no backend is specified):
  1. OpenAI API        — pip install memx[openai]   + OPENAI_API_KEY set
  2. Google Vertex AI  — pip install memx[vertex]    + default credentials configured
  3. sentence-transformers — pip install memx[local]  (fully offline)
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Interface for text embedding backends."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into normalized L2 embeddings.

        Returns:
            (N, dim) float32 ndarray with unit-norm rows.
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...


# ------------------------------------------------------------------
# Concrete backends
# ------------------------------------------------------------------


class OpenAIEmbedder(Embedder):
    """Embeddings via the OpenAI API (text-embedding-3-small by default)."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        try:
            import openai  # noqa: F811
        except ImportError:
            raise ImportError("openai is required: pip install memx[openai]") from None

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._dim: int | None = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype="float32")

        resp = self._client.embeddings.create(input=texts, model=self._model)
        vecs = np.array([e.embedding for e in resp.data], dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs /= norms

        if self._dim is None:
            self._dim = vecs.shape[1]
        return vecs

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self.encode(["dim probe"]).shape[1]
        return self._dim


class VertexEmbedder(Embedder):
    """Embeddings via Google Vertex AI (text-embedding-005 by default)."""

    def __init__(
        self,
        model: str = "text-embedding-005",
        project: str | None = None,
        location: str = "us-central1",
    ):
        try:
            from vertexai.language_models import TextEmbeddingModel  # noqa: F811
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform is required: pip install memx[vertex]"
            ) from None

        if project:
            import vertexai as _vtx

            _vtx.init(project=project, location=location)

        self._model = TextEmbeddingModel.from_pretrained(model)
        self._dim: int | None = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype="float32")

        embeddings = self._model.get_embeddings(texts)
        vecs = np.array([e.values for e in embeddings], dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs /= norms

        if self._dim is None:
            self._dim = vecs.shape[1]
        return vecs

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self.encode(["dim probe"]).shape[1]
        return self._dim


class SentenceTransformerEmbedder(Embedder):
    """Embeddings via sentence-transformers (fully local, no API key needed)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F811
        except ImportError:
            raise ImportError(
                "sentence-transformers is required: pip install memx[local]"
            ) from None

        self._model = SentenceTransformer(model_name)
        self._dim_val: int = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

    @property
    def dim(self) -> int:
        return self._dim_val


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def auto_embedder(model: str | Embedder | None = None) -> Embedder:
    """Create an embedder, auto-detecting the best available backend.

    Args:
        model: One of:
            - An ``Embedder`` instance (returned as-is).
            - A model name string — routed by prefix:
              ``"text-embedding-*"`` → OpenAI, otherwise → sentence-transformers.
            - ``None`` → auto-detect in priority order.

    Priority when *model* is None:
        1. OpenAI  (if ``openai`` installed **and** ``OPENAI_API_KEY`` set)
        2. Vertex AI  (if ``google-cloud-aiplatform`` installed)
        3. sentence-transformers  (if installed)
    """
    if isinstance(model, Embedder):
        return model

    if model is not None:
        if model.startswith("text-embedding-"):
            return OpenAIEmbedder(model=model)
        return SentenceTransformerEmbedder(model_name=model)

    # Auto-detect
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIEmbedder()
        except ImportError:
            pass

    try:
        return VertexEmbedder()
    except (ImportError, Exception):
        pass

    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        pass

    raise ImportError(
        "No embedding backend available. Install one of:\n"
        "  pip install memx[openai]   # OpenAI API (recommended)\n"
        "  pip install memx[vertex]   # Google Vertex AI\n"
        "  pip install memx[local]    # sentence-transformers (local)\n"
    )
