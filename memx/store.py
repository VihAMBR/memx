"""
Memory store: embed every message and index with FAISS.
No gating. No filtering. Store everything.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError as e:
    raise ImportError("faiss-cpu is required: pip install faiss-cpu") from e


class MemoryStore:
    """
    Flat store of conversation turns with lazy FAISS indexing.

    Design choice: store *every* message without filtering.
    gated-mem experiments showed that surprise-based gating filtered out
    precisely the temporal and knowledge-update facts the benchmark queries.
    Storing everything costs ~$0 (no LLM calls) and avoids that failure mode.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.memories: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.index = None  # faiss.IndexFlatIP
        self._dirty = False

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, text: str, timestamp: str, role: str, session_id: int) -> None:
        """Append a single message. Index is rebuilt lazily on next search."""
        self.memories.append(
            {
                "text": text,
                "timestamp": timestamp,
                "role": role,
                "session_id": session_id,
            }
        )
        self._dirty = True

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """Embed all stored messages and build a FAISS inner-product index."""
        if not self.memories:
            return
        texts = [m["text"] for m in self.memories]
        self.embeddings = self.encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        self._dirty = False

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 50) -> list[tuple[dict, float]]:
        """
        Return the top-k memories by cosine similarity (normalized → inner product).
        Rebuilds the index automatically if new messages were added since the last build.
        """
        if not self.memories:
            return []
        if self._dirty or self.index is None:
            self.build_index()

        q_emb = self.encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

        actual_k = min(top_k, len(self.memories))
        scores, indices = self.index.search(q_emb, actual_k)

        results = []
        for j, i in enumerate(indices[0]):
            if i < len(self.memories):
                results.append((self.memories[i], float(scores[0][j])))
        return results

    # ------------------------------------------------------------------
    # Persistence (optional)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist memories (and embeddings) to disk."""
        import pickle, pathlib

        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "memories.pkl", "wb") as f:
            pickle.dump(self.memories, f)
        if self.embeddings is not None:
            np.save(str(p / "embeddings.npy"), self.embeddings)

    def load(self, path: str) -> None:
        """Restore memories and rebuild the FAISS index."""
        import pickle, pathlib

        p = pathlib.Path(path)
        with open(p / "memories.pkl", "rb") as f:
            self.memories = pickle.load(f)
        emb_path = p / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(str(emb_path))
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
            self._dirty = False
        else:
            self._dirty = True
