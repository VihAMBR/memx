"""
Memory store: embed every message and index with FAISS.
No gating. No filtering. Store everything.

Backed by SQLite (via db.py) — every add() persists immediately.
FAISS index is rebuilt lazily from whatever is in memory.
On init, loads all existing memories from the database.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError as e:
    raise ImportError("faiss-cpu is required: pip install faiss-cpu") from e

from .db import MemoryDB


class MemoryStore:
    """
    Flat store of conversation turns with lazy FAISS indexing and SQLite persistence.

    Design choice: store *every* message without filtering.
    gated-mem experiments showed that surprise-based gating filtered out
    precisely the temporal and knowledge-update facts the benchmark queries.
    """

    def __init__(self, db: MemoryDB, model_name: str | SentenceTransformer = "all-MiniLM-L6-v2"):
        self.db = db
        self.encoder = (
            model_name if isinstance(model_name, SentenceTransformer)
            else SentenceTransformer(model_name)
        )
        self.memories: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.index = None  # faiss.IndexFlatIP
        self._dirty = False

        # Load existing memories from SQLite
        self._load_from_db()

    def _load_from_db(self) -> None:
        """Hydrate in-memory list from SQLite."""
        self.memories = self.db.get_all_memories()
        if self.memories:
            self._dirty = True

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, text: str, timestamp: str, role: str, session_id: int) -> int:
        """
        Store a single message. Writes to SQLite immediately.
        Returns the database row ID.
        """
        row_id = self.db.insert_memory(text, timestamp, role, session_id)
        self.memories.append(
            {
                "id": row_id,
                "text": text,
                "timestamp": timestamp,
                "role": role,
                "session_id": session_id,
            }
        )
        self._dirty = True
        return row_id

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
        Return the top-k memories by cosine similarity (normalized dot product).
        Rebuilds the index automatically if new messages were added since last build.
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
            if 0 <= i < len(self.memories):
                results.append((self.memories[i], float(scores[0][j])))
        return results
