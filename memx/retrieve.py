"""
Retrieval: FAISS candidate fetch → cross-encoder rerank → context structuring.

Two-stage pipeline:
  1. FAISS returns top-50 candidates quickly (approximate but fast).
  2. Cross-encoder reranks them for precision (local model, ~10 ms/query).

gated-mem experiments showed this pushed:
  - Temporal accuracy:  67.7% → 76.9%
  - Knowledge-update:   84.6% → 93.3%
"""
from __future__ import annotations

from datetime import datetime as _dt

import numpy as np

from .store import MemoryStore


def _human_date(iso_str: str) -> str:
    try:
        dt = _dt.fromisoformat(iso_str)
    except (ValueError, TypeError):
        return iso_str or "unknown date"
    return f"{dt.strftime('%B')} {dt.day}, {dt.year}"


class Retriever:
    """Two-stage retriever: FAISS candidates + cross-encoder reranking."""

    def __init__(self, reranker: str | Reranker | None = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.store = store
        self.reranker = (
            reranker_model if isinstance(reranker_model, CrossEncoder)
            else CrossEncoder(reranker_model)
        )

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        candidates: int = 50,
    ) -> list[dict]:
        """
        Return the top-k most relevant memories for *query*.

        Stage 1: FAISS cosine search → *candidates* rough results.
        Stage 2: Cross-encoder rerank → keep top-k with highest precision score.
        """
        raw = self.store.search(query, top_k=candidates)
        if not raw:
            return []

        pairs = [(query, mem["text"]) for mem, _ in raw]
        rerank_scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(raw, rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [mem for (mem, _), _ in ranked[:top_k]]

    # ------------------------------------------------------------------
    # Context structuring (Day 4 innovation)
    # ------------------------------------------------------------------

    def structure_context(self, query: str, memories: list[dict]) -> str:
        """
        Organise retrieved memories into topical clusters with temporal ordering.

        Instead of handing the LLM a flat list, we:
          - Cluster semantically similar memories (cosine > 0.70).
          - Sort each cluster chronologically.
          - Flag multi-session clusters that may contain knowledge updates.

        This pre-structures the reasoning the LLM would otherwise have to do,
        consistently lowering error rates on multi-session and knowledge-update
        question types.
        """
        if not memories:
            return ""

        texts = [m["text"] for m in memories]
        embeddings = self.store.encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

        # Greedy clustering by cosine similarity threshold
        groups: list[list[int]] = []
        assigned: set[int] = set()

        for i in range(len(memories)):
            if i in assigned:
                continue
            group = [i]
            assigned.add(i)
            for j in range(i + 1, len(memories)):
                if j in assigned:
                    continue
                sim = float(embeddings[i] @ embeddings[j])
                if sim > 0.70:
                    group.append(j)
                    assigned.add(j)
            groups.append(group)

        output_parts: list[str] = []
        for group_indices in groups:
            group_mems = [memories[i] for i in group_indices]
            group_mems.sort(key=lambda m: m.get("timestamp", ""))

            if len(group_mems) == 1:
                output_parts.append(f"- {group_mems[0]['text']}")
            else:
                sessions = {m.get("session_id") for m in group_mems}
                is_multi = len(sessions) > 1
                header = (
                    f"[Topic cluster · {len(group_mems)} memories"
                    f" across {len(sessions)} session(s)]"
                )
                if is_multi:
                    header += (
                        "  ⚠️  Contains updates — "
                        "[CURRENT] supersedes [SUPERSEDED] entries"
                    )

                lines = [header]

                if is_multi:
                    max_sid = max(m.get("session_id", 0) for m in group_mems)
                    last_sid = None
                    for m in group_mems:
                        sid = m.get("session_id")
                        if sid != last_sid:
                            ts = m.get("timestamp", "")
                            lines.append(
                                f"  --- Session {sid} ({_human_date(ts)}) ---"
                            )
                            last_sid = sid
                        tag = "[CURRENT]" if sid == max_sid else "[SUPERSEDED]"
                        lines.append(f"  - {tag} {m['text']}")
                else:
                    for m in group_mems:
                        lines.append(f"  - {m['text']}")

                output_parts.append("\n".join(lines))

        return "\n\n".join(output_parts)

    # ------------------------------------------------------------------
    # Convenience: retrieve + structure in one call
    # ------------------------------------------------------------------

    def retrieve_and_structure(
        self,
        query: str,
        top_k: int = 20,
        candidates: int = 50,
    ) -> tuple[list[dict], str]:
        """
        Returns (memories_list, structured_context_string).
        Use *memories_list* for metadata; pass *structured_context_string* to the LLM.
        """
        memories = self.retrieve(query, top_k=top_k, candidates=candidates)
        context = self.structure_context(query, memories)
        return memories, context
