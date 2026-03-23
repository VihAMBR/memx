"""
MemorySystem: ties store → retrieve → reason → profile into one object.

Usage example:

    from memx import MemorySystem

    mem = MemorySystem()

    # Ingest a session (list of {"role": ..., "content": ...} dicts)
    mem.ingest_session(messages, session_id=1, session_date="2024-06-01")

    # Answer a question
    answer = mem.answer("What does the user prefer for breakfast?")
    print(answer)
"""
from __future__ import annotations

import os
import logging
from typing import Callable

from .store import MemoryStore
from .retrieve import Retriever
from .reason import classify_query, build_prompt
from .profile import SemanticProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default LLM wrapper (OpenAI)
# ---------------------------------------------------------------------------

def _default_llm_fn(model: str = "gpt-4o-mini") -> Callable[[str], str]:
    """
    Returns a simple OpenAI chat-completion wrapper.
    Reads OPENAI_API_KEY from the environment.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai package required: pip install openai") from e

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    return call


# ---------------------------------------------------------------------------
# MemorySystem
# ---------------------------------------------------------------------------

class MemorySystem:
    """
    Four-component memory system:

      1. MemoryStore   — embed and store every message (no filtering)
      2. Retriever     — FAISS candidates + cross-encoder rerank + context structuring
      3. SemanticProfile — one LLM call per session to maintain a user model
      4. reason        — query-adaptive CoT prompts
    """

    def __init__(
        self,
        encoder_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model: str = "gpt-4o-mini",
        llm_fn: Callable[[str], str] | None = None,
        update_profile: bool = True,
    ):
        """
        Args:
            encoder_model:  SentenceTransformer model for dense retrieval.
            reranker_model: CrossEncoder model for precision reranking.
            llm_model:      OpenAI model name (used if llm_fn is None).
            llm_fn:         Custom LLM callable: (prompt: str) -> str.
                            Supply this to use Anthropic, local models, etc.
            update_profile: Whether to run the profile update after each session.
                            Disable for offline / no-LLM-budget runs.
        """
        self.store = MemoryStore(model_name=encoder_model)
        self.retriever = Retriever(self.store, reranker_model=reranker_model)
        self.profile_engine = SemanticProfile()
        self.update_profile_enabled = update_profile

        if llm_fn is not None:
            self.llm = llm_fn
        else:
            self.llm = _default_llm_fn(model=llm_model)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_session(
        self,
        messages: list[dict],
        session_id: int,
        session_date: str,
    ) -> None:
        """
        Ingest all messages from a single conversation session.

        Cost:
          - 0 LLM calls for storage (pure embedding, local).
          - 1 LLM call for the semantic profile update (if enabled).

        Args:
            messages:     List of {"role": str, "content": str}.
            session_id:   Monotonically increasing integer.
            session_date: ISO date string, e.g. "2024-06-01".
        """
        for msg in messages:
            text = f"{session_date} | {msg['role']}: {msg['content']}"
            self.store.add(
                text=text,
                timestamp=session_date,
                role=msg["role"],
                session_id=session_id,
            )

        if self.update_profile_enabled and messages:
            try:
                self.profile_engine.update_after_session(
                    session_messages=messages,
                    session_date=session_date,
                    llm_fn=self.llm,
                )
            except Exception:
                logger.exception("Profile update failed; continuing without it.")

    def ingest_message(
        self,
        text: str,
        role: str,
        session_id: int,
        timestamp: str,
    ) -> None:
        """
        Add a single message (for streaming / real-time ingestion).
        Profile update is NOT triggered here — call update_profile() explicitly
        at session end.
        """
        self.store.add(text=text, timestamp=timestamp, role=role, session_id=session_id)

    def update_profile(self, messages: list[dict], session_date: str) -> None:
        """Manually trigger a profile update (use with ingest_message)."""
        self.profile_engine.update_after_session(messages, session_date, self.llm)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        top_k: int = 20,
        candidates: int = 50,
        return_prompt: bool = False,
    ) -> str | tuple[str, str]:
        """
        Answer a question using retrieved memories and CoT prompting.

        Cost: 1 LLM call (plus local cross-encoder reranking, ~10 ms).

        Args:
            question:      The question to answer.
            top_k:         How many memories to pass to the LLM.
            candidates:    How many FAISS candidates to rerank from.
            return_prompt: If True, return (answer, prompt) for debugging.
        """
        if not self.store.memories:
            answer = "No memories stored yet."
            return (answer, "") if return_prompt else answer

        # 1. Retrieve + structure
        memories, structured_context = self.retriever.retrieve_and_structure(
            question, top_k=top_k, candidates=candidates
        )

        # 2. Classify query
        query_type = classify_query(question)

        # 3. Build prompt
        prompt = build_prompt(
            question=question,
            memories_context=structured_context,
            profile_section=self.profile_engine.format_for_context(),
            query_type=query_type,
        )

        # 4. Call LLM
        answer = self.llm(prompt)

        if return_prompt:
            return answer, prompt
        return answer

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def retrieve_raw(self, query: str, top_k: int = 10) -> list[dict]:
        """Return raw memory dicts (no LLM call) — useful for debugging."""
        return self.retriever.retrieve(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save the memory store and profile to disk."""
        import pathlib

        d = pathlib.Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.store.save(str(d / "store"))
        self.profile_engine.save(str(d / "profile.json"))
        logger.info("MemorySystem saved to %s", directory)

    def load(self, directory: str) -> None:
        """Load a previously saved memory system."""
        import pathlib

        d = pathlib.Path(directory)
        self.store.load(str(d / "store"))
        self.profile_engine.load(str(d / "profile.json"))
        logger.info("MemorySystem loaded from %s", directory)
