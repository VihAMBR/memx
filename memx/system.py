"""
MemorySystem — the public API for memx.

Usage:
    from memx import MemorySystem

    mem = MemorySystem(user_id="user_123")

    # Add messages one at a time (session management is automatic)
    mem.add("user", "I just moved to San Francisco from New York")
    mem.add("assistant", "Welcome to SF! What brought you here?")
    mem.add("user", "Got a new job at a startup in SOMA")
    mem.end_session()

    # Get structured context for your own prompt (no LLM call)
    context = mem.get_context("Where does the user live?")

    # Or get a full answer (1 LLM call)
    answer = mem.answer("Where does the user live?")
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable

from .db import MemoryDB
from .store import MemoryStore
from .retrieve import Retriever
from .reason import classify_query, build_prompt
from .profile import SemanticProfile

logger = logging.getLogger(__name__)

LLMFunction = Callable[[str], str]

# Default session gap: 30 minutes of inactivity starts a new session.
_DEFAULT_SESSION_GAP_MINUTES = 30


class MemorySystem:
    """
    Four-component memory system with a message-level API.

    Components:
      1. MemoryStore      — store every message, embed with SentenceTransformer, FAISS index
      2. Retriever        — FAISS candidates → cross-encoder rerank → context structuring
      3. SemanticProfile  — one LLM call per session; injected free at query time
      4. reason           — query-adaptive CoT prompts

    The primary method is ``get_context(query)`` — returns structured memories + profile
    as a string ready to inject into any LLM prompt. No LLM call required.

    ``answer(query)`` is a convenience wrapper that calls ``get_context()`` and passes
    the result through an LLM. Requires an LLM function.

    Persistence is automatic via SQLite. Memories survive process restarts.
    """

    def __init__(
        self,
        user_id: str = "default",
        db_path: str = "~/.memx",
        encoder_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm: LLMFunction | None = None,
        session_gap_minutes: int = _DEFAULT_SESSION_GAP_MINUTES,
    ):
        """
        Args:
            user_id:             Unique identifier for this user's memory store.
            db_path:             Directory for SQLite databases. One file per user.
            encoder_model:       SentenceTransformer model for dense retrieval.
            reranker_model:      CrossEncoder model for precision reranking.
            llm:                 Optional callable (prompt: str) -> str.
                                 Required for answer() and profile updates.
                                 Pass None for pure retrieval (no API costs).
            session_gap_minutes: Minutes of inactivity before auto-starting a new session.
        """
        self._db = MemoryDB(user_id, db_path)
        self.store = MemoryStore(self._db, model_name=encoder_model)
        self.retriever = Retriever(self.store, reranker_model=reranker_model)
        self.profile_engine = SemanticProfile(self._db)
        self.llm = llm
        self._session_gap = timedelta(minutes=session_gap_minutes)

        # Session state
        self._current_session_id: int = self._db.get_latest_session_id() or 1
        self._current_session_messages: list[dict] = []
        self._last_message_time: datetime | None = None
        self._session_start_time: str | None = None

    # ------------------------------------------------------------------
    # Message-level API
    # ------------------------------------------------------------------

    def add(
        self,
        role: str,
        content: str,
        timestamp: str | None = None,
    ) -> None:
        """
        Add a single message. Session management is automatic.

        If more than ``session_gap_minutes`` have passed since the last message,
        a new session is started (and the previous session's profile is updated
        if an LLM is configured).

        Args:
            role:      "user" or "assistant".
            content:   The message text.
            timestamp: ISO-format string. Defaults to now.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        now = datetime.fromisoformat(timestamp)

        # Auto-detect session boundaries
        if self._should_start_new_session(now):
            self._end_current_session()
            self._current_session_id += 1

        if self._session_start_time is None:
            self._session_start_time = timestamp
            self._db.upsert_session(
                self._current_session_id, started_at=timestamp, msg_count=0,
            )

        # Store the message
        text = f"{timestamp} | {role}: {content}"
        self.store.add(
            text=text,
            timestamp=timestamp,
            role=role,
            session_id=self._current_session_id,
        )
        self._current_session_messages.append({"role": role, "content": content})
        self._last_message_time = now

        # Update session metadata
        self._db.upsert_session(
            self._current_session_id,
            msg_count=len(self._current_session_messages),
        )

    def end_session(self) -> None:
        """
        Explicitly end the current session.

        Triggers a profile update if an LLM function is configured.
        Call this at the end of a conversation. If you forget, the next
        ``add()`` call will auto-detect the boundary.
        """
        self._end_current_session()
        self._current_session_id += 1

    # ------------------------------------------------------------------
    # Retrieval — the primary interface
    # ------------------------------------------------------------------

    def get_context(
        self,
        query: str,
        top_k: int = 20,
        candidates: int = 50,
    ) -> str:
        """
        Retrieve structured memory context for a query. **No LLM call.**

        Returns a formatted string containing:
          - The semantic profile (if available)
          - Retrieved memories, clustered by topic and sorted chronologically

        This string is ready to inject into any LLM prompt as context.

        Args:
            query:      The question or topic to retrieve memories for.
            top_k:      Number of memories to include after reranking.
            candidates: Number of FAISS candidates to rerank from.
        """
        if not self.store.memories:
            return ""

        _, structured = self.retriever.retrieve_and_structure(
            query, top_k=top_k, candidates=candidates,
        )

        parts: list[str] = []

        profile_ctx = self.profile_engine.format_for_context()
        if profile_ctx:
            parts.append(f"User Profile:\n{profile_ctx}")

        if structured:
            parts.append(f"Relevant Memories:\n{structured}")

        return "\n\n".join(parts)

    def answer(
        self,
        question: str,
        top_k: int = 20,
        candidates: int = 50,
    ) -> str:
        """
        Answer a question using retrieved memories and CoT prompting.

        Requires an LLM function (passed at init or via ``self.llm``).
        Cost: 1 LLM call + local cross-encoder reranking (~10 ms).

        For most use cases, prefer ``get_context()`` and handle the LLM call yourself.
        """
        if self.llm is None:
            raise RuntimeError(
                "answer() requires an LLM function. "
                "Pass llm=my_function when creating MemorySystem, "
                "or use get_context() for LLM-free retrieval."
            )

        if not self.store.memories:
            return "No memories stored yet."

        _, structured = self.retriever.retrieve_and_structure(
            question, top_k=top_k, candidates=candidates,
        )

        query_type = classify_query(question)
        prompt = build_prompt(
            question=question,
            memories_context=structured,
            profile_section=self.profile_engine.format_for_context(),
            query_type=query_type,
        )

        return self.llm(prompt)

    # ------------------------------------------------------------------
    # Bulk ingestion (for benchmarks / migration)
    # ------------------------------------------------------------------

    def ingest_session(
        self,
        messages: list[dict],
        session_id: int,
        session_date: str,
    ) -> None:
        """
        Ingest a complete session at once. Useful for benchmarks and data migration.

        Args:
            messages:     List of {"role": str, "content": str}.
            session_id:   Integer session identifier.
            session_date: ISO date string.
        """
        for msg in messages:
            text = f"{session_date} | {msg['role']}: {msg['content']}"
            self.store.add(text, session_date, msg["role"], session_id)

        self._db.upsert_session(
            session_id, started_at=session_date, msg_count=len(messages),
        )

        if self.llm and messages:
            try:
                self.profile_engine.update_after_session(
                    messages, session_date, self.llm,
                )
            except Exception:
                logger.exception("Profile update failed for session %s", session_id)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return raw memory dicts (no LLM call). Useful for debugging."""
        return self.retriever.retrieve(query, top_k=top_k)

    @property
    def profile(self) -> dict:
        """Current semantic profile as a dict."""
        return self.profile_engine.profile

    @property
    def memory_count(self) -> int:
        """Total number of stored memories."""
        return len(self.store.memories)

    # ------------------------------------------------------------------
    # Session management internals
    # ------------------------------------------------------------------

    def _should_start_new_session(self, now: datetime) -> bool:
        """Check if enough time has passed to warrant a new session."""
        if self._last_message_time is None:
            return False
        return (now - self._last_message_time) > self._session_gap

    def _end_current_session(self) -> None:
        """Finalize the current session: update profile, reset state."""
        if not self._current_session_messages:
            return

        timestamp = (
            self._last_message_time.isoformat()
            if self._last_message_time
            else datetime.now().isoformat()
        )

        self._db.upsert_session(
            self._current_session_id,
            ended_at=timestamp,
            msg_count=len(self._current_session_messages),
        )

        # Update semantic profile (one LLM call, if configured)
        if self.llm:
            try:
                session_date = self._session_start_time or timestamp
                self.profile_engine.update_after_session(
                    self._current_session_messages, session_date, self.llm,
                )
            except Exception:
                logger.exception(
                    "Profile update failed for session %s",
                    self._current_session_id,
                )

        self._current_session_messages = []
        self._last_message_time = None
        self._session_start_time = None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush any pending session and close the database."""
        self._end_current_session()
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
