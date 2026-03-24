"""
Semantic profile: one LLM call per session to maintain a structured user model.

Cost model:
  - Ingestion: 1 LLM call per session (not per message). Skipped if no LLM configured.
  - Query time: 0 additional calls — the profile is pre-computed.

The profile is injected as ~500 tokens of structured JSON into every query prompt.
This gives the LLM pre-digested preference/belief information that it would
otherwise have to piece together from 20 raw memory snippets.
"""
from __future__ import annotations

import json
import logging
from typing import Callable

from .db import MemoryDB

logger = logging.getLogger(__name__)

LLMFunction = Callable[[str], str]


class SemanticProfile:
    """
    Maintains a structured JSON profile of the user, updated once per session.
    Persisted to SQLite via MemoryDB.

    Works without an LLM — the profile simply stays empty. All read methods
    return safe defaults. The profile becomes useful only when an LLM is
    provided and end_session() is called.
    """

    def __init__(self, db: MemoryDB) -> None:
        self.db = db
        self.profile: dict = db.get_profile()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_after_session(
        self,
        session_messages: list[dict],
        session_date: str,
        llm_fn: LLMFunction,
    ) -> None:
        """
        Update the profile from a completed session.

        Args:
            session_messages: List of {"role": str, "content": str} dicts.
            session_date: ISO-format date string for the session.
            llm_fn: Any callable (prompt: str) -> str.
        """
        if not session_messages:
            return

        current_json = (
            json.dumps(self.profile, indent=2) if self.profile else "No profile yet."
        )
        conversation = "\n".join(
            f"{m['role']}: {m['content']}" for m in session_messages
        )

        prompt = f"""You are maintaining a structured profile of a user based on their conversations.

Current profile:
{current_json}

New conversation session ({session_date}):
{conversation}

Update the profile based on the new conversation. Rules:
- Extract preferences, beliefs, habits, and opinions — both explicit and implicit.
- Track factual information: job, location, relationships, interests, hobbies.
- If new information contradicts the existing profile, UPDATE it and note what changed.
- Remove stale information that has been superseded by newer facts.
- Keep the profile under 500 tokens total.
- Output valid JSON only, no explanation, no markdown fences.

Output the complete updated profile as a JSON object:"""

        response = llm_fn(prompt)

        # Strip potential markdown fences
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                ln for ln in lines if not ln.strip().startswith("```")
            )

        try:
            self.profile = json.loads(cleaned)
            self.db.set_profile(self.profile)
        except json.JSONDecodeError:
            logger.warning(
                "SemanticProfile: LLM returned non-JSON; profile not updated. "
                "Response: %.200s",
                response,
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def format_for_context(self) -> str:
        """Return the profile formatted for injection into a prompt."""
        if not self.profile:
            return ""
        return json.dumps(self.profile, indent=2)

    def is_empty(self) -> bool:
        return not bool(self.profile)
