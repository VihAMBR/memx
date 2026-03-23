"""
Semantic profile: one LLM call per session to maintain a structured user model.

Cost model:
  - Ingestion: 1 LLM call per session (not per message).
  - Query time: 0 additional calls — the profile is pre-computed.

The profile is injected as ~500 tokens of structured JSON into every query prompt.
This gives the LLM pre-digested preference/belief information that it would
otherwise have to piece together from 20 raw memory snippets.

gated-mem experiments showed that forcing the LLM to enumerate preference signals
at query time added +40% on preference questions. The semantic profile offloads
that enumeration to ingestion time, making it free at query time.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class SemanticProfile:
    """
    Maintains a structured JSON profile of the user, updated once per session.

    The profile tracks:
      - Preferences and opinions (explicit and implicit)
      - Factual facts: job, location, relationships, interests
      - Beliefs and values
      - Changes over time (what superseded what)
    """

    def __init__(self) -> None:
        self.profile: dict = {}

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_after_session(
        self,
        session_messages: list[dict],
        session_date: str,
        llm_fn,  # callable(prompt: str) -> str
    ) -> None:
        """
        Update the profile from a completed session.

        Args:
            session_messages: List of {"role": str, "content": str} dicts.
            session_date: ISO-format date string for the session.
            llm_fn: Any callable that takes a prompt string and returns a string.
                    Typically wraps your OpenAI / Anthropic client.
        """
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
        except json.JSONDecodeError:
            logger.warning(
                "SemanticProfile: LLM returned non-JSON response; profile not updated.\n"
                "Raw response: %s",
                response[:200],
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import pathlib

        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.profile, indent=2))

    def load(self, path: str) -> None:
        import pathlib

        p = pathlib.Path(path)
        if p.exists():
            self.profile = json.loads(p.read_text())
