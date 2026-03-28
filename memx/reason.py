"""
Query classification and chain-of-thought prompt templates.

Classification is regex-based (no LLM call, no latency).
Each template enforces the specific reasoning strategy that proved effective
in gated-mem experiments:

  - temporal:          chronological ordering + date arithmetic
  - preference:        enumerate signals, weight recency
  - knowledge_update:  most-recent-wins resolution
  - counting:          explicit enumeration before totalling
  - general:           connect facts across sessions
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Query classifier
# ---------------------------------------------------------------------------

_TEMPORAL_SIGNALS = [
    "when", "how long", "last time", "most recent", "first time",
    "before", "after", "during", "date", "month", "year",
    "how many days", "how many weeks", "how many months",
    "since", "until", "ago", "start", "end",
]

_PREFERENCE_SIGNALS = [
    "prefer", "like", "favorite", "favour", "enjoy", "opinion",
    "feel about", "think about", "recommend", "suggestion",
    "taste", "style", "interested in", "passion",
]

_UPDATE_SIGNALS = [
    "still", "current", "now", "change", "update", "move", "switch",
    "new", "latest", "anymore", "used to", "recently changed",
    "no longer", "switched to",
]

_COUNT_SIGNALS = [
    "how many", "how much", "list all", "what are all",
    "count", "total", "number of",
]


def classify_query(query: str) -> str:
    """
    Classify a query into one of five categories using keyword matching.

    Returns one of: 'temporal', 'preference', 'knowledge_update', 'counting', 'general'.

    Precedence order matters: temporal before preference because "when did she
    start preferring X?" is temporal, not preference.
    """
    q = query.lower()

    if any(s in q for s in _TEMPORAL_SIGNALS):
        return "temporal"
    if any(s in q for s in _PREFERENCE_SIGNALS):
        return "preference"
    if any(s in q for s in _UPDATE_SIGNALS):
        return "knowledge_update"
    if any(s in q for s in _COUNT_SIGNALS):
        return "counting"
    return "general"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_BASE_HEADER = """\
You have access to memories from past conversations with a user.
Answer using ONLY the evidence in these memories — do not invent information.

IMPORTANT: Memories may be tagged [CURRENT] or [SUPERSEDED].
[CURRENT] = the latest known information. [SUPERSEDED] = replaced by a newer fact.
When both exist for the same topic, ALWAYS use [CURRENT] and ignore [SUPERSEDED].\
"""

PROMPTS: dict[str, str] = {
    "temporal": _BASE_HEADER + """

Memories (retrieved by relevance, grouped by topic):
{memories}

{profile_section}
{question_date}Question: {question}

Think step by step:
1. List every memory that mentions a relevant time, date, or event.
2. Sort these events chronologically using the session dates provided.
3. If any memories are tagged [SUPERSEDED], they mark a PREVIOUS state. [CURRENT] marks the latest state. Use both to establish the timeline of changes.
4. If the question asks "when", identify the specific date from the session headers.
5. If the question asks "how long ago" or "how many months since", compute from the event date to Today's date shown above.
6. Show your date arithmetic explicitly (e.g. "March 2024 → August 2024 = 5 months").
7. Give your final answer based only on the chronological evidence.

Answer:""",

    "preference": _BASE_HEADER + """

Memories (retrieved by relevance, grouped by topic):
{memories}

{profile_section}
{question_date}Question: {question}

Think step by step:
1. Check the User Profile above first — it contains pre-distilled preference data.
2. List every memory where the user expresses a preference, opinion, or makes a choice.
3. Note whether each signal is explicit ("I love X") or implicit (user chose X, reacted positively to X).
4. If any preferences are tagged [SUPERSEDED], they were replaced. Use ONLY [CURRENT] preferences as the user's actual stance.
5. Look for patterns across multiple memories and sessions.
6. Synthesise into a clear answer weighted by [CURRENT] tags first, then recency and frequency.

Answer:""",

    "knowledge_update": _BASE_HEADER + """

Memories (retrieved by relevance, grouped by topic):
{memories}

{profile_section}
{question_date}Question: {question}

Think step by step:
1. Check the User Profile above first — especially the "changes_log" for a history of updates.
2. List every memory relevant to this topic, with its session date.
3. Check for [SUPERSEDED] and [CURRENT] tags. [SUPERSEDED] = old, replaced information. [CURRENT] = the answer you should use.
4. If tagged memories exist, the [CURRENT] entry is the definitive answer. State what changed: "[SUPERSEDED] said X, but [CURRENT] says Y."
5. If no tags exist, sort by date and use the most recent memory.
6. Answer with the [CURRENT] or most recent information ONLY.

Answer:""",

    "counting": _BASE_HEADER + """

Memories (retrieved by relevance, grouped by topic):
{memories}

{profile_section}
{question_date}Question: {question}

Think step by step:
1. List every distinct item that matches what the question is asking about.
2. Number each one explicitly: 1. … 2. … 3. …
3. If an item is tagged [SUPERSEDED], it may no longer apply — check if a [CURRENT] entry replaces it. Only count items that are still valid.
4. Check for duplicates — the same item mentioned across sessions counts once.
5. Give the final count and the complete list.

Answer:""",

    "general": _BASE_HEADER + """

Memories (retrieved by relevance, grouped by topic):
{memories}

{profile_section}
{question_date}Question: {question}

Think step by step:
1. Identify every memory directly relevant to this question.
2. If the answer requires connecting facts from different sessions, note the session dates and connect the facts chronologically.
3. If there are contradictions, check for [CURRENT] vs [SUPERSEDED] tags. [CURRENT] always wins. If no tags, use the most recent session's information.
4. Give a clear, specific answer grounded in the evidence.

Answer:""",
}


def build_prompt(
    question: str,
    memories_context: str,
    profile_section: str = "",
    query_type: str | None = None,
    question_date: str | None = None,
) -> str:
    """
    Build the full prompt string for an LLM call.

    Args:
        question: The user's question.
        memories_context: Pre-formatted memory string (from Retriever.structure_context).
        profile_section: Optional semantic profile block (from SemanticProfile).
        query_type: Override the auto-classified type if already known.
        question_date: Human-readable date string for "today". Used as temporal anchor.
    """
    qt = query_type or classify_query(question)
    template = PROMPTS.get(qt, PROMPTS["general"])

    ps = ""
    if profile_section:
        ps = f"\nUser Profile (distilled from conversation history):\n{profile_section}\n"

    qd = f"Today's date: {question_date}\n" if question_date else ""

    return template.format(
        memories=memories_context or "(no memories retrieved)",
        profile_section=ps,
        question=question,
        question_date=qd,
    )
