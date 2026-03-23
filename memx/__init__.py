"""
memx — a minimal, high-accuracy memory system for long-context AI conversations.

Architecture:
  1. MemoryStore      — store every message, embed with SentenceTransformer, index with FAISS
  2. Retriever        — two-stage: FAISS (50 candidates) → cross-encoder rerank (top 20)
  3. SemanticProfile  — one LLM call per session; injected free at query time
  4. reason           — query-adaptive CoT prompts (temporal / preference / KU / counting / general)

Quick start:
    from memx import MemorySystem

    mem = MemorySystem()
    mem.ingest_session(messages, session_id=1, session_date="2024-06-01")
    print(mem.answer("What is the user's favourite food?"))
"""

from .system import MemorySystem
from .store import MemoryStore
from .retrieve import Retriever
from .reason import classify_query, build_prompt, PROMPTS
from .profile import SemanticProfile

__version__ = "0.1.0"
__all__ = [
    "MemorySystem",
    "MemoryStore",
    "Retriever",
    "classify_query",
    "build_prompt",
    "PROMPTS",
    "SemanticProfile",
]
