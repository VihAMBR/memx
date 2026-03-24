"""
memx — a minimal, high-accuracy memory system for long-context AI conversations.

Quick start:
    from memx import MemorySystem

    mem = MemorySystem(user_id="user_123")
    mem.add("user", "I just moved to San Francisco")
    mem.add("assistant", "Welcome to SF!")
    mem.end_session()

    context = mem.get_context("Where does the user live?")
"""

from .system import MemorySystem
from .store import MemoryStore
from .retrieve import Retriever
from .reason import classify_query, build_prompt, PROMPTS
from .profile import SemanticProfile
from .db import MemoryDB

__version__ = "0.1.0"
__all__ = [
    "MemorySystem",
    "MemoryStore",
    "Retriever",
    "classify_query",
    "build_prompt",
    "PROMPTS",
    "SemanticProfile",
    "MemoryDB",
]
