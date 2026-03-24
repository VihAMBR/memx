"""
Basic memx usage — no LLM required.

Demonstrates: add messages, end session, retrieve context.
"""
from memx import MemorySystem

# Create a memory system for a user.
# Memories persist to ~/.memx/alice.db (SQLite).
mem = MemorySystem(user_id="alice")

# Add messages one at a time — just like your agent would.
mem.add("user", "I just moved to San Francisco from New York")
mem.add("assistant", "Welcome to SF! What brought you here?")
mem.add("user", "Got a new job at a startup in SOMA")
mem.add("assistant", "That's exciting! What kind of startup?")
mem.add("user", "It's an AI company, working on language models")

# End the session — this triggers a profile update if an LLM is configured.
mem.end_session()

# Later: retrieve context for your own prompt (no LLM call, no API cost).
context = mem.get_context("Where does the user live?")
print("=== Context for 'Where does the user live?' ===")
print(context)
print()

context = mem.get_context("What does the user do for work?")
print("=== Context for 'What does the user do for work?' ===")
print(context)
print()

# You can also search raw memories for debugging.
print(f"Total memories stored: {mem.memory_count}")
results = mem.search("San Francisco", top_k=3)
for r in results:
    print(f"  - {r['text']}")

mem.close()
