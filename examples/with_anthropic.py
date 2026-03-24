"""
memx with Anthropic Claude as the LLM backend.

Demonstrates: custom LLM function, profile updates, answer().
"""
import anthropic
from memx import MemorySystem

# Create a Claude wrapper that matches memx's expected signature: (str) -> str
client = anthropic.Anthropic()


def claude(prompt: str) -> str:
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# Pass the LLM function to memx.
# With an LLM configured, end_session() updates the semantic profile
# and answer() becomes available.
mem = MemorySystem(user_id="bob", llm=claude)

# Simulate two sessions
mem.add("user", "I love Italian food, especially pasta carbonara")
mem.add("assistant", "Great taste! Have you tried any good spots in your area?")
mem.add("user", "There's an amazing place in North Beach called Trattoria")
mem.end_session()

# Second session, a week later
mem.add("user", "I've been cooking more at home lately", timestamp="2024-03-15T19:00:00")
mem.add("assistant", "What have you been making?")
mem.add("user", "Mostly Japanese food actually, I'm into ramen now", timestamp="2024-03-15T19:01:00")
mem.end_session()

# The profile should now show the preference evolution
print("=== Profile ===")
print(mem.profile)
print()

# get_context() works without an LLM call
context = mem.get_context("What food does the user prefer?")
print("=== Context ===")
print(context)
print()

# answer() uses the LLM to generate a response
answer = mem.answer("What food does the user prefer?")
print("=== Answer ===")
print(answer)

mem.close()
