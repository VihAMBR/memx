"""Tests for query classification and prompt building."""
from memx.reason import classify_query, build_prompt


class TestClassifyQuery:
    def test_temporal_signals(self):
        assert classify_query("When did the user move?") == "temporal"
        assert classify_query("How long has she lived there?") == "temporal"
        assert classify_query("What was the most recent trip?") == "temporal"
        assert classify_query("What happened after the meeting?") == "temporal"

    def test_preference_signals(self):
        assert classify_query("What food does the user prefer?") == "preference"
        assert classify_query("What's their favorite color?") == "preference"
        assert classify_query("Does the user enjoy hiking?") == "preference"

    def test_knowledge_update_signals(self):
        assert classify_query("Does the user still live in NYC?") == "knowledge_update"
        assert classify_query("What is their current job?") == "knowledge_update"
        assert classify_query("Did they switch teams?") == "knowledge_update"

    def test_counting_signals(self):
        assert classify_query("How many pets does the user have?") == "counting"
        assert classify_query("List all the cities they visited") == "counting"
        assert classify_query("What is the total count?") == "counting"

    def test_general_fallback(self):
        assert classify_query("Tell me about the user") == "general"
        assert classify_query("What is the capital of France?") == "general"

    def test_temporal_takes_precedence_over_preference(self):
        # "when" + "prefer" → temporal wins
        assert classify_query("When did she start preferring Italian food?") == "temporal"


class TestBuildPrompt:
    def test_includes_question(self):
        prompt = build_prompt("Where does the user live?", "some memories")
        assert "Where does the user live?" in prompt

    def test_includes_memories(self):
        prompt = build_prompt("test?", "MEMORY_CONTENT_HERE")
        assert "MEMORY_CONTENT_HERE" in prompt

    def test_includes_profile_when_provided(self):
        prompt = build_prompt("test?", "memories", profile_section='{"name": "Alice"}')
        assert "Alice" in prompt

    def test_no_profile_when_empty(self):
        prompt = build_prompt("test?", "memories", profile_section="")
        assert "User Profile" not in prompt

    def test_temporal_prompt_has_chronological_instruction(self):
        prompt = build_prompt("When did they move?", "memories", query_type="temporal")
        assert "chronological" in prompt.lower()

    def test_counting_prompt_has_enumerate_instruction(self):
        prompt = build_prompt("How many pets?", "memories", query_type="counting")
        assert "Number each one" in prompt

    def test_empty_memories_shows_placeholder(self):
        prompt = build_prompt("test?", "")
        assert "no memories retrieved" in prompt
