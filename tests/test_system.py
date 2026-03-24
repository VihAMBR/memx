"""Tests for MemorySystem: the public API."""
import tempfile
import pytest
from memx import MemorySystem


@pytest.fixture
def mem():
    """A MemorySystem with no LLM (pure retrieval mode)."""
    with tempfile.TemporaryDirectory() as d:
        m = MemorySystem(user_id="test", db_path=d, llm=None)
        yield m
        m.close()


@pytest.fixture
def mem_with_llm():
    """A MemorySystem with a fake LLM for testing."""
    with tempfile.TemporaryDirectory() as d:
        def fake_llm(prompt: str) -> str:
            if "profile" in prompt.lower() and "JSON" in prompt:
                return '{"name": "Test User", "location": "SF"}'
            return "This is a test answer based on the memories."

        m = MemorySystem(user_id="test_llm", db_path=d, llm=fake_llm)
        yield m
        m.close()


class TestAddAndRetrieve:
    def test_add_single_message(self, mem):
        mem.add("user", "Hello world")
        assert mem.memory_count == 1

    def test_add_multiple_messages(self, mem):
        mem.add("user", "First message")
        mem.add("assistant", "Response")
        mem.add("user", "Follow up")
        assert mem.memory_count == 3

    def test_get_context_returns_string(self, mem):
        mem.add("user", "I live in San Francisco")
        mem.add("user", "I work at a startup")
        context = mem.get_context("Where does the user live?")
        assert isinstance(context, str)
        assert "San Francisco" in context

    def test_get_context_empty_store(self, mem):
        context = mem.get_context("anything")
        assert context == ""

    def test_search_returns_dicts(self, mem):
        mem.add("user", "I have a cat named Luna")
        results = mem.search("pet", top_k=1)
        assert len(results) == 1
        assert "Luna" in results[0]["text"]


class TestSessionManagement:
    def test_end_session_resets(self, mem):
        mem.add("user", "Message 1")
        mem.add("user", "Message 2")
        mem.end_session()
        # After end_session, internal session messages should be cleared
        assert mem._current_session_messages == []

    def test_auto_session_boundary(self, mem):
        mem.add("user", "Morning message", timestamp="2024-01-01T09:00:00")
        first_session = mem._current_session_id

        # 2 hours later — should auto-start new session
        mem.add("user", "Afternoon message", timestamp="2024-01-01T11:00:00")
        assert mem._current_session_id == first_session + 1

    def test_no_boundary_within_gap(self, mem):
        mem.add("user", "Message 1", timestamp="2024-01-01T09:00:00")
        first_session = mem._current_session_id

        # 5 minutes later — same session
        mem.add("user", "Message 2", timestamp="2024-01-01T09:05:00")
        assert mem._current_session_id == first_session


class TestPersistence:
    def test_memories_survive_restart(self):
        with tempfile.TemporaryDirectory() as d:
            # First instance
            mem1 = MemorySystem(user_id="persist_test", db_path=d, llm=None)
            mem1.add("user", "I live in New York")
            mem1.add("user", "I work at a bank")
            mem1.close()

            # Second instance — same user, same db
            mem2 = MemorySystem(user_id="persist_test", db_path=d, llm=None)
            assert mem2.memory_count == 2
            context = mem2.get_context("Where does the user live?")
            assert "New York" in context
            mem2.close()


class TestAnswer:
    def test_answer_requires_llm(self, mem):
        mem.add("user", "test")
        with pytest.raises(RuntimeError, match="requires an LLM"):
            mem.answer("test question")

    def test_answer_with_llm(self, mem_with_llm):
        mem_with_llm.add("user", "I live in San Francisco")
        answer = mem_with_llm.answer("Where?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_answer_empty_store(self, mem_with_llm):
        answer = mem_with_llm.answer("anything")
        assert answer == "No memories stored yet."


class TestProfileUpdate:
    def test_profile_updates_on_end_session(self, mem_with_llm):
        mem_with_llm.add("user", "I just moved to San Francisco")
        mem_with_llm.end_session()
        assert mem_with_llm.profile.get("location") == "SF"

    def test_no_profile_without_llm(self, mem):
        mem.add("user", "I live in SF")
        mem.end_session()
        assert mem.profile == {}


class TestBulkIngest:
    def test_ingest_session(self, mem):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        mem.ingest_session(messages, session_id=10, session_date="2024-06-01")
        assert mem.memory_count == 2


class TestContextManager:
    def test_with_statement(self):
        with tempfile.TemporaryDirectory() as d:
            with MemorySystem(user_id="ctx_test", db_path=d) as mem:
                mem.add("user", "test message")
                assert mem.memory_count == 1
