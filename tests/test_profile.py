"""Tests for SemanticProfile."""
import tempfile
import pytest
from memx.db import MemoryDB
from memx.profile import SemanticProfile


@pytest.fixture
def profile():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB("test_user", db_dir=d)
        p = SemanticProfile(db)
        yield p, db
        db.close()


class TestSemanticProfile:
    def test_starts_empty(self, profile):
        p, _ = profile
        assert p.is_empty()
        assert p.format_for_context() == ""

    def test_update_with_llm(self, profile):
        p, _ = profile

        def fake_llm(prompt: str) -> str:
            return '{"location": "San Francisco", "job": "engineer"}'

        messages = [
            {"role": "user", "content": "I live in San Francisco and work as an engineer"},
        ]
        p.update_after_session(messages, "2024-01-01", fake_llm)

        assert not p.is_empty()
        assert p.profile["location"] == "San Francisco"

    def test_profile_persists_to_db(self, profile):
        p, db = profile

        def fake_llm(prompt: str) -> str:
            return '{"name": "Alice"}'

        messages = [{"role": "user", "content": "My name is Alice"}]
        p.update_after_session(messages, "2024-01-01", fake_llm)

        # Read back from DB
        assert db.get_profile()["name"] == "Alice"

    def test_bad_json_doesnt_crash(self, profile):
        p, _ = profile

        def bad_llm(prompt: str) -> str:
            return "This is not JSON at all"

        messages = [{"role": "user", "content": "test"}]
        p.update_after_session(messages, "2024-01-01", bad_llm)

        # Profile should remain empty — no crash
        assert p.is_empty()

    def test_markdown_fences_stripped(self, profile):
        p, _ = profile

        def fenced_llm(prompt: str) -> str:
            return '```json\n{"city": "NYC"}\n```'

        messages = [{"role": "user", "content": "I live in NYC"}]
        p.update_after_session(messages, "2024-01-01", fenced_llm)
        assert p.profile["city"] == "NYC"

    def test_empty_messages_noop(self, profile):
        p, _ = profile

        call_count = 0
        def counting_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return '{}'

        p.update_after_session([], "2024-01-01", counting_llm)
        assert call_count == 0

    def test_format_for_context(self, profile):
        p, _ = profile
        p.profile = {"food": "Italian", "city": "SF"}
        ctx = p.format_for_context()
        assert "Italian" in ctx
        assert "SF" in ctx
