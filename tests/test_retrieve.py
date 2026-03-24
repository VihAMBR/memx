"""Tests for Retriever: reranking and context structuring."""
import tempfile
import pytest
from memx.db import MemoryDB
from memx.store import MemoryStore
from memx.retrieve import Retriever


@pytest.fixture
def retriever():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB("test_user", db_dir=d)
        store = MemoryStore(db)

        # Add a variety of memories
        store.add("2024-01-01 | user: I moved to San Francisco", "2024-01-01", "user", 1)
        store.add("2024-01-01 | user: I work at a startup", "2024-01-01", "user", 1)
        store.add("2024-02-01 | user: I love hiking in Marin", "2024-02-01", "user", 2)
        store.add("2024-02-01 | user: I tried a new trail last weekend", "2024-02-01", "user", 2)
        store.add("2024-03-01 | user: I got a dog named Max", "2024-03-01", "user", 3)
        store.add("2024-03-01 | user: Max loves the park", "2024-03-01", "user", 3)
        store.add("2024-04-01 | user: I switched jobs to Google", "2024-04-01", "user", 4)

        ret = Retriever(store)
        yield ret
        db.close()


class TestRetriever:
    def test_retrieve_returns_results(self, retriever):
        results = retriever.retrieve("Where does the user live?", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, dict) for r in results)

    def test_retrieve_respects_top_k(self, retriever):
        results = retriever.retrieve("Tell me everything", top_k=2, candidates=5)
        assert len(results) <= 2

    def test_reranking_improves_relevance(self, retriever):
        results = retriever.retrieve("What kind of pet does the user have?", top_k=3)
        texts = [r["text"] for r in results]
        # "dog" or "Max" should appear in top results
        assert any("dog" in t or "Max" in t for t in texts)


class TestContextStructuring:
    def test_structure_context_produces_output(self, retriever):
        memories = retriever.retrieve("hiking", top_k=5)
        structured = retriever.structure_context("hiking", memories)
        assert len(structured) > 0

    def test_structure_context_empty(self, retriever):
        assert retriever.structure_context("test", []) == ""

    def test_retrieve_and_structure(self, retriever):
        memories, context = retriever.retrieve_and_structure("dog", top_k=3)
        assert len(memories) <= 3
        assert isinstance(context, str)
