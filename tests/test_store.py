"""Tests for MemoryStore + SQLite persistence."""
import os
import tempfile
import pytest
from memx.db import MemoryDB
from memx.store import MemoryStore


@pytest.fixture
def tmp_db():
    """Create a MemoryDB in a temp directory, clean up after."""
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB("test_user", db_dir=d)
        yield db, d
        db.close()


@pytest.fixture
def store(tmp_db):
    db, _ = tmp_db
    return MemoryStore(db)


class TestMemoryStore:
    def test_add_persists_to_db(self, store, tmp_db):
        db, _ = tmp_db
        store.add("hello world", "2024-01-01", "user", 1)
        assert len(store.memories) == 1
        assert db.memory_count() == 1

    def test_search_returns_results(self, store):
        store.add("I love Italian food", "2024-01-01", "user", 1)
        store.add("The weather is nice today", "2024-01-02", "user", 1)
        store.add("I had pasta for dinner", "2024-01-03", "user", 1)

        results = store.search("What food does the user like?", top_k=2)
        assert len(results) == 2
        # Food-related memories should rank higher
        texts = [r[0]["text"] for r in results]
        assert any("Italian" in t or "pasta" in t for t in texts)

    def test_search_empty_store(self, store):
        results = store.search("anything", top_k=5)
        assert results == []

    def test_lazy_index_rebuild(self, store):
        store.add("first message", "2024-01-01", "user", 1)
        store.search("test")  # builds index
        assert not store._dirty

        store.add("second message", "2024-01-02", "user", 1)
        assert store._dirty  # dirty again after add

        store.search("test")  # rebuilds
        assert not store._dirty

    def test_memories_survive_reload(self, tmp_db):
        db, d = tmp_db
        store1 = MemoryStore(db)
        store1.add("persistent memory", "2024-01-01", "user", 1)
        store1.add("another memory", "2024-01-02", "assistant", 1)

        # Create a new store from the same DB
        store2 = MemoryStore(db)
        assert len(store2.memories) == 2
        assert store2.memories[0]["text"] == "persistent memory"


class TestMemoryDB:
    def test_insert_and_retrieve(self, tmp_db):
        db, _ = tmp_db
        db.insert_memory("hello", "2024-01-01", "user", 1)
        memories = db.get_all_memories()
        assert len(memories) == 1
        assert memories[0]["text"] == "hello"

    def test_profile_roundtrip(self, tmp_db):
        db, _ = tmp_db
        db.set_profile({"name": "Alice", "city": "SF"})
        profile = db.get_profile()
        assert profile["name"] == "Alice"
        assert profile["city"] == "SF"

    def test_session_upsert(self, tmp_db):
        db, _ = tmp_db
        db.upsert_session(1, started_at="2024-01-01", msg_count=5)
        db.upsert_session(1, ended_at="2024-01-01", msg_count=10)
        assert db.get_latest_session_id() == 1

    def test_empty_profile(self, tmp_db):
        db, _ = tmp_db
        assert db.get_profile() == {}
