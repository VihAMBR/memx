# memx

A minimal, high-accuracy memory system for AI agents.
Four components. No gating. No graph databases. No multi-agent pipelines.

## Install

```bash
pip install -e .
```

## Quick start

```python
from memx import MemorySystem

mem = MemorySystem(user_id="user_123")

# Add messages — session management is automatic
mem.add("user", "I just moved to San Francisco from New York")
mem.add("assistant", "Welcome to SF! What brought you here?")
mem.add("user", "Got a new job at a startup in SOMA")
mem.end_session()

# Next day, new session
mem.add("user", "The commute from the Mission is brutal")

# Get structured context for your own prompt (no LLM call)
context = mem.get_context("Where does the user live?")

# Or get a full answer (requires an LLM function)
# answer = mem.answer("Where does the user live?")
```

### With an LLM

```python
from openai import OpenAI
from memx import MemorySystem

client = OpenAI()

def llm(prompt: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content

mem = MemorySystem(user_id="user_123", llm=llm)

mem.add("user", "I love hiking, especially in the mountains")
mem.end_session()  # triggers profile update

# answer() uses CoT prompting over retrieved + structured memories
print(mem.answer("What are the user's hobbies?"))
```

Works with any LLM — OpenAI, Anthropic, local models. Just pass a `(str) -> str` callable.

## How it works

Built on a single insight from the [gated-mem](https://github.com/VihAMBR/gated-mem) experiments:
**the bottleneck is not what you store — it's how you retrieve and reason**.

1. **MemoryStore** — store every message. Embed with `all-MiniLM-L6-v2`. Index with FAISS. Zero LLM calls, zero filtering, zero false negatives. Backed by SQLite — memories survive restarts.

2. **Retriever** — FAISS fetches 50 candidates; a local cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks for precision (~10 ms). Retrieved memories are clustered by topic and sorted chronologically before reaching the LLM.

3. **SemanticProfile** — one LLM call per session builds a structured JSON user model. Injected free at query time (~500 tokens). Optional — everything works without it.

4. **reason** — five query-adaptive CoT prompt templates: `temporal`, `preference`, `knowledge_update`, `counting`, `general`. A zero-latency regex classifier routes each query.

### Cost model

| Operation | LLM calls |
|---|---|
| `add()` a message | 0 |
| `end_session()` (profile update) | 1 (optional) |
| `get_context()` | 0 |
| `answer()` | 1 |

## API

### `MemorySystem(user_id, db_path, llm, ...)`

| Param | Default | Description |
|---|---|---|
| `user_id` | `"default"` | Unique user identifier. One SQLite file per user. |
| `db_path` | `"~/.memx"` | Directory for SQLite databases. |
| `llm` | `None` | Optional `(str) -> str` callable. Required for `answer()` and profile updates. |
| `encoder_model` | `"all-MiniLM-L6-v2"` | SentenceTransformer model for embeddings. |
| `reranker_model` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | CrossEncoder for reranking. |
| `session_gap_minutes` | `30` | Minutes of inactivity before auto-starting a new session. |

### Methods

| Method | LLM? | Description |
|---|---|---|
| `add(role, content, timestamp?)` | No | Add a message. Session management is automatic. |
| `end_session()` | Optional | End current session. Triggers profile update if LLM configured. |
| `get_context(query, top_k=20)` | **No** | Return structured memories + profile as a string. The primary interface. |
| `answer(question, top_k=20)` | Yes | Get a full answer using CoT prompting. Convenience wrapper. |
| `search(query, top_k=10)` | No | Return raw memory dicts. For debugging. |
| `ingest_session(messages, session_id, session_date)` | Optional | Bulk ingest. For benchmarks and migration. |

### Context manager

```python
with MemorySystem(user_id="alice") as mem:
    mem.add("user", "Hello")
    context = mem.get_context("greeting")
# Automatically flushes session and closes DB
```

## Persistence

Memories are stored in SQLite (one file per user at `~/.memx/<user_id>.db`). Every `add()` call writes immediately. The FAISS index is rebuilt lazily in memory. If the process crashes, no data is lost.

## Benchmarks

Target numbers on LongMemEval (500 instances):

| Category | gated-mem (CoT) | memx target | What drives the gain |
|---|---|---|---|
| IE-User | 95.7% | 95%+ | Already solved |
| IE-Asst | 98.2% | 98%+ | Already solved |
| Preferences | 76.7% | 85%+ | Semantic profile + preference CoT |
| Multi-session | 63.2% | 70%+ | Context structuring + general CoT |
| Temporal | 72.2% | 78%+ | Cross-encoder rerank + temporal CoT |
| Knowledge-update | 84.6% | 90%+ | Cross-encoder rerank + KU CoT |
| **Overall** | **78.2%** | **83%+** | Everything combined |

```bash
# Run benchmarks
python bench/run_longmemeval.py --data path/to/longmemeval.json --n 500
python bench/eval_longmemeval.py --predictions results/memx_longmemeval.json
```

## Project structure

```
memx/
├── memx/
│   ├── system.py      # MemorySystem — the public API
│   ├── store.py       # MemoryStore — embed + FAISS + SQLite
│   ├── retrieve.py    # Retriever — rerank + context structuring
│   ├── reason.py      # Query classifier + CoT prompt templates
│   ├── profile.py     # SemanticProfile — lazy user model
│   └── db.py          # SQLite persistence layer
├── bench/             # Benchmark runners
├── examples/          # Usage examples
├── tests/             # Unit tests
└── pyproject.toml
```

## License

MIT
