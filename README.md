# memx

A minimal, high-accuracy memory system for long-context AI conversations.
Four components. No gating. No graph databases. No multi-agent pipelines.

## How it works

memx is built on a single insight from the [gated-mem](https://github.com/VihAMBR/gated-mem) experiments:
**the bottleneck is not what you store — it's how you retrieve and reason**.
Weeks of sophisticated encoding gates (surprise gating, neuroplasticity, entity tracking)
produced marginal gains or regressions. A single CoT prompt change produced +5.8% overall
and +40% on preference questions. memx is the distillation of that finding into four clean components:

1. **MemoryStore** — store every message. Embed with `all-MiniLM-L6-v2`. Index with FAISS.
   Zero LLM calls, zero filtering decisions, zero false negatives.

2. **Retriever** — two-stage pipeline: FAISS fetches 50 rough candidates; a local
   `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks them for precision (~10 ms).
   Retrieved memories are then *structured* into topical clusters sorted chronologically
   before being handed to the LLM — so the LLM sees organised evidence, not a flat dump.

3. **SemanticProfile** — one LLM call per session builds and updates a structured JSON
   user model (preferences, facts, beliefs). The profile is injected free at query time,
   giving the LLM pre-digested preference signals without additional API cost.

4. **reason** — five query-adaptive CoT prompt templates:
   `temporal`, `preference`, `knowledge_update`, `counting`, `general`.
   A zero-latency regex classifier routes each query to the right template.

**Cost model:** 0 LLM calls per message at ingestion · 1 per session (profile) · 1 per query.

## Results

| Category | gated-mem (CoT) | memx (target) | What drives the gain |
|---|---|---|---|
| IE-User | 95.7% | 95%+ | Already solved |
| IE-Asst | 98.2% | 98%+ | Already solved |
| Preferences | 76.7% | 85%+ | Semantic profile + preference CoT |
| Multi-session | 63.2% | 70%+ | Context structuring + general CoT |
| Temporal | 72.2% | 78%+ | Cross-encoder rerank + temporal CoT |
| Knowledge-update | 84.6% | 90%+ | Cross-encoder rerank + KU CoT |
| **Overall** | **78.2%** | **83%+** | Everything combined |

Competitive landscape (LongMemEval, 500 instances):

| System | Accuracy |
|---|---|
| Supermemory | 85.86% |
| **memx (target)** | **83%+** |
| Hindsight | ~80% |
| Mastra | ~78% |
| gated-mem | 78.2% |
| Mem0 | ~76% |
| Zep | ~72% |

## Installation

```bash
git clone https://github.com/VihAMBR/memx
cd memx
pip install -e .
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

## Quick start

```python
from memx import MemorySystem

mem = MemorySystem()

# Ingest sessions (one per conversation)
sessions = [
    {
        "session_id": 1,
        "date": "2024-03-01",
        "messages": [
            {"role": "user", "content": "I love hiking, especially in the mountains."},
            {"role": "assistant", "content": "That sounds great! Any favourite trails?"},
            {"role": "user", "content": "I did the Haute Route last summer, it was incredible."},
        ],
    },
    {
        "session_id": 2,
        "date": "2024-06-15",
        "messages": [
            {"role": "user", "content": "I just got back from the Tour du Mont Blanc."},
        ],
    },
]

for s in sessions:
    mem.ingest_session(s["messages"], session_id=s["session_id"], session_date=s["date"])

# Answer questions
print(mem.answer("What hikes has the user done?"))
print(mem.answer("When did the user do the Haute Route?"))
print(mem.answer("What are the user's outdoor preferences?"))
```

## Running benchmarks

```bash
# LongMemEval (requires dataset)
python bench/run_longmemeval.py \
    --data path/to/longmemeval_oracle.json \
    --output results/memx_run.json \
    --n 500

python bench/eval_longmemeval.py \
    --predictions results/memx_run.json \
    --output results/memx_scores.json

# LoCoMo
python bench/run_locomo.py \
    --data dataset/locomo10.json \
    --output results/memx_locomo.json
```

## Using a custom LLM

```python
import anthropic
from memx import MemorySystem

client = anthropic.Anthropic()

def claude_fn(prompt: str) -> str:
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

mem = MemorySystem(llm_fn=claude_fn)
```

## Project structure

```
memx/
├── memx/
│   ├── __init__.py
│   ├── store.py      # MemoryStore: embed + FAISS
│   ├── retrieve.py   # Retriever: rerank + context structuring
│   ├── reason.py     # Query classifier + CoT prompt templates
│   ├── profile.py    # SemanticProfile: lazy user model
│   └── system.py     # MemorySystem: ties everything together
├── bench/
│   ├── run_longmemeval.py
│   ├── eval_longmemeval.py
│   └── run_locomo.py
├── requirements.txt
├── pyproject.toml
└── .env.example
```
