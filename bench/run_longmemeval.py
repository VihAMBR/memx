"""
Run memx against the LongMemEval benchmark.

Usage:
    python bench/run_longmemeval.py \\
        --data path/to/longmemeval_oracle.json \\
        --output results/memx_run.json \\
        --n 500 \\
        --model gpt-4o-mini

LongMemEval format expected:
    [
      {
        "question": "...",
        "answer": "...",
        "question_type": "single-session-user" | "temporal-reasoning" | ...,
        "sessions": [
          {
            "session_id": 1,
            "date": "2024-01-15",
            "messages": [{"role": "user"|"assistant", "content": "..."}]
          }
        ]
      }
    ]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import pathlib
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).parent.parent / ".env")
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    print("openai package required: pip install openai")
    sys.exit(1)

from sentence_transformers import SentenceTransformer, CrossEncoder
from memx import MemorySystem


def load_dataset(path: str, n: int | None = None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if n:
        data = data[:n]
    return data


def _make_llm(client: OpenAI, model: str):
    def llm_fn(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
    return llm_fn


def _parse_longmemeval_date(raw: str) -> str:
    """Convert '2023/04/10 (Mon) 17:50' → '2023-04-10'."""
    date_part = raw.split("(")[0].strip()
    return date_part.replace("/", "-")


def run_instance(
    instance: dict,
    llm_fn,
    instance_id: int,
    encoder=None,
    reranker=None,
) -> dict:
    """Run a single LongMemEval instance through memx."""
    mem = MemorySystem(
        user_id=f"longmemeval_{instance_id}",
        db_path="~/.memx/bench",
        llm=llm_fn,
        encoder_model=encoder or "all-MiniLM-L6-v2",
        reranker_model=reranker or "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    sessions = instance.get("haystack_sessions", [])
    dates = instance.get("haystack_dates", [])
    session_ids = instance.get("haystack_session_ids", [])

    for idx, msgs_raw in enumerate(sessions):
        sid = idx + 1
        date = _parse_longmemeval_date(dates[idx]) if idx < len(dates) else "2024-01-01"
        messages = [{"role": m["role"], "content": m["content"]} for m in msgs_raw]
        mem.ingest_session(messages, session_id=sid, session_date=date)

    question = instance["question"]
    raw_qdate = instance.get("question_date", "")
    q_date = _parse_longmemeval_date(raw_qdate) if raw_qdate else None

    try:
        predicted = mem.answer(question, question_date=q_date)
    except Exception as e:
        predicted = f"ERROR: {e}"

    mem.close()

    return {
        "question": question,
        "question_type": instance.get("question_type", "unknown"),
        "gold_answer": instance.get("answer", ""),
        "predicted": predicted,
        "num_sessions": len(sessions),
    }


def main():
    parser = argparse.ArgumentParser(description="Run memx on LongMemEval")
    parser.add_argument("--data", required=True, help="Path to LongMemEval JSON file")
    parser.add_argument("--output", default="results/memx_longmemeval.json")
    parser.add_argument("--n", type=int, default=None, help="Max instances to run")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    dataset = load_dataset(args.data, args.n)
    print(f"Loaded {len(dataset)} instances from {args.data}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    llm_fn = _make_llm(client, args.model)

    print("Loading embedding + reranker models (one-time)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("Models loaded.")

    results: list[dict] = []
    done_ids: set[int] = set()
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done_ids = set(range(len(results)))
        print(f"Resuming: {len(results)} instances already done")

    start = time.time()
    for i, instance in enumerate(tqdm(dataset, desc="Running memx")):
        if i in done_ids:
            continue

        result = run_instance(
            instance, llm_fn=llm_fn, instance_id=i,
            encoder=encoder, reranker=reranker,
        )
        result["instance_id"] = i
        results.append(result)

        # Save incrementally
        if (i + 1) % 10 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    # Final save
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start
    print(f"\nDone. {len(results)} instances in {elapsed:.1f}s → {out_path}")


if __name__ == "__main__":
    main()
