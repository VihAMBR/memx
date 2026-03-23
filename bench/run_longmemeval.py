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

from memx import MemorySystem


def load_dataset(path: str, n: int | None = None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if n:
        data = data[:n]
    return data


def run_instance(instance: dict, model: str) -> dict:
    """Run a single LongMemEval instance through memx."""
    mem = MemorySystem(llm_model=model, update_profile=True)

    # Ingest all sessions
    for session in instance.get("sessions", []):
        sid = session.get("session_id", 0)
        date = session.get("date", "2024-01-01")
        messages = session.get("messages", [])
        mem.ingest_session(messages, session_id=sid, session_date=date)

    # Answer the question
    question = instance["question"]
    try:
        predicted = mem.answer(question)
    except Exception as e:
        predicted = f"ERROR: {e}"

    return {
        "question": question,
        "question_type": instance.get("question_type", "unknown"),
        "gold_answer": instance.get("answer", ""),
        "predicted": predicted,
        "num_sessions": len(instance.get("sessions", [])),
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

    # Resume support
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

        result = run_instance(instance, model=args.model)
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
