"""
Run memx against the BEAM benchmark (100K split from Mohammadta/BEAM).

Usage:
    python bench/run_beam.py --output results/memx_beam.json --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import pathlib
import re

from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import time as _time

try:
    from openai import OpenAI
except ImportError:
    print("openai package required: pip install openai")
    sys.exit(1)

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from memx import MemorySystem


def _retry_call(fn, *args, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = min(2 ** attempt * 5, 120)
                print(f"\n  Rate limited, retrying in {wait}s...")
                _time.sleep(wait)
            else:
                raise
    return fn(*args, **kwargs)


EVAL_TYPES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "knowledge_update",
    "multi_session_reasoning",
    "temporal_reasoning",
]

JUDGE_PROMPT = """You are evaluating whether a predicted answer is correct.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

The predicted answer is CORRECT if it captures the key information in the gold answer.
For abstention questions, the answer is CORRECT if it correctly states the information is unavailable.
Minor wording differences are acceptable.

Reply with a single word: CORRECT or INCORRECT"""


def judge(client, question: str, gold: str, predicted: str) -> bool:
    resp = _retry_call(
        client.chat.completions.create,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, gold=gold, predicted=predicted
        )}],
        temperature=0.0,
        max_tokens=5,
    )
    verdict = (resp.choices[0].message.content or "").strip().upper()
    return verdict.startswith("CORRECT")


def _extract_time_anchor_date(anchor: str) -> str:
    """Extract ISO date from time_anchor like 'March-15-2024'."""
    m = re.match(r"(\w+)-(\d{1,2})-(\d{4})", anchor or "")
    if not m:
        return "2024-01-01"
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    month = months.get(m.group(1).lower(), "01")
    return f"{m.group(3)}-{month}-{int(m.group(2)):02d}"


def main():
    parser = argparse.ArgumentParser(description="Run memx on BEAM")
    parser.add_argument("--output", default="results/memx_beam.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--split", default="100K", choices=["100K", "500K", "1M"])
    args = parser.parse_args()

    print(f"Loading BEAM dataset ({args.split} split)...")
    ds = load_dataset("Mohammadta/BEAM", split=args.split)
    print(f"Loaded {len(ds)} conversations.")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def llm_fn(prompt: str) -> str:
        resp = _retry_call(client.chat.completions.create,
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("Models loaded.")

    all_results = []

    for ci, conv in enumerate(tqdm(ds, desc="BEAM conversations")):
        chat_groups = conv["chat"]

        mem = MemorySystem(
            user_id=f"beam_{ci}",
            db_path="~/.memx/bench_beam",
            llm=llm_fn,
            encoder_model=encoder,
            reranker_model=reranker,
        )

        for gi, group in enumerate(chat_groups):
            first_anchor = group[0].get("time_anchor", "") if group else ""
            session_date = _extract_time_anchor_date(first_anchor)
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in group
            ]
            mem.ingest_session(messages, session_id=gi + 1, session_date=session_date)

        pq_raw = conv["probing_questions"]
        probing = ast.literal_eval(pq_raw) if isinstance(pq_raw, str) else pq_raw

        for qtype in EVAL_TYPES:
            questions = probing.get(qtype, [])
            for q in questions:
                question = q.get("question", "")
                gold = q.get("answer") or q.get("ideal_response") or q.get("ideal_summary", "")

                try:
                    predicted = mem.answer(question)
                except Exception as e:
                    predicted = f"ERROR: {e}"

                result = {
                    "conversation_id": conv.get("conversation_id", f"beam_{ci}"),
                    "question": question,
                    "gold_answer": gold,
                    "predicted": predicted,
                    "question_type": qtype,
                    "correct": judge(client, question, gold, predicted),
                }
                all_results.append(result)

        mem.close()

        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if all_results:
        total = len(all_results)
        correct = sum(r.get("correct", False) for r in all_results)
        print(f"\nBEAM accuracy: {correct}/{total} = {correct / total:.1%}")

        from collections import defaultdict
        by_type = defaultdict(list)
        for r in all_results:
            by_type[r["question_type"]].append(r.get("correct", False))
        print(f"\n{'Category':<30} {'Acc':>8} {'N':>6}")
        print("-" * 50)
        for cat, vals in sorted(by_type.items()):
            acc = sum(vals) / len(vals) if vals else 0
            print(f"{cat:<30} {acc:>7.1%} {len(vals):>6}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
