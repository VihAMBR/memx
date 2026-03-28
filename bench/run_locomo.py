"""
Run memx against the LoCoMo benchmark (locomo10.json from snap-research/locomo).

Usage:
    python bench/run_locomo.py --data data/locomo10.json --output results/memx_locomo.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import pathlib

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


_MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


def _parse_locomo_date(raw: str) -> str:
    """Convert '1:56 pm on 8 May, 2023' → '2023-05-08'."""
    m = re.search(r"(\d{1,2})\s+(\w+),?\s+(\d{4})", raw)
    if not m:
        return "2023-01-01"
    day, month_name, year = m.group(1), m.group(2).lower(), m.group(3)
    month = _MONTHS.get(month_name, "01")
    return f"{year}-{month}-{int(day):02d}"


def _extract_sessions(conv_data: dict) -> list[dict]:
    """Extract ordered sessions from LoCoMo conversation dict."""
    sessions = []
    i = 1
    while True:
        key = f"session_{i}"
        date_key = f"session_{i}_date_time"
        if key not in conv_data:
            break
        turns = conv_data[key]
        date_raw = conv_data.get(date_key, "")
        date_iso = _parse_locomo_date(date_raw)
        speaker_a = conv_data.get("speaker_a", "A")
        messages = []
        for turn in turns:
            role = "user" if turn.get("speaker") == speaker_a else "assistant"
            messages.append({"role": role, "content": turn.get("text", "")})
        sessions.append({"session_id": i, "date": date_iso, "messages": messages})
        i += 1
    return sessions


JUDGE_PROMPT = """You are evaluating whether a predicted answer is correct.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

The predicted answer is CORRECT if it contains all the key information in the gold answer.
Minor wording differences and extra context are acceptable.

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


_LOCOMO_CATS = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-domain", 5: "adversarial"}


def main():
    parser = argparse.ArgumentParser(description="Run memx on LoCoMo")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/memx_locomo.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    with open(args.data) as f:
        dataset = json.load(f)

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

    for ci, conv in enumerate(tqdm(dataset, desc="Conversations")):
        sessions = _extract_sessions(conv["conversation"])

        mem = MemorySystem(
            user_id=f"locomo_{ci}",
            db_path="~/.memx/bench_locomo",
            llm=llm_fn,
            encoder_model=encoder,
            reranker_model=reranker,
        )

        for sess in sessions:
            mem.ingest_session(sess["messages"], sess["session_id"], sess["date"])

        for qa in conv.get("qa", []):
            question = qa["question"]
            gold = qa.get("answer", "")
            cat = qa.get("category", 0)
            cat_name = _LOCOMO_CATS.get(cat, f"cat_{cat}")

            try:
                predicted = mem.answer(question)
            except Exception as e:
                predicted = f"ERROR: {e}"

            result = {
                "conversation_id": conv.get("sample_id", f"conv_{ci}"),
                "question": question,
                "gold_answer": gold,
                "predicted": predicted,
                "question_type": cat_name,
            }

            if not args.no_judge:
                result["correct"] = judge(client, question, gold, predicted)

            all_results.append(result)

        mem.close()

        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if not args.no_judge and all_results:
        total = len(all_results)
        correct = sum(r.get("correct", False) for r in all_results)
        print(f"\nLoCoMo accuracy: {correct}/{total} = {correct / total:.1%}")

        from collections import defaultdict
        by_type = defaultdict(list)
        for r in all_results:
            by_type[r["question_type"]].append(r.get("correct", False))
        print(f"\n{'Category':<15} {'Acc':>8} {'N':>6}")
        print("-" * 35)
        for cat, vals in sorted(by_type.items()):
            acc = sum(vals) / len(vals) if vals else 0
            print(f"{cat:<15} {acc:>7.1%} {len(vals):>6}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
