"""
Run memx against the ConvoMem benchmark (Salesforce/ConvoMem from HuggingFace).

Downloads one file per evidence type, samples items, and evaluates.

Usage:
    python bench/run_convomem.py --output results/memx_convomem.json --n 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib
import random

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

from huggingface_hub import list_repo_files, hf_hub_download
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

EVIDENCE_TYPES = [
    "abstention_evidence",
    "assistant_facts_evidence",
    "changing_evidence",
    "implicit_connection_evidence",
    "preference_evidence",
    "user_evidence",
]

JUDGE_PROMPT = """You are evaluating whether a predicted answer is correct.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

The predicted answer is CORRECT if it captures the key factual content of the gold answer.
For abstention questions, the answer is CORRECT if it correctly declines to answer.
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


def _load_samples(n: int) -> list[dict]:
    """Download one file per evidence type and sample n items total."""
    all_files = list_repo_files("Salesforce/ConvoMem", repo_type="dataset")
    by_type: dict[str, list[str]] = {}
    for f in all_files:
        if not f.endswith(".json"):
            continue
        for et in EVIDENCE_TYPES:
            if f"/evidence_questions/{et}/" in f:
                by_type.setdefault(et, []).append(f)
                break

    per_type = max(1, n // len(EVIDENCE_TYPES))
    samples = []

    for et in EVIDENCE_TYPES:
        files = by_type.get(et, [])
        if not files:
            continue
        chosen_file = files[0]
        print(f"  Downloading {et}: {chosen_file.split('/')[-1][:50]}...")
        path = hf_hub_download("Salesforce/ConvoMem", chosen_file, repo_type="dataset")
        with open(path) as f:
            data = json.load(f)
        items = data.get("evidence_items", [])
        for item in items[:per_type]:
            item["evidence_type"] = et
            samples.append(item)

    random.seed(42)
    random.shuffle(samples)
    return samples[:n]


def _chunk_messages(messages: list[dict], chunk_size: int = 20) -> list[list[dict]]:
    return [messages[i : i + chunk_size] for i in range(0, len(messages), chunk_size)]


def main():
    parser = argparse.ArgumentParser(description="Run memx on ConvoMem")
    parser.add_argument("--output", default="results/memx_convomem.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading ConvoMem samples (target: {args.n})...")
    samples = _load_samples(args.n)
    print(f"Loaded {len(samples)} samples across {len(EVIDENCE_TYPES)} evidence types.")

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

    for i, sample in enumerate(tqdm(samples, desc="ConvoMem")):
        question = sample["question"]
        gold = sample["answer"]
        evidence_type = sample.get("evidence_type", "unknown")

        all_messages = []
        for conv in sample.get("conversations", []):
            for msg in conv.get("messages", []):
                speaker = msg.get("speaker", "User")
                role = "user" if speaker.lower() in ("user", "human") else "assistant"
                all_messages.append({"role": role, "content": msg.get("text", "")})

        mem = MemorySystem(
            user_id=f"convomem_{i}",
            db_path="~/.memx/bench_convomem",
            llm=llm_fn,
            encoder_model=encoder,
            reranker_model=reranker,
        )

        chunks = _chunk_messages(all_messages, chunk_size=20)
        for si, chunk in enumerate(chunks):
            mem.ingest_session(chunk, session_id=si + 1, session_date="2024-01-01")

        try:
            predicted = mem.answer(question)
        except Exception as e:
            predicted = f"ERROR: {e}"

        result = {
            "question": question,
            "gold_answer": gold,
            "predicted": predicted,
            "question_type": evidence_type,
            "correct": judge(client, question, gold, predicted),
        }
        all_results.append(result)
        mem.close()

        if (i + 1) % 50 == 0:
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if all_results:
        total = len(all_results)
        correct = sum(r.get("correct", False) for r in all_results)
        print(f"\nConvoMem accuracy: {correct}/{total} = {correct / total:.1%}")

        from collections import defaultdict
        by_type = defaultdict(list)
        for r in all_results:
            by_type[r["question_type"]].append(r.get("correct", False))
        print(f"\n{'Category':<35} {'Acc':>8} {'N':>6}")
        print("-" * 55)
        for cat, vals in sorted(by_type.items()):
            acc = sum(vals) / len(vals) if vals else 0
            print(f"{cat:<35} {acc:>7.1%} {len(vals):>6}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
