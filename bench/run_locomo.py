"""
Run memx against the LoCoMo benchmark.

LoCoMo format (dataset/locomo10.json from the original paper):
    [
      {
        "conversation_id": "...",
        "qa_pairs": [
          {"question": "...", "answer": "...", "type": "..."}
        ],
        "sessions": [
          {
            "session_id": 1,
            "date": "2023-03-01",
            "conversation": [{"speaker": "A"|"B", "text": "..."}]
          }
        ]
      }
    ]

Usage:
    python bench/run_locomo.py \\
        --data dataset/locomo10.json \\
        --output results/memx_locomo.json \\
        --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib

from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from memx import MemorySystem

try:
    from openai import OpenAI
except ImportError:
    print("openai package required: pip install openai")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------

def locomo_to_sessions(conv: dict) -> list[dict]:
    """Convert LoCoMo conversation format to memx session format."""
    sessions = []
    for sess in conv.get("sessions", []):
        messages = []
        for turn in sess.get("conversation", []):
            speaker = turn.get("speaker", "user")
            role = "user" if speaker == "A" else "assistant"
            messages.append({"role": role, "content": turn.get("text", "")})
        sessions.append(
            {
                "session_id": sess.get("session_id", 0),
                "date": sess.get("date", "2024-01-01"),
                "messages": messages,
            }
        )
    return sessions


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are evaluating whether a predicted answer is correct.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

The predicted answer is CORRECT if it contains all the key information in the gold answer.
Minor wording differences and extra context are acceptable.

Reply with a single word: CORRECT or INCORRECT"""


def judge(client, question: str, gold: str, predicted: str) -> bool:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": JUDGE_PROMPT.format(
                question=question, gold=gold, predicted=predicted
            )}
        ],
        temperature=0.0,
        max_tokens=5,
    )
    verdict = (resp.choices[0].message.content or "").strip().upper()
    return verdict.startswith("CORRECT")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run memx on LoCoMo")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/memx_locomo.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judging")
    args = parser.parse_args()

    with open(args.data) as f:
        dataset = json.load(f)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    for conv in tqdm(dataset, desc="Conversations"):
        # Build memory system for this conversation
        mem = MemorySystem(llm_model=args.model, update_profile=True)

        sessions = locomo_to_sessions(conv)
        for sess in sessions:
            mem.ingest_session(sess["messages"], sess["session_id"], sess["date"])

        # Answer all QA pairs
        for qa in conv.get("qa_pairs", []):
            question = qa["question"]
            gold = qa.get("answer", "")
            qtype = qa.get("type", "general")

            try:
                predicted = mem.answer(question)
            except Exception as e:
                predicted = f"ERROR: {e}"

            result = {
                "conversation_id": conv.get("conversation_id", ""),
                "question": question,
                "gold_answer": gold,
                "predicted": predicted,
                "question_type": qtype,
            }

            if not args.no_judge:
                result["correct"] = judge(client, question, gold, predicted)

            all_results.append(result)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if not args.no_judge:
        total = len(all_results)
        correct = sum(r["correct"] for r in all_results)
        print(f"\nLoCoMo accuracy: {correct}/{total} = {correct/total:.1%}")

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
