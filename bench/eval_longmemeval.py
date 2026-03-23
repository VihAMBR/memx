"""
Evaluate memx predictions against LongMemEval gold answers using GPT-4o-mini as judge.

Usage:
    python bench/eval_longmemeval.py \\
        --predictions results/memx_longmemeval.json \\
        --output results/memx_scores.json

Outputs per-category accuracy and overall accuracy, plus a markdown table
for the README.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib
from collections import defaultdict

from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

try:
    from openai import OpenAI
except ImportError:
    print("openai package required: pip install openai")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Judge prompts (per question type, adapted from LongMemEval paper)
# ---------------------------------------------------------------------------

JUDGE_PROMPTS: dict[str, str] = {
    "temporal-reasoning": """You are evaluating whether a predicted answer correctly answers a temporal question.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

The predicted answer is CORRECT if:
- It gives the right date, duration, or time period
- Off-by-one errors in months are acceptable (e.g. "5 months" vs "6 months")
- The answer may include additional reasoning as long as the conclusion is correct

Reply with a single word: CORRECT or INCORRECT""",

    "knowledge-update": """You are evaluating whether a predicted answer reflects the most recent information.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

The predicted answer is CORRECT if:
- It gives the most up-to-date information matching the gold answer
- It's acceptable if it mentions outdated information as long as it clearly identifies the latest

Reply with a single word: CORRECT or INCORRECT""",

    "single-session-user": """You are evaluating whether a predicted answer is factually correct.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

The predicted answer is CORRECT if it matches the gold answer in all key facts.
Minor wording differences are fine.

Reply with a single word: CORRECT or INCORRECT""",

    "single-session-assistant": """You are evaluating whether a predicted answer is factually correct.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

The predicted answer is CORRECT if it matches the gold answer in all key facts.
Minor wording differences are fine.

Reply with a single word: CORRECT or INCORRECT""",

    "multi-session": """You are evaluating a multi-session memory question.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

The predicted answer is CORRECT if it correctly integrates information from
multiple sessions to reach the right conclusion.

Reply with a single word: CORRECT or INCORRECT""",

    "preference": """You are evaluating a user preference question.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

The predicted answer is CORRECT if it accurately reflects the user's preference
as stated in the gold answer. The predicted answer may contain additional
supporting evidence or reasoning.

Reply with a single word: CORRECT or INCORRECT""",

    "default": """You are evaluating whether a predicted answer is correct.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Reply with a single word: CORRECT or INCORRECT""",
}


def judge_answer(client, question: str, gold: str, predicted: str, qtype: str) -> bool:
    template = JUDGE_PROMPTS.get(qtype, JUDGE_PROMPTS["default"])
    prompt = template.format(question=question, gold=gold, predicted=predicted)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5,
    )
    verdict = (resp.choices[0].message.content or "").strip().upper()
    return verdict.startswith("CORRECT")


# ---------------------------------------------------------------------------
# Category name normalisation
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "single-session-user": "IE-User",
    "single-session-assistant": "IE-Asst",
    "multi-session": "Multi-session",
    "temporal-reasoning": "Temporal",
    "knowledge-update": "Knowledge-update",
    "preference": "Preference",
}


def normalise_type(raw: str) -> str:
    return _TYPE_MAP.get(raw.lower().replace("_", "-"), raw)


# ---------------------------------------------------------------------------
# Scoring + reporting
# ---------------------------------------------------------------------------

def compute_scores(results: list[dict]) -> dict:
    by_type: dict[str, list[bool]] = defaultdict(list)
    all_correct: list[bool] = []

    for r in results:
        correct = r.get("correct", False)
        cat = normalise_type(r.get("question_type", "unknown"))
        by_type[cat].append(correct)
        all_correct.append(correct)

    scores = {}
    for cat, vals in sorted(by_type.items()):
        acc = sum(vals) / len(vals) if vals else 0.0
        scores[cat] = {"accuracy": acc, "n": len(vals), "correct": sum(vals)}

    scores["Overall"] = {
        "accuracy": sum(all_correct) / len(all_correct) if all_correct else 0.0,
        "n": len(all_correct),
        "correct": sum(all_correct),
    }
    return scores


def print_table(scores: dict):
    print("\n" + "=" * 55)
    print(f"{'Category':<22} {'Acc':>8} {'Correct':>9} {'N':>6}")
    print("-" * 55)
    for cat, vals in scores.items():
        if cat == "Overall":
            continue
        acc = vals["accuracy"]
        print(f"{cat:<22} {acc:>7.1%} {vals['correct']:>9} {vals['n']:>6}")
    print("-" * 55)
    ov = scores["Overall"]
    print(f"{'Overall':<22} {ov['accuracy']:>7.1%} {ov['correct']:>9} {ov['n']:>6}")
    print("=" * 55)


def markdown_table(scores: dict) -> str:
    rows = ["| Category | Accuracy | N |", "|---|---|---|"]
    for cat, vals in scores.items():
        if cat == "Overall":
            continue
        rows.append(f"| {cat} | {vals['accuracy']:.1%} | {vals['n']} |")
    ov = scores["Overall"]
    rows.append(f"| **Overall** | **{ov['accuracy']:.1%}** | {ov['n']} |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate memx LongMemEval predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="results/memx_scores.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    with open(args.predictions) as f:
        results: list[dict] = json.load(f)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip already-judged items
    judged: list[dict] = []
    if args.resume and out_path.exists():
        with open(out_path) as f:
            judged = json.load(f)
        done = len(judged)
        print(f"Resuming: {done} already judged")
        results = results[done:]

    for r in tqdm(results, desc="Judging"):
        correct = judge_answer(
            client,
            question=r["question"],
            gold=r["gold_answer"],
            predicted=r["predicted"],
            qtype=r.get("question_type", "default"),
        )
        r["correct"] = correct
        judged.append(r)

        # Incremental save
        if len(judged) % 20 == 0:
            with open(out_path, "w") as f:
                json.dump(judged, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(judged, f, indent=2)

    scores = compute_scores(judged)
    print_table(scores)
    print("\nMarkdown table:\n")
    print(markdown_table(scores))

    scores_path = out_path.with_suffix(".scores.json")
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nScores saved to {scores_path}")


if __name__ == "__main__":
    main()
