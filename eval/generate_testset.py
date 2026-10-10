"""
Auto-generate a test set from your own reviews, using a LangChain LLM.

Run from the project root:
    python -m eval.generate_testset path/to/reviews.csv --n 30

How it works
  1. Group opinions by (brand, opinion_theme, opinion_sentiment).
  2. Pick N groups at random (seeded, so it is repeatable).
  3. For each group, show up to 8 opinions to the LLM and ask for ONE
     stakeholder-style question plus a short reference answer.
  4. Save question + reference answer + gold_opinion_ids (EVERY opinion in
     the group). Search is later scored on whether those ids come back.

Why a group and not a single opinion: reviews repeat themselves. If 40 people
say "battery dies fast", finding any of them is a correct retrieval. One
single "right id" would mark the other 39 as wrong.

IMPORTANT: auto-generated questions are a starting point, not truth.
Read ~10 of them before trusting any score. Delete or fix bad ones.
"""
import argparse
import json
import os

from pydantic import BaseModel, Field

import config
from data_loader import load_dataset
from eval.metrics import make_llm

SYSTEM = """You write evaluation questions for a Voice-of-Customer Q&A system.

You get customer opinions that share the same brand, theme and sentiment.
Write ONE question a business stakeholder might ask that these opinions answer,
and a short reference answer (1-3 sentences) using ONLY these opinions.

Rules:
- Mention the brand and the topic naturally.
- Paraphrase. Do NOT copy exact phrases from the opinions, so the test checks
  meaning, not keyword overlap.
- Open question ("what do customers say about...", "why are people unhappy with..."),
  never yes/no.
- Never use the words "opinions", "dataset" or "reviews" in the question.
"""


class QA(BaseModel):
    question: str = Field(description="The stakeholder question")
    reference_answer: str = Field(description="Short answer grounded only in the given opinions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--n", type=int, default=30, help="how many questions to generate")
    ap.add_argument("--min-group", type=int, default=3, help="skip groups with fewer opinions than this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=config.EVAL_TESTSET_PATH)
    args = ap.parse_args()

    df = load_dataset(args.csv_path)
    group_cols = ["brand", "opinion_theme", "opinion_sentiment"]
    groups = [(key, g) for key, g in df.groupby(group_cols) if len(g) >= args.min_group]

    if not groups:
        raise SystemExit(
            f"No (brand, theme, sentiment) group has >= {args.min_group} opinions. "
            "Lower --min-group (try 1 for tiny datasets)."
        )

    # seeded shuffle of the groups, then take N
    import random
    random.Random(args.seed).shuffle(groups)
    groups = groups[: args.n]
    print(f"Generating {len(groups)} questions (judge model: {config.EVAL_JUDGE_MODEL})...")

    llm = make_llm(config.EVAL_JUDGE_MODEL, temperature=0.2).with_structured_output(QA)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, ((brand, theme, sentiment), g) in enumerate(groups, start=1):
            sample = g["opinion"].head(8).tolist()
            human = (
                f"Brand: {brand}\nTheme: {theme}\nSentiment: {sentiment}\n\nOpinions:\n"
                + "\n".join(f"- {o}" for o in sample)
            )
            try:
                qa = llm.invoke([("system", SYSTEM), ("human", human)])
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}] skipped ({type(e).__name__}: {e})")
                continue

            row = {
                "id": f"q{i:03d}",
                "question": qa.question.strip(),
                "reference_answer": qa.reference_answer.strip(),
                "gold_opinion_ids": g["opinion_id"].tolist(),
                "brand": brand,
                "theme": theme,
                "sentiment": sentiment,
                "group_size": len(g),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            print(f"  [{i}] {row['question']}")

    print(f"\nWrote {written} questions to {args.out}")
    print("Next: read ~10 of them, fix or delete bad ones, then run  python -m eval.run_eval <csv>")


if __name__ == "__main__":
    main()
