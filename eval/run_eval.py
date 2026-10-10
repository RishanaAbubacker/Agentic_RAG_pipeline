"""
Run the test set through the real pipeline and score it.

From the project root:
    python -m eval.run_eval path/to/reviews.csv --label baseline
    python -m eval.run_eval path/to/reviews.csv --label with-rerank --limit 10
    python -m eval.run_eval path/to/reviews.csv --no-judge      # search metrics only, no judge API calls

Outputs
    eval/results.csv     one row per question (answers, scores, judge reasons)
    eval/history.jsonl   one line per run: label, models, key settings, average scores
                         -> compare two lines to show before/after a change

Change one thing at a time (e.g. turn the reranker off, or move a threshold),
re-run with a new --label, compare the two history lines. That comparison is
the number that belongs on a resume.
"""
import argparse
import json
import os
from datetime import datetime

import pandas as pd

import config
from main import build_agent
from eval.metrics import build_judges, make_llm, retrieval_metrics, run_judges

SEARCH_COLS = ["hit", "recall_at_k", "precision_at_k", "mrr"]
JUDGE_COLS = ["groundedness", "helpfulness", "retrieval_relevance", "correctness"]


def load_testset(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(
            f"No test set at {path}. Create one first:\n"
            f"  python -m eval.generate_testset <reviews.csv> --n 30"
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean_of(results: pd.DataFrame, col: str):
    """Average ignoring failed/skipped rows (None). Returns (mean, n_scored)."""
    if col not in results.columns:
        return None, 0
    s = pd.to_numeric(results[col], errors="coerce").dropna()
    return (float(s.mean()) if len(s) else None), len(s)


def fmt(v):
    return "n/a" if v is None else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--testset", default=config.EVAL_TESTSET_PATH)
    ap.add_argument("--out", default=config.EVAL_RESULTS_PATH)
    ap.add_argument("--label", default="run", help="name for this run, e.g. baseline / with-rerank")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM judges (search metrics only)")
    ap.add_argument("--reindex", action="store_true")
    args = ap.parse_args()

    testset = load_testset(args.testset)
    if args.limit:
        testset = testset[: args.limit]

    agent = build_agent(args.csv_path, reindex=args.reindex)
    judges = None if args.no_judge else build_judges(make_llm(config.EVAL_JUDGE_MODEL))

    rows = []
    for i, t in enumerate(testset, start=1):
        print(f"[{i}/{len(testset)}] {t['question']}")
        row = {k: t.get(k) for k in ("id", "question", "reference_answer", "brand", "theme", "sentiment", "group_size")}
        try:
            res = agent.ask(t["question"])
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            print(f"    agent error: {row['error']}")
            continue

        info = res["retrieval_info"]
        row.update(
            answer=res["answer"],
            grounded=res["grounded"],
            steps=res["steps"],
            n_contexts=len(res["contexts"]),
            searched=bool(info),  # False = agent never used semantic_search for this question
            sufficient=all(x["sufficient"] for x in info) if info else None,
            attempts=max((x["attempts"] for x in info), default=None),
        )
        row.update(retrieval_metrics(res["retrieved_ids"], t["gold_opinion_ids"]))
        if judges:
            row.update(run_judges(judges, t["question"], res["answer"], res["contexts"], t["reference_answer"]))
        rows.append(row)

    results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results.to_csv(args.out, index=False)

    # ------------------------------------------------------------ summary
    n = len(results)
    n_err = int(results["error"].notna().sum()) if "error" in results.columns else 0
    metrics = {}
    for col in SEARCH_COLS + JUDGE_COLS:
        metrics[col], _ = mean_of(results, col)
    pipeline = {
        "used_search_rate": mean_of(results, "searched")[0],
        "sufficient_rate": mean_of(results, "sufficient")[0],
        "avg_attempts": mean_of(results, "attempts")[0],
        "grounded_rate": mean_of(results, "grounded")[0],
    }
    judge_errors = 0
    for col in JUDGE_COLS:
        reason = f"{col}_reason"
        if reason in results.columns:
            judge_errors += int(results[reason].astype(str).str.startswith("JUDGE ERROR").sum())

    print("\n" + "=" * 54)
    print(f"RUN '{args.label}'   questions: {n}   agent errors: {n_err}")
    print("=" * 54)
    print("SEARCH QUALITY")
    print(f"  hit rate         {fmt(metrics['hit'])}   (any right doc in top {config.EVAL_RECALL_K})")
    print(f"  recall@{config.EVAL_RECALL_K:<2}        {fmt(metrics['recall_at_k'])}")
    print(f"  precision@{config.EVAL_RECALL_K:<2}     {fmt(metrics['precision_at_k'])}   (lower bound, see README)")
    print(f"  MRR              {fmt(metrics['mrr'])}")
    if judges:
        print("ANSWER QUALITY (judge pass rate)")
        for col in JUDGE_COLS:
            m, k = mean_of(results, col)
            print(f"  {col:<20} {fmt(m)}   (scored {k}/{n})")
    print("PIPELINE")
    print(f"  used semantic_search     {fmt(pipeline['used_search_rate'])}   (low = agent routed to aggregate/refused)")
    print(f"  retrieval 'sufficient'   {fmt(pipeline['sufficient_rate'])}")
    print(f"  avg retry attempts       {fmt(pipeline['avg_attempts'])}")
    if judges:
        print(f"  judge errors             {judge_errors}   (these rows are left out of the averages)")

    if "recall_at_k" in results.columns:
        worst = results.dropna(subset=["recall_at_k"]).nsmallest(min(5, n), "recall_at_k")
        if len(worst):
            print("\nWEAKEST 5 (by recall) - read these first:")
            for _, r in worst.iterrows():
                print(f"  recall {r['recall_at_k']:.2f} | {r['id']} | {r['question']}")

    # ------------------------------------------------------------ history
    snapshot = {
        "label": args.label,
        "time": datetime.now().isoformat(timespec="seconds"),
        "n_questions": n,
        "agent_errors": n_err,
        "judge_errors": judge_errors,
        "models": {"llm": config.LLM_MODEL, "judge": None if args.no_judge else config.EVAL_JUDGE_MODEL,
                   "embedding": config.EMBEDDING_BACKEND, "reranker": config.RERANKER_MODEL},
        "settings": {k: getattr(config, k) for k in (
            "CANDIDATE_K", "RRF_K", "RERANK_TOP_N", "RERANK_GOOD_THRESHOLD", "RERANK_BAD_THRESHOLD",
            "MIN_RELEVANT_DOCS", "MAX_RETRIEVAL_RETRIES", "MAX_DOCS_TO_LLM", "EVAL_RECALL_K")},
        "metrics": metrics,
        "pipeline": pipeline,
    }
    hist_path = os.path.join(os.path.dirname(args.out) or ".", "history.jsonl")
    with open(hist_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot) + "\n")

    print(f"\nPer-question results: {args.out}")
    print(f"Run history (compare labels): {hist_path}")


if __name__ == "__main__":
    main()
