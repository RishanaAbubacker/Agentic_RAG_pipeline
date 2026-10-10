"""
Scoring helpers. Two kinds of checks, deliberately separate:

1. SEARCH quality  (plain Python, free, deterministic)
   Did the right opinion_ids come back?  -> hit, recall@k, precision@k, MRR

2. ANSWER quality  (LLM judges via LangChain + openevals, costs API calls)
   groundedness        : is the answer supported by the docs the LLM saw?
   helpfulness         : does the answer actually address the question?
   retrieval_relevance : are the retrieved docs relevant to the question?
   correctness         : does the answer match the reference answer?

A judge can fail (bad key, rate limit, odd output). Failures are stored as
None plus the error text, NOT as 0 — a crashed judge must never look like
a bad answer.
"""
import config
from langchain_openai import ChatOpenAI


def make_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """LangChain chat model. Same key / base_url settings as the pipeline,
    so Azure OpenAI works by setting OPENAI_BASE_URL."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )


# ---------------------------------------------------------------- search

def retrieval_metrics(retrieved_ids: list[str], gold_ids: list[str], k: int | None = None) -> dict:
    """
    gold_ids = every opinion that SHOULD count as a good answer source.

    recall_at_k    : found gold docs / min(len(gold), k). The min() matters:
                     if 40 docs are "correct" but we only look at k=15, perfect
                     recall is impossible, so the best score is capped fairly.
    precision_at_k : share of the top-k that are gold. This is a LOWER BOUND —
                     a retrieved doc outside the gold set may still be useful
                     (e.g. same theme, different subtheme), and counts as wrong here.
    mrr            : 1 / rank of the first gold doc (1.0 = first result was right).
    """
    k = k or config.EVAL_RECALL_K
    top = retrieved_ids[:k]
    gold = set(gold_ids)

    hits = [i for i in top if i in gold]
    found = len(set(hits))
    denom = min(len(gold), k)

    rr = 0.0
    for rank, doc_id in enumerate(top, start=1):
        if doc_id in gold:
            rr = 1.0 / rank
            break

    return {
        "hit": float(found > 0),
        "recall_at_k": found / denom if denom else 0.0,
        "precision_at_k": len(hits) / len(top) if top else 0.0,
        "mrr": rr,
        "n_retrieved": len(top),
    }


# ---------------------------------------------------------------- answers

def build_judges(judge_llm) -> dict:
    """LLM-as-judge evaluators from openevals (LangChain's eval library).
    Each returns pass/fail per call; averaged over questions = pass rate."""
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import (
        CORRECTNESS_PROMPT,
        RAG_GROUNDEDNESS_PROMPT,
        RAG_HELPFULNESS_PROMPT,
        RAG_RETRIEVAL_RELEVANCE_PROMPT,
    )

    def mk(prompt, key):
        return create_llm_as_judge(prompt=prompt, judge=judge_llm, feedback_key=key)

    return {
        "groundedness": mk(RAG_GROUNDEDNESS_PROMPT, "groundedness"),
        "helpfulness": mk(RAG_HELPFULNESS_PROMPT, "helpfulness"),
        "retrieval_relevance": mk(RAG_RETRIEVAL_RELEVANCE_PROMPT, "retrieval_relevance"),
        "correctness": mk(CORRECTNESS_PROMPT, "correctness"),
    }


def run_judges(judges: dict, question: str, answer: str, contexts: list[str], reference: str) -> dict:
    context_text = "\n\n".join(contexts)

    calls = {
        "groundedness": dict(context=context_text, outputs=answer),
        "helpfulness": dict(inputs=question, outputs=answer),
        "retrieval_relevance": dict(inputs=question, context=context_text),
        "correctness": dict(inputs=question, outputs=answer, reference_outputs=reference),
    }
    needs_context = {"groundedness", "retrieval_relevance"}

    out = {}
    for name, kwargs in calls.items():
        if name in needs_context and not contexts:
            # agent refused / never retrieved: nothing to be grounded in or relevant to
            out[name], out[f"{name}_reason"] = None, "skipped: no retrieved context"
            continue
        try:
            res = judges[name](**kwargs)
            out[name] = float(res["score"])  # True/False -> 1.0/0.0
            out[f"{name}_reason"] = res.get("comment", "")
        except Exception as e:  # noqa: BLE001 - record, never crash the whole run
            out[name] = None
            out[f"{name}_reason"] = f"JUDGE ERROR: {type(e).__name__}: {e}"
    return out
