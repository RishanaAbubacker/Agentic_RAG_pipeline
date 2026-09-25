"""
Grades retrieved docs for relevance BEFORE they reach the answering LLM.

Two-stage, cheap-first:
  1. Distance threshold buckets each doc: good / bad / borderline.
  2. Only borderline docs get an LLM judgment call (one batched call, not
     one per doc). Most queries never hit this stage.

This is the "does retrieval actually match the question" checkpoint —
separate from and upstream of the answering agent's own reasoning.
"""
import json
from openai import OpenAI
import config

_client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)


def _distance_bucket(distance: float) -> str:
    if distance <= config.DISTANCE_GOOD_THRESHOLD:
        return "good"
    if distance >= config.DISTANCE_BAD_THRESHOLD:
        return "bad"
    return "borderline"


def _rerank_bucket(score: float) -> str:
    # opposite direction from distance: HIGHER rerank_score = better match
    if score >= config.RERANK_GOOD_THRESHOLD:
        return "good"
    if score <= config.RERANK_BAD_THRESHOLD:
        return "bad"
    return "borderline"


def _llm_judge(query: str, borderline_docs: list[dict]) -> list[bool]:
    """One call judges all borderline docs at once. Returns bool per doc, same order."""
    if not borderline_docs:
        return []
    numbered = "\n".join(f"{i + 1}. {d['text']}" for i, d in enumerate(borderline_docs))
    prompt = (
        f"Question: {query}\n\nPassages:\n{numbered}\n\n"
        "For each passage, does it actually help answer the question? Be strict — "
        'tangentially related is not enough. Reply ONLY as JSON: {"1": true/false, "2": true/false, ...}'
    )
    resp = _client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    try:
        verdicts = json.loads(resp.choices[0].message.content)
        return [bool(verdicts.get(str(i + 1), False)) for i in range(len(borderline_docs))]
    except Exception:
        # judge output didn't parse -> fail safe, treat as not relevant
        return [False] * len(borderline_docs)


def grade(query: str, docs: list[dict]) -> dict:
    """Grades docs carrying a Chroma 'distance' field (plain vector search, no rerank)."""
    good, borderline, bad = [], [], []
    for d in docs:
        bucket = _distance_bucket(d["distance"])
        {"good": good, "bad": bad, "borderline": borderline}[bucket].append(d)

    if borderline:
        verdicts = _llm_judge(query, borderline)
        for d, is_relevant in zip(borderline, verdicts):
            (good if is_relevant else bad).append(d)

    return {
        "relevant": good,
        "relevant_count": len(good),
        "borderline_checked": len(borderline),
        "sufficient": len(good) >= config.MIN_RELEVANT_DOCS,
    }


def grade_reranked(query: str, docs: list[dict]) -> dict:
    """Grades docs carrying a 'rerank_score' field (post fusion + cross-encoder rerank)."""
    good, borderline, bad = [], [], []
    for d in docs:
        bucket = _rerank_bucket(d["rerank_score"])
        {"good": good, "bad": bad, "borderline": borderline}[bucket].append(d)

    if borderline:
        verdicts = _llm_judge(query, borderline)
        for d, is_relevant in zip(borderline, verdicts):
            (good if is_relevant else bad).append(d)

    return {
        "relevant": good,
        "relevant_count": len(good),
        "borderline_checked": len(borderline),
        "sufficient": len(good) >= config.MIN_RELEVANT_DOCS,
    }
