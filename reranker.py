"""
Cross-encoder reranker: scores (query, doc) pairs directly, together, in
one pass — more accurate than comparing two separate embeddings, because
it can actually reason about how the specific pair relates. Slower per
pair, so only run on the top RERANK_TOP_N fused candidates, never the
whole corpus.

Model loads once (module-level cache) and gets reused across calls —
loading it fresh per query would be slow.
"""
from sentence_transformers import CrossEncoder
import config

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(config.RERANKER_MODEL)
    return _model


def rerank(query: str, docs: list[dict], top_n: int | None = None) -> list[dict]:
    top_n = top_n or config.RERANK_TOP_N
    candidates = docs[:top_n]
    if not candidates:
        return []

    pairs = [(query, d["text"]) for d in candidates]
    scores = _get_model().predict(pairs)

    for d, s in zip(candidates, scores):
        d["rerank_score"] = float(s)

    return sorted(candidates, key=lambda d: d["rerank_score"], reverse=True)
