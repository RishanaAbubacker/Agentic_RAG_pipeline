"""
Full retrieval checkpoint, in order:

  1. Run 3 retrievers: BM25 keyword search, vector search on the original
     query, vector search on an LLM-rewritten query.
  2. Fuse all 3 ranked lists with RRF (rank-based, not score-based).
  3. Rerank the top fused candidates with a local cross-encoder.
  4. Grade the reranked results (grade_reranked, keyed on rerank_score
     instead of the old distance-based grade()).
  5. If not enough "good" docs: retry with an escalating strategy, capped:
       attempt 1: widen candidate_k (more inputs into the fusion)
       attempt 2: drop metadata filters (cast a wider net)
       attempt 3: rewrite the query again, differently, last resort

Query rewriting happens ONCE up front (needed for the initial fusion, not
saved for a later step like the single-retriever version was) plus again
only if attempt 3 is actually reached.
"""
from openai import OpenAI
import config
from fusion import reciprocal_rank_fusion
from reranker import rerank
from retrieval_grader import grade_reranked

_client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)

_REWRITE_PROMPTS = [
    "Rewrite this search query to be broader and more likely to match relevant "
    "customer review passages. Return ONLY the rewritten query.",
    "Rewrite this search query using different wording and synonyms than the "
    "original, staying on the same topic. Return ONLY the rewritten query.",
]


def _rewrite_query(query: str, variant: int = 0) -> str:
    instruction = _REWRITE_PROMPTS[min(variant, len(_REWRITE_PROMPTS) - 1)]
    resp = _client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": f"{instruction}\n\nQuery: {query}"}],
    )
    return resp.choices[0].message.content.strip()


def _active(filters: dict):
    active = {k: v for k, v in filters.items() if v}
    return active or None


def _fuse_and_rerank(store, bm25_index, query, rewritten_query, filters, candidate_k):
    where = _active(filters)
    vector_hits = store.search(query, k=candidate_k, where=where)
    vector_hits_rw = store.search(rewritten_query, k=candidate_k, where=where)
    bm25_hits = bm25_index.search(query, k=candidate_k, filters=where)

    fused = reciprocal_rank_fusion([vector_hits, vector_hits_rw, bm25_hits])
    return rerank(query, fused)


def corrective_search(store, bm25_index, query: str, brand=None, category=None,
                       opinion_sentiment=None, opinion_theme=None, k=None) -> dict:
    candidate_k = k or config.CANDIDATE_K
    filters = {"brand": brand, "category": category,
               "opinion_sentiment": opinion_sentiment, "opinion_theme": opinion_theme}
    rewritten_query = _rewrite_query(query, variant=0)
    trace = []

    for attempt in range(config.MAX_RETRIEVAL_RETRIES + 1):
        reranked_docs = _fuse_and_rerank(store, bm25_index, query, rewritten_query, filters, candidate_k)
        result = grade_reranked(query, reranked_docs)
        trace.append({
            "attempt": attempt,
            "candidate_k": candidate_k,
            "active_filters": _active(filters),
            "relevant_count": result["relevant_count"],
        })

        if result["sufficient"] or attempt == config.MAX_RETRIEVAL_RETRIES:
            return {
                "docs": result["relevant"] or reranked_docs,  # fall back to best-available over nothing
                "sufficient": result["sufficient"],
                "attempts": attempt + 1,
                "trace": trace,
            }

        if attempt == 0:
            candidate_k = min(candidate_k * config.RETRY_K_MULTIPLIER, config.MAX_K_CAP)
        elif attempt == 1:
            filters = {kk: None for kk in filters}
        elif attempt == 2:
            rewritten_query = _rewrite_query(query, variant=1)

    return {"docs": [], "sufficient": False, "attempts": config.MAX_RETRIEVAL_RETRIES + 1, "trace": trace}
