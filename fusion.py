"""
Reciprocal Rank Fusion: combines multiple ranked lists into one ranked
list, using RANK POSITION only — not raw scores. This matters because
BM25 scores and vector distances live on completely different, non-
comparable scales; fusing on raw scores would silently let whichever
list has bigger numbers dominate. Rank position is the only fair currency.

Formula per doc: sum of 1/(RRF_K + rank) across every list it appears in.
Appearing near the top of multiple lists beats appearing at #1 of just one.
"""
import config


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], id_key: str = "opinion_id") -> list[dict]:
    scores: dict = {}
    doc_lookup: dict = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc["metadata"][id_key]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (config.RRF_K + rank + 1)
            doc_lookup.setdefault(doc_id, doc)  # keep first-seen copy, content is identical either way

    fused_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
    fused = []
    for doc_id in fused_ids:
        d = dict(doc_lookup[doc_id])
        d["rrf_score"] = scores[doc_id]
        fused.append(d)
    return fused
