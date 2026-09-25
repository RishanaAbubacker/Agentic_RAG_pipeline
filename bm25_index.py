"""
Keyword search side of hybrid retrieval. Catches exact-term matches
(model numbers, brand names, specific words) that embeddings sometimes
blur past. Rebuilt in memory at startup — pure counting/tokenizing, no
model involved, cheap even at hundreds of thousands of rows.

No native metadata filtering in BM25Okapi, so filtering happens AFTER
scoring: rank the whole corpus, then walk down the ranked list skipping
non-matching docs until k are collected. Correct, just means a very
narrow filter can return fewer than k hits — expected, not a bug.
"""
from rank_bm25 import BM25Okapi


def _matches(metadata: dict, filters: dict) -> bool:
    return all(metadata.get(key) == val for key, val in filters.items() if val)


class BM25Index:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        tokenized = [d["text"].lower().split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int, filters: dict | None = None) -> list[dict]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        out = []
        for i in ranked_idx:
            doc = self.docs[i]
            if filters and not _matches(doc["metadata"], filters):
                continue
            out.append({"text": doc["text"], "metadata": doc["metadata"], "bm25_score": float(scores[i])})
            if len(out) >= k:
                break
        return out
