"""
Chunking strategy: one document per OPINION, not per review.

Why: your dataset already has opinion-level granularity (theme/subtheme/
sentiment per opinion). Embedding whole reviews mixes multiple sentiments/
topics into one vector -> mushy retrieval, which is exactly the "improper
chunking" failure mode you saw at work. Opinion-level chunks retrieve
precisely; review_text is kept as metadata for context when citing.
"""
import pandas as pd


def build_documents(df: pd.DataFrame) -> list[dict]:
    docs = []
    for _, row in df.iterrows():
        text = (
            f"Product: {row['product_name']} | Brand: {row['brand']} | Category: {row['category']}\n"
            f"Customer opinion: {row['opinion']}\n"
            f"Sentiment: {row['opinion_sentiment']} (score {row['sentiment_score']}) | "
            f"Theme: {row['opinion_theme']} / {row['opinion_subtheme']}"
        )
        metadata = {
            "opinion_id": row["opinion_id"],
            "review_id": str(row["review_id"]),
            "product_name": str(row["product_name"]),
            "brand": str(row["brand"]),
            "category": str(row["category"]),
            "price": float(row["price"]) if pd.notna(row["price"]) else None,
            "opinion_sentiment": str(row["opinion_sentiment"]),
            "sentiment_score": float(row["sentiment_score"]) if pd.notna(row["sentiment_score"]) else None,
            "opinion_theme": str(row["opinion_theme"]),
            "opinion_subtheme": str(row["opinion_subtheme"]),
            "review_text": row["review_text"][:1000],  # capped, for citation context only
        }
        docs.append({"id": row["opinion_id"], "text": text, "metadata": metadata})
    return docs
