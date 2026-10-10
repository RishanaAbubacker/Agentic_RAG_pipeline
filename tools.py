"""
Two tools, two different jobs — this is the "agentic" part:

- semantic_search: fuzzy/topical questions ("what do people complain about
  in the strap material") -> vector search, optionally metadata-filtered.
- aggregate: numeric/structured questions ("avg sentiment score for Brand X
  in Electronics", "count of negative opinions by theme") -> pandas, NOT
  vector search. Vector search can't count or average reliably; routing
  these to pandas is what kills a whole class of hallucination.
"""
import pandas as pd


def semantic_search_tool(store, query: str, brand=None, category=None,
                          opinion_sentiment=None, opinion_theme=None, k=8):
    where = {}
    for key, val in [("brand", brand), ("category", category),
                      ("opinion_sentiment", opinion_sentiment),
                      ("opinion_theme", opinion_theme)]:
        if val:
            where[key] = val
    results = store.search(query, k=k, where=where if where else None)
    return results


def aggregate_tool(df: pd.DataFrame, brand=None, category=None, opinion_theme=None,
                    opinion_sentiment=None, group_by=None, metric="sentiment_score",
                    agg="mean"):
    """group_by: one of brand/category/opinion_theme/opinion_subtheme/opinion_sentiment, or None."""
    filtered = df.copy()
    if brand:
        filtered = filtered[filtered["brand"].str.lower() == brand.lower()]
    if category:
        filtered = filtered[filtered["category"].str.lower() == category.lower()]
    if opinion_theme:
        filtered = filtered[filtered["opinion_theme"].str.lower() == opinion_theme.lower()]
    if opinion_sentiment:
        filtered = filtered[filtered["opinion_sentiment"].str.lower() == opinion_sentiment.lower()]

    if filtered.empty:
        return {"row_count": 0, "result": None}

    if group_by and group_by in filtered.columns:
        grouped = filtered.groupby(group_by)[metric].agg(agg)
        return {"row_count": len(filtered), "result": grouped.to_dict()}

    if agg == "count":
        result = len(filtered)
    else:
        result = filtered[metric].agg(agg)
    return {"row_count": len(filtered), "result": result}


# OpenAI function-calling schema for both tools
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Search customer opinions by meaning/topic. Use for open-ended 'what do customers say/like/hate about X' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "brand": {"type": "string"},
                    "category": {"type": "string"},
                    "opinion_sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "opinion_theme": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate",
            "description": "Compute counts/averages/sums over the review dataset. Use for any numeric or 'how many / on average' question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "brand": {"type": "string"},
                    "category": {"type": "string"},
                    "opinion_theme": {"type": "string"},
                    "opinion_sentiment": {"type": "string"},
                    "group_by": {"type": "string", "enum": ["brand", "category", "opinion_theme", "opinion_subtheme", "opinion_sentiment"]},
                    "metric": {"type": "string", "enum": ["sentiment_score", "price"], "default": "sentiment_score"},
                    "agg": {"type": "string", "enum": ["mean", "sum", "count", "min", "max"], "default": "mean"},
                },
                "required": [],
            },
        },
    },
]
