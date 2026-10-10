"""Load + validate the reviews+metadata dataset. CSV in, clean df out."""
import pandas as pd
from config import REQUIRED_COLUMNS


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Rename your columns or edit REQUIRED_COLUMNS in config.py."
        )

    # ids: one row = one opinion. review_id groups opinions from same review.
    if "review_id" not in df.columns:
        df["review_id"] = df.index.astype(str)
    df["opinion_id"] = df["review_id"].astype(str) + "_" + df.groupby("review_id").cumcount().astype(str)

    # basic hygiene — bad rows silently poison embeddings/aggregations later
    df["opinion"] = df["opinion"].fillna("").astype(str)
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df = df[df["opinion"].str.strip() != ""].reset_index(drop=True)
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df
