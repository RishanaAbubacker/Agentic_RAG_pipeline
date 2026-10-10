"""
Usage:
    python main.py path/to/reviews.csv          # builds index if needed, then chat loop
    python main.py path/to/reviews.csv --reindex # force rebuild index
"""
import sys
from data_loader import load_dataset
from chunking import build_documents
from vector_store import VectorStore
from bm25_index import BM25Index
from agent import VoCAgent


def build_agent(csv_path: str, reindex: bool = False) -> VoCAgent:
    """Load data, build/reuse the indexes, return a ready agent.
    Shared by this chat loop and by eval/run_eval.py so both test the same pipeline."""
    print("Loading dataset...")
    df = load_dataset(csv_path)
    print(f"Loaded {len(df)} opinions from {df['review_id'].nunique()} reviews.")

    docs = build_documents(df)  # needed for BM25 every run — it's in-memory only, not persisted like Chroma

    store = VectorStore()
    if reindex or store.is_empty():
        print("Building vector index (one-time, opinion-level chunks)...")
        store.index(docs)
        print(f"Indexed {len(docs)} opinion documents.")
    else:
        print("Using existing vector index. Run with --reindex to rebuild.")

    print("Building BM25 keyword index (in-memory, rebuilt every run)...")
    bm25_index = BM25Index(docs)

    return VoCAgent(store, bm25_index, df)


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/reviews.csv [--reindex]")
        sys.exit(1)

    agent = build_agent(sys.argv[1], reindex="--reindex" in sys.argv)

    print("\nReady. Ask questions (blank line to quit).\n")
    while True:
        q = input("> ").strip()
        if not q:
            break
        result = agent.ask(q)
        print(f"\n{result['answer']}\n")
        if not result.get("grounded", True):
            print("[unverified — no supporting data found]\n")


if __name__ == "__main__":
    main()
