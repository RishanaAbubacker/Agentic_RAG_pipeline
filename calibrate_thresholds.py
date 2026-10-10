"""
Run this ONCE after you index your real dataset, before trusting the
retrieval-quality checkpoint. Chroma distances are not comparable across
embedding models/metrics, so DISTANCE_GOOD_THRESHOLD / DISTANCE_BAD_THRESHOLD
in config.py must be set from what your data actually produces, not guessed.

Usage:
    python calibrate_thresholds.py "your reviews csv path" "a query you know is well-covered" "a query you know isn't in the data"
"""
import sys
from data_loader import load_dataset
from chunking import build_documents
from vector_store import VectorStore


def main():
    if len(sys.argv) < 4:
        print('Usage: python calibrate_thresholds.py <csv_path> "<good query>" "<bad/unrelated query>"')
        sys.exit(1)

    csv_path, good_query, bad_query = sys.argv[1], sys.argv[2], sys.argv[3]

    df = load_dataset(csv_path)
    store = VectorStore()
    if store.is_empty():
        print("Index empty, building it first...")
        store.index(build_documents(df))

    for label, query in [("GOOD query (expect low distances)", good_query),
                          ("BAD query (expect high distances)", bad_query)]:
        print(f"\n--- {label}: \"{query}\" ---")
        results = store.search(query, k=10)
        for r in results:
            print(f"  distance={r['distance']:.3f}  |  {r['text'][:80]}")

    print(
        "\nRead the two lists. Set DISTANCE_GOOD_THRESHOLD in config.py just above "
        "where the GOOD list's distances cluster. Set DISTANCE_BAD_THRESHOLD just "
        "below where the BAD list's distances cluster. Gap between them = borderline "
        "zone the LLM judge handles."
    )


if __name__ == "__main__":
    main()
