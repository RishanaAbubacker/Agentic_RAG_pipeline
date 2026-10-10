import os

# --- Embeddings ---
# Local/free by default. Swap to "text-embedding-3-small" via OpenAI/Azure
# when you move to cloud stage 2 (change EMBEDDING_BACKEND to "openai").
EMBEDDING_BACKEND = os.getenv("RAG_EMBEDDING_BACKEND", "local")  # "local" | "openai"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# --- LLM (agent brain) ---
# Works with OpenAI SDK. Point base_url/key at Azure OpenAI later without
# changing any pipeline code.
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # set for Azure OpenAI endpoint

# --- Vector store ---
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "voc_reviews"

# --- Retrieval / Agent ---
TOP_K = 8
MAX_AGENT_STEPS = 4          # hard cap on tool-call loops, avoid runaway agent
MIN_DOCS_TO_ANSWER = 1        # below this -> refuse instead of hallucinate
MAX_DOCS_TO_LLM = 10          # top-ranked docs shown to the answering LLM (cap by count, not characters)

# --- Retrieval quality checkpoint ---
# Chroma distance is embedding/metric dependent (default = L2, not cosine).
# THESE NUMBERS ARE NOT UNIVERSAL — calibrate against your own data: run a
# few known-relevant and known-irrelevant queries, print distances, set
# thresholds from what you actually see. Values below are a starting guess
# for all-MiniLM-L6-v2 + Chroma default L2.
DISTANCE_GOOD_THRESHOLD = 0.9    # <= this: confidently relevant, skip LLM check
DISTANCE_BAD_THRESHOLD = 1.5     # >= this: confidently irrelevant, skip LLM check
                                   # between the two: borderline, LLM judges it
MIN_RELEVANT_DOCS = 3             # need >= this many "good" docs to call retrieval sufficient
MAX_RETRIEVAL_RETRIES = 3         # escalation steps: widen k -> drop filters -> rewrite query again
RETRY_K_MULTIPLIER = 2
MAX_K_CAP = 30

# --- Hybrid retrieval: RRF fusion (BM25 + vector[original] + vector[rewritten]) ---
RRF_K = 60            # standard RRF damping constant, rarely needs changing
CANDIDATE_K = 20      # how many hits EACH retriever contributes before fusion

# --- Reranker (local cross-encoder, scores query+doc pairs directly) ---
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 15         # only rerank top N fused candidates — reranking is slow, cap the cost
# Cross-encoder outputs raw logits, NOT 0-1 bounded. These are guesses —
# same calibration warning as the distance thresholds above.
RERANK_GOOD_THRESHOLD = 0.0
RERANK_BAD_THRESHOLD = -3.0

# --- Evaluation (see eval/) ---
# Judge model grades answers. Using a STRONGER model than LLM_MODEL is better:
# a model grading its own answers tends to be too kind.
EVAL_JUDGE_MODEL = os.getenv("RAG_EVAL_JUDGE_MODEL", LLM_MODEL)
EVAL_TESTSET_PATH = "eval/testset.jsonl"
EVAL_RESULTS_PATH = "eval/results.csv"
EVAL_RECALL_K = 15            # recall/precision measured on the top K retrieved docs

# --- Required columns in your dataset ---
# review_id is auto-generated if missing. opinion_id is auto-generated per row.
REQUIRED_COLUMNS = [
    "product_name", "brand", "price", "category",
    "review_text", "opinion", "opinion_sentiment",
    "sentiment_score", "opinion_theme", "opinion_subtheme",
]
