# VoC Review RAG (agentic, local-first)

## Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # or Azure OpenAI key
python main.py sample_data/reviews_sample.csv
```
First run builds the Chroma index (`./chroma_db/`). Reuses it after that
unless you pass `--reindex`.

## Files
| File | Job |
|---|---|
| `data_loader.py` | load CSV, validate schema, generate `opinion_id` |
| `chunking.py` | build one retrieval doc per opinion (not per review) |
| `vector_store.py` | Chroma wrapper — swap for cloud vector DB later, nothing else changes |
| `tools.py` | `semantic_search` (fuzzy/topical) + `aggregate` (counts/averages via pandas) |
| `bm25_index.py`, `fusion.py`, `reranker.py` | keyword search, RRF fusion of 3 retrievers, local cross-encoder rerank |
| `retrieval_grader.py`, `corrective_retrieve.py` | grade retrieved docs, retry with a wider/looser search if weak |
| `agent.py` | tool-calling loop, grounding check before returning an answer |
| `main.py` | CLI chat loop |
| `eval/` | test-set generator + scorer (see Evaluation below). Full detail in `ARCHITECTURE.md` |

## Evaluation
```bash
pip install -r requirements.txt
python -m eval.generate_testset reviews.csv --n 30     # LLM writes questions from your reviews
# read ~10 of eval/testset.jsonl, fix or delete bad ones
python -m eval.run_eval reviews.csv --label baseline   # score the pipeline
python -m eval.run_eval reviews.csv --no-judge         # search metrics only, no judge API calls
```
Scores search (hit, recall@k, precision@k, MRR) and answers (groundedness,
helpfulness, correctness, retrieval relevance via LangChain + openevals judges).
Each run appends a line to `eval/history.jsonl`: change one thing, re-run with a
new `--label`, compare the two lines.

## Design decisions (so future-you remembers why)
- **Opinion-level chunks, not review-level.** Your dataset already has
  per-opinion sentiment/theme — embedding whole reviews blends topics and
  kills retrieval precision. This was your "improper chunking" problem.
- **Two tools, not one.** Semantic search can't count or average reliably.
  Numeric questions ("avg sentiment for Brand X") go to pandas via
  `aggregate`, not the vector store. This is most of what fixes hallucination
  on structured questions.
- **Grounding check.** Agent only answers if a tool call actually returned
  data this turn. Empty retrieval -> explicit "no data" response, not a
  confident guess.
- **MAX_AGENT_STEPS = 4.** Hard cap so the agent can't loop forever on a
  bad multi-step decomposition.

## Path to cloud (stage 2)
- Swap `EMBEDDING_BACKEND = "openai"` in `config.py` for `text-embedding-3-small`
  (or point at Azure OpenAI via `OPENAI_BASE_URL`).
- Swap `vector_store.py` internals for Azure AI Search / GCP Vertex Matching
  Engine — keep the same `search()`/`index()` method signatures so
  `agent.py`, `tools.py`, `main.py` don't need to change.
- For 10k-500k rows, BigQuery/Azure as source of truth for `aggregate_tool`
  (query warehouse directly instead of loading full df into pandas) once
  volume gets past comfortable in-memory size.

## Known gaps / next steps
- No conversation memory across turns (each `ask()` is stateless) — add if
  stakeholders will ask follow-ups like "what about just the negative ones."
- Eval covers topical (semantic_search) questions only. No tests yet for
  aggregate questions (compare against pandas) or "not in the data" refusals.
- Aggregate tool trusts exact-match brand/category strings — add fuzzy
  matching or a canonical value list if your brand names are inconsistent.
