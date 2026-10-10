"""
Agentic RAG loop:
  question -> LLM picks tool(s) -> execute -> feed results back -> LLM
  decides: answer, or call another tool (multi-step) -> repeat up to
  MAX_AGENT_STEPS -> final answer, checked for grounding before return.

Grounding check: if the final answer isn't backed by anything retrieved
this turn, return a refusal instead of a confident-sounding guess. This
is the fix for the hallucination problem you flagged from the work system.

ask() also returns what was retrieved (retrieved_ids, contexts) so the
eval/ scripts can score search quality and answer quality.
"""
import json
from openai import OpenAI
import config
from corrective_retrieve import corrective_search
from tools import aggregate_tool, TOOL_SCHEMAS

SYSTEM_PROMPT = """You are a Voice-of-Customer analyst assistant. You answer
questions ONLY using the tool results you retrieve in this conversation —
never from prior knowledge about the brands/products. If tool results don't
support an answer, say so plainly instead of guessing.

Rules:
- Open-ended / thematic questions -> use semantic_search.
- Counting, averages, comparisons -> use aggregate, not semantic_search.
- Cite evidence: for semantic answers, reference opinion themes/sentiment
  you actually retrieved. For aggregate answers, state the row_count.
- semantic_search results include a "sufficient" flag. If it's false,
  the retrieval checkpoint could not confirm strong matches even after
  retrying — say your answer is based on weak/partial evidence, don't
  present it with full confidence.
- If row_count is 0, or semantic_search returns no docs at all, say the
  data doesn't support an answer — do not fill the gap yourself.
"""


def _slim_for_llm(name: str, result):
    """What the answering LLM actually sees.

    Raw semantic_search results are big (full metadata, review_text, scores,
    retry trace). Cutting that JSON at a fixed character count dropped most
    docs AND the trailing 'sufficient' flag, and left invalid JSON. So:
    keep only what the LLM needs, put 'sufficient' first, cap by DOC COUNT
    (best-ranked first) instead of by characters.
    """
    if name == "semantic_search" and isinstance(result, dict) and "docs" in result:
        docs = result["docs"][: config.MAX_DOCS_TO_LLM]
        return {
            "sufficient": result["sufficient"],
            "attempts": result["attempts"],
            "docs": [{"opinion_id": d["metadata"]["opinion_id"], "text": d["text"]} for d in docs],
        }
    return result


class VoCAgent:
    def __init__(self, store, bm25_index, df):
        self.store = store
        self.bm25_index = bm25_index
        self.df = df
        self.client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)

    def _execute_tool(self, name: str, args: dict):
        if name == "semantic_search":
            return corrective_search(self.store, self.bm25_index, **args)
        if name == "aggregate":
            return aggregate_tool(self.df, **args)
        return {"error": f"unknown tool {name}"}

    @staticmethod
    def _pack(answer, grounded, steps, evidence, contexts, retrieved_ids, retrieval_info):
        return {
            "answer": answer,
            "grounded": grounded,
            "steps": steps,
            "evidence": evidence,
            # eval hooks:
            "contexts": contexts,                                  # text the answering LLM actually saw
            "retrieved_ids": list(dict.fromkeys(retrieved_ids)),   # every opinion_id the checkpoint returned, ranked, de-duped
            "retrieval_info": retrieval_info,                      # per search: sufficient flag + attempts used
        }

    def ask(self, question: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        evidence_used = []  # tracks whether any tool actually returned data
        contexts, retrieved_ids, retrieval_info = [], [], []

        def pack(answer, grounded, steps):
            return self._pack(answer, grounded, steps, evidence_used, contexts, retrieved_ids, retrieval_info)

        for step in range(config.MAX_AGENT_STEPS):
            resp = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                # model thinks it's ready to answer
                if not evidence_used:
                    return pack(
                        "No data was retrieved to support an answer — rephrasing the question or checking filters (brand/category spelling) may help.",
                        False, step,
                    )
                return pack(msg.content, True, step)

            messages.append(msg.model_dump(exclude_unset=True))
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                result = self._execute_tool(name, args)
                llm_view = _slim_for_llm(name, result)

                has_data = (
                    (isinstance(result, dict) and "docs" in result and len(result["docs"]) >= config.MIN_DOCS_TO_ANSWER)
                    or (isinstance(result, dict) and "row_count" in result and result.get("row_count", 0) > 0)
                )
                if has_data:
                    evidence_used.append({"tool": name, "args": args})

                # record what was retrieved, for eval
                if name == "semantic_search" and isinstance(result, dict) and "docs" in result:
                    retrieved_ids.extend(d["metadata"]["opinion_id"] for d in result["docs"])
                    contexts.extend(d["text"] for d in llm_view["docs"])
                    retrieval_info.append({"sufficient": result["sufficient"], "attempts": result["attempts"]})
                elif name == "aggregate":
                    contexts.append(json.dumps(result, default=str))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(llm_view, default=str)[:12000],  # safety cap only; llm_view is already small
                })

        return pack("Reached max reasoning steps without a confident answer.", False, config.MAX_AGENT_STEPS)
