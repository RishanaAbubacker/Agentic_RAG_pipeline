"""
Agentic RAG loop:
  question -> LLM picks tool(s) -> execute -> feed results back -> LLM
  decides: answer, or call another tool (multi-step) -> repeat up to
  MAX_AGENT_STEPS -> final answer, checked for grounding before return.

Grounding check: if the final answer isn't backed by anything retrieved
this turn, return a refusal instead of a confident-sounding guess. This
is the fix for the hallucination problem you flagged from the work system.
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

    def ask(self, question: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        evidence_used = []  # tracks whether any tool actually returned data

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
                    return {
                        "answer": "No data was retrieved to support an answer — rephrasing the question or checking filters (brand/category spelling) may help.",
                        "grounded": False,
                        "steps": step,
                    }
                return {"answer": msg.content, "grounded": True, "steps": step, "evidence": evidence_used}

            messages.append(msg.model_dump(exclude_unset=True))
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments or "{}")
                result = self._execute_tool(tool_call.function.name, args)

                has_data = (
                    (isinstance(result, dict) and "docs" in result and len(result["docs"]) >= config.MIN_DOCS_TO_ANSWER)
                    or (isinstance(result, dict) and "row_count" in result and result.get("row_count", 0) > 0)
                )
                if has_data:
                    evidence_used.append({"tool": tool_call.function.name, "args": args})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str)[:6000],  # cap payload size
                })

        return {"answer": "Reached max reasoning steps without a confident answer.", "grounded": False, "steps": config.MAX_AGENT_STEPS}
