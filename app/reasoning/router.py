from __future__ import annotations

from app.llm.groq_client import GroqLLM
from app.schemas.chat import ChatTurn
from app.schemas.reasoning import QueryPlan


ROUTER_SYSTEM = """You are the query planner for a YouTube video assistant.
Return a concise retrieval plan. Never answer the user's question.

Routes:
- FACTUAL_QA: direct facts from the video
- SUMMARY: asks to summarize broad content
- NOTES: asks for study notes
- QUIZ: asks to create a quiz
- TIMESTAMP_QUERY: explicitly asks about a time window or moment
- COMPARISON: compares two or more concepts from the video
- EXPLANATION: asks to explain a concept deeply/simply

Rewrite follow-up questions into standalone search queries using conversation history when needed.
Use decomposition only for questions that truly need multiple distinct searches.
If the user gives a time range, convert it to seconds. Otherwise start_time and end_time must be null.
If decomposition is not needed, subqueries must be an empty list.
"""


class QueryRouter:
    def __init__(self, llm: GroqLLM) -> None:
        self.llm = llm

    def plan(self, question: str, history: list[ChatTurn]) -> QueryPlan:
        recent = history[-6:]
        history_text = "\n".join(f"{turn.role}: {turn.content}" for turn in recent) or "No prior conversation."
        return self.llm.structured(
            [
                {"role": "system", "content": ROUTER_SYSTEM},
                {
                    "role": "user",
                    "content": f"Conversation:\n{history_text}\n\nCurrent question:\n{question}",
                },
            ],
            QueryPlan,
            name="query_plan",
            reasoning_effort="low",
            max_tokens=700,
        )
