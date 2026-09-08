from __future__ import annotations


def choose_search_queries(standalone_query: str, subqueries: list[str]) -> list[str]:
    """Use the planner output without spending another LLM call when decomposition is unnecessary."""
    clean = [query.strip() for query in subqueries if query.strip()]
    if clean:
        return clean[:4]
    return [standalone_query.strip()]
