from __future__ import annotations


def merge_query_text(primary: str, subqueries: list[str]) -> str:
    """Creates one reranking query while retrieval may run across several decomposed searches."""
    parts = [primary.strip(), *[item.strip() for item in subqueries if item.strip()]]
    return " | ".join(dict.fromkeys(part for part in parts if part))
