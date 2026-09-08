from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class QueryMetrics:
    route: str = ""
    retrieval_retries: int = 0
    generation_retries: int = 0
    evidence_grade: str = ""
    grounded: bool | None = None
    retrieved_chunk_ids: list[int] = field(default_factory=list)
