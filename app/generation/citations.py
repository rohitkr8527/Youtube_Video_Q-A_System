from __future__ import annotations

from urllib.parse import urlencode

from app.schemas.chat import SourceCitation
from app.schemas.retrieval import RetrievedChunk


def format_timestamp(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def timestamp_url(video_id: str, seconds: float) -> str:
    return f"https://www.youtube.com/watch?{urlencode({'v': video_id, 't': f'{max(0, int(seconds))}s'})}"


def build_citations(chunks: list[RetrievedChunk], limit: int = 4) -> list[SourceCitation]:
    citations: list[SourceCitation] = []
    seen: set[int] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        start = format_timestamp(chunk.start_time)
        end = format_timestamp(chunk.end_time)
        excerpt = " ".join(chunk.text.split())
        if len(excerpt) > 220:
            excerpt = excerpt[:217].rstrip() + "..."
        citations.append(
            SourceCitation(
                label=f"{start}–{end}",
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                timestamp=start,
                url=timestamp_url(chunk.video_id, chunk.start_time),
                excerpt=excerpt,
            )
        )
        if len(citations) >= limit:
            break
    return citations
