from app.generation.citations import build_citations, format_timestamp, timestamp_url
from app.schemas.retrieval import RetrievedChunk


def test_format_timestamp():
    assert format_timestamp(62) == "1:02"
    assert format_timestamp(3662) == "1:01:02"


def test_citation_is_deterministic():
    item = RetrievedChunk(
        chunk_id=4,
        video_id="dQw4w9WgXcQ",
        title="Test",
        text="A useful explanation from the transcript.",
        start_time=62,
        end_time=87,
        topic="test",
    )
    citation = build_citations([item])[0]
    assert citation.label == "1:02–1:27"
    assert "t=62s" in citation.url
