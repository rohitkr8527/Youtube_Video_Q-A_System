from app.schemas.chat import SourceCitation
from app.schemas.retrieval import RetrievedChunk
from evaluation.benchmark_models import TimeRange
from evaluation.metric_utils import (
    citation_hit_counts,
    hit_at_k,
    percentile,
    reciprocal_rank_by_time,
    ranges_overlap,
)


def chunk(chunk_id: int, start: float, end: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        video_id="v",
        title="t",
        text="x",
        start_time=start,
        end_time=end,
    )


def test_ranges_overlap():
    assert ranges_overlap(10, 20, 15, 25)
    assert ranges_overlap(10, 20, 20, 30)
    assert not ranges_overlap(10, 19, 20, 30)


def test_retrieval_metrics_by_timestamp():
    results = [chunk(1, 0, 10), chunk(2, 20, 30), chunk(3, 40, 50)]
    gt = [TimeRange(start=23, end=27)]
    assert hit_at_k(results, gt, 1) == 0
    assert hit_at_k(results, gt, 2) == 1
    assert reciprocal_rank_by_time(results, gt) == 0.5


def test_citation_hit_counts():
    citations = [
        SourceCitation(label="0:20", start_time=20, end_time=30, timestamp="0:20", url="https://x", excerpt="a"),
        SourceCitation(label="0:40", start_time=40, end_time=50, timestamp="0:40", url="https://x", excerpt="b"),
    ]
    correct, total = citation_hit_counts(citations, [TimeRange(start=25, end=28)])
    assert (correct, total) == (1, 2)


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([10], 95) == 10
