from app.retrieval.fusion import reciprocal_rank_fusion
from app.schemas.retrieval import RetrievedChunk


def chunk(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=i,
        video_id="abc",
        title="Test",
        text=f"chunk {i}",
        start_time=float(i),
        end_time=float(i + 1),
        topic="test",
    )


def test_rrf_rewards_items_present_in_multiple_rankings():
    dense = [chunk(1), chunk(2), chunk(3)]
    sparse = [chunk(2), chunk(4), chunk(1)]
    fused = reciprocal_rank_fusion([dense, sparse], limit=4)
    assert fused[0].chunk_id == 2
    assert {item.chunk_id for item in fused} == {1, 2, 3, 4}
