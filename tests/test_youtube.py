import pytest

from app.ingestion.youtube import canonical_url, extract_video_id


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=1",
        "https://youtube.com/shorts/dQw4w9WgXcQ",
        "https://youtube.com/embed/dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
    ],
)
def test_extract_video_id(url):
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_invalid_youtube_url():
    with pytest.raises(ValueError):
        extract_video_id("https://example.com/video")


def test_canonical_url():
    assert canonical_url("dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
