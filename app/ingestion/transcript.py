from __future__ import annotations

from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi

from app.schemas.video import TranscriptSegment


def _normalize_entry(entry: Any) -> TranscriptSegment:
    if isinstance(entry, dict):
        return TranscriptSegment(
            text=str(entry.get("text", "")).strip(),
            start=float(entry.get("start", 0.0)),
            duration=float(entry.get("duration", 0.0)),
        )

    return TranscriptSegment(
        text=str(getattr(entry, "text", "")).strip(),
        start=float(getattr(entry, "start", 0.0)),
        duration=float(getattr(entry, "duration", 0.0)),
    )


def fetch_transcript(video_id: str) -> list[TranscriptSegment]:
    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, "fetch"):
            fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
            raw = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)
        else:
            raw = YouTubeTranscriptApi.get_transcript(video_id)  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError(
            "A usable transcript could not be loaded for this video. The video may have captions disabled or unavailable."
        ) from exc

    segments = [_normalize_entry(item) for item in raw]
    segments = [segment for segment in segments if segment.text]
    if not segments:
        raise RuntimeError("The video transcript is empty.")
    return segments
