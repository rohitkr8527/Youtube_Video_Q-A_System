from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import re

def extract_video_id(url: str) -> str | None:
    # Improved regex to accurately capture YouTube video IDs
    match = re.search(
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([0-9A-Za-z_-]{11})",
        url
    )
    return match.group(1) if match else None

def get_transcript(video_url: str) -> tuple[str, list[dict]]:
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL or unable to extract video ID.")

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        raise RuntimeError(f"Transcript not available for this video: {e}")

    full_text = " ".join([entry['text'] for entry in transcript])
    return full_text, transcript
