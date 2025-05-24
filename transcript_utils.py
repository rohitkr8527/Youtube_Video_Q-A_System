from youtube_transcript_api import YouTubeTranscriptApi
import re

def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def get_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    full_text = " ".join([entry['text'] for entry in transcript])
    return full_text, transcript

