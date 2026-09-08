from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import requests


VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


@dataclass(slots=True)
class YouTubeMetadata:
    video_id: str
    title: str
    url: str
    thumbnail_url: str


def extract_video_id(url: str) -> str:
    raw = url.strip()
    if VIDEO_ID_RE.fullmatch(raw):
        return raw

    parsed = urlparse(raw)
    host = parsed.netloc.lower().replace("www.", "")

    candidate: str | None = None
    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.path.startswith("/shorts/") or parsed.path.startswith("/embed/") or parsed.path.startswith("/live/"):
            parts = [p for p in parsed.path.split("/") if p]
            candidate = parts[1] if len(parts) > 1 else None
    elif host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]

    if not candidate or not VIDEO_ID_RE.fullmatch(candidate):
        raise ValueError("Enter a valid YouTube video URL.")
    return candidate


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def fetch_metadata(url: str) -> YouTubeMetadata:
    video_id = extract_video_id(url)
    canonical = canonical_url(video_id)
    title = "YouTube video"
    thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": canonical, "format": "json"},
            timeout=10,
        )
        if response.ok:
            payload = response.json()
            title = payload.get("title") or title
            thumbnail = payload.get("thumbnail_url") or thumbnail
    except requests.RequestException:
        pass

    return YouTubeMetadata(
        video_id=video_id,
        title=title,
        url=canonical,
        thumbnail_url=thumbnail,
    )
