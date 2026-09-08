from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings
from app.schemas.video import VideoChunk, VideoInfo


class VideoRepository:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _dir(self, video_id: str) -> Path:
        path = self.settings.video_data_path / video_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, video: VideoInfo, chunks: list[VideoChunk]) -> None:
        directory = self._dir(video.video_id)
        (directory / "video.json").write_text(video.model_dump_json(indent=2), encoding="utf-8")
        (directory / "chunks.json").write_text(
            json.dumps([chunk.model_dump() for chunk in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_video(self, video_id: str) -> VideoInfo:
        path = self._dir(video_id) / "video.json"
        if not path.exists():
            raise FileNotFoundError(f"Video {video_id} has not been processed.")
        return VideoInfo.model_validate_json(path.read_text(encoding="utf-8"))

    def get_chunks(self, video_id: str) -> list[VideoChunk]:
        path = self._dir(video_id) / "chunks.json"
        if not path.exists():
            raise FileNotFoundError(f"Video {video_id} has not been processed.")
        data = json.loads(path.read_text(encoding="utf-8"))
        return [VideoChunk.model_validate(item) for item in data]
