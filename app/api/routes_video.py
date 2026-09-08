from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.ingestion.metadata import VideoRepository
from app.schemas.video import ProcessVideoRequest, VideoInfo
from app.services import get_video_service


router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("/process", response_model=VideoInfo)
def process_video(request: ProcessVideoRequest) -> VideoInfo:
    try:
        return get_video_service().process(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="We couldn't prepare this video right now. Please try again or use another video.",
        ) from exc


@router.get("/{video_id}", response_model=VideoInfo)
def get_video(video_id: str) -> VideoInfo:
    try:
        return VideoRepository().get_video(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
