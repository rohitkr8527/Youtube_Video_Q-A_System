from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings


router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "groq_configured": bool(settings.groq_api_key),
    }
