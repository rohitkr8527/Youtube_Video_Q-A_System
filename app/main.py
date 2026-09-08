from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_chat import router as rag_router
from app.api.routes_health import router as health_router
from app.api.routes_video import router as video_router
from app.config import get_settings


settings = get_settings()
app = FastAPI(
    title="VideoRAG API",
    version="1.0.0",
    description="Advanced YouTube RAG backend.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(video_router)
app.include_router(rag_router)
