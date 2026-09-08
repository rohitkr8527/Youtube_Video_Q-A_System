from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.llm.groq_client import LLMConfigurationError
from app.schemas.chat import ChatRequest, ChatResponse, ContentRequest, ContentResponse, QuizResponse
from app.services import get_rag_service


router = APIRouter(tags=["rag"])


def _handle(call):
    try:
        return call()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=503,
            detail="This app is not ready to answer questions yet.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while working with this video. Please try again.",
        ) from exc


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return _handle(lambda: get_rag_service().chat(request))


@router.post("/summary", response_model=ContentResponse)
def summary(request: ContentRequest) -> ContentResponse:
    return _handle(lambda: get_rag_service().summary(request.video_id))


@router.post("/notes", response_model=ContentResponse)
def notes(request: ContentRequest) -> ContentResponse:
    return _handle(lambda: get_rag_service().notes(request.video_id))


@router.post("/quiz", response_model=QuizResponse)
def quiz(request: ContentRequest) -> QuizResponse:
    return _handle(lambda: get_rag_service().quiz(request.video_id))
