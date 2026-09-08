from __future__ import annotations

import os

import requests


API_BASE = os.getenv("VIDEORAG_API_URL", f"http://{os.getenv('API_HOST', '127.0.0.1')}:{os.getenv('API_PORT', '8000')}")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))


class APIError(RuntimeError):
    pass


def _post(path: str, payload: dict):
    try:
        response = requests.post(f"{API_BASE}{path}", json=payload, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise APIError("The app is temporarily unavailable. Please try again in a moment.") from exc
    if not response.ok:
        try:
            message = response.json().get("detail", response.text)
        except ValueError:
            message = response.text
        raise APIError(str(message))
    return response.json()


def process_video(url: str) -> dict:
    return _post("/videos/process", {"url": url})


def ask(video_id: str, question: str, history: list[dict]) -> dict:
    return _post("/chat", {"video_id": video_id, "question": question, "history": history})


def summary(video_id: str) -> dict:
    return _post("/summary", {"video_id": video_id})


def notes(video_id: str) -> dict:
    return _post("/notes", {"video_id": video_id})


def quiz(video_id: str) -> dict:
    return _post("/quiz", {"video_id": video_id})
