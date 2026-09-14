from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from groq import BadRequestError, RateLimitError

from app.ingestion.metadata import VideoRepository
from app.llm.groq_client import GroqLLM
from app.services import VideoService
from evaluation.benchmark_models import (
    BenchmarkCase,
    GeneratedAnswerableCase,
    GeneratedUnanswerableQuestions,
    TimeRange,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evaluation" / "video_manifest.json"
OUTPUT = ROOT / "evaluation" / "benchmark_dataset.jsonl"
QUESTIONS_PER_VIDEO = 10
UNANSWERABLE_PER_VIDEO = 2
MAX_RATE_LIMIT_RETRIES = 3
MAX_AUTOMATIC_WAIT_SECONDS = 20 * 60

T = TypeVar("T")


class GroundTruthRateLimitError(RuntimeError):
    pass

CATEGORY_SEQUENCE = [
    "factual",
    "conceptual",
    "exact_term",
    "explanation",
    "comparison",
    "factual",
    "conceptual",
    "multi_part",
    "explanation",
    "conceptual",
]

DIFFICULTY_BY_CATEGORY = {
    "factual": "easy",
    "exact_term": "easy",
    "conceptual": "medium",
    "explanation": "medium",
    "comparison": "hard",
    "multi_part": "hard",
}

SYSTEM = """You create rigorous evaluation questions for a YouTube transcript RAG benchmark.
You are given a small contiguous transcript window with exact timestamps.
Create ONE natural question whose answer is fully supported by this window.
The reference answer must use only information explicitly present in the window.
Do not mention chunk IDs, transcript windows, retrieval, or the evaluation process.
The requested category describes the STYLE of question you should create.
Create a question that follows that category as closely as possible.
Return only the fields required by the supplied JSON schema.
Do not add category, difficulty, or other metadata.
The relevant_start and relevant_end must stay inside the supplied window and should tightly cover the evidence needed for the answer.
"""

UNANSWERABLE_SYSTEM = """You create deliberately unanswerable questions for a video-RAG benchmark.
Given a video title and a broad transcript/topic summary, produce questions that are plausible user questions but whose answers are NOT stated in the video.
Avoid personal/private questions unless the video is biographical. Prefer adjacent technical concepts that are clearly outside the supplied content.
Do not create trick wording; the point is to test whether the assistant abstains instead of hallucinating.
"""


def _load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _ensure_video(
    item: dict,
    service: VideoService | None,
    repo: VideoRepository,
) -> VideoService | None:
    try:
        repo.get_video(item["video_id"])
        repo.get_chunks(item["video_id"])
    except FileNotFoundError:
        print(f"Processing: {item['title']}")
        service = service or VideoService()
        service.process(item["url"])
    return service


def _parse_duration_seconds(value: str) -> float | None:
    match = re.search(
        r"(?:(?P<hours>\d+(?:\.\d+)?)h)?"
        r"(?:(?P<minutes>\d+(?:\.\d+)?)m)?"
        r"(?P<seconds>\d+(?:\.\d+)?)s",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return (
        float(match.group("hours") or 0) * 3600
        + float(match.group("minutes") or 0) * 60
        + float(match.group("seconds") or 0)
    )


def _rate_limit_wait_seconds(error: RateLimitError) -> float:
    retry_after = error.response.headers.get("retry-after")
    try:
        parsed_header = float(retry_after) if retry_after is not None else 0
    except ValueError:
        parsed_header = 0
    if parsed_header > 0:
        return parsed_header

    # Groq includes values such as "Please try again in 15m43.92s" in the body.
    body = error.body if isinstance(error.body, dict) else {}
    detail = body.get("error", {}) if isinstance(body.get("error"), dict) else {}
    message = str(detail.get("message") or error)
    retry_text = re.search(r"try again in\s+([^.]*(?:\.\d+)?s)", message, re.IGNORECASE)
    parsed_message = _parse_duration_seconds(retry_text.group(1)) if retry_text else None
    return parsed_message or 60.0


def _with_rate_limit_retry(operation: Callable[[], T], description: str) -> T:
    for retry_number in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return operation()
        except RateLimitError as error:
            wait_seconds = _rate_limit_wait_seconds(error) + 1
            retries_left = MAX_RATE_LIMIT_RETRIES - retry_number
            if retries_left <= 0 or wait_seconds > MAX_AUTOMATIC_WAIT_SECONDS:
                raise GroundTruthRateLimitError(
                    "Groq's token quota is still exhausted. Completed cases have been saved; "
                    "run this command again after the quota resets."
                ) from error
            print(
                f"Groq rate limit reached while {description}. "
                f"Waiting {wait_seconds:.0f}s, then retrying ({retries_left} retries left)...",
                flush=True,
            )
            time.sleep(wait_seconds)
        except BadRequestError as error:
            if "json_validate_failed" in str(error) or "max completion tokens" in str(error):
                retries_left = MAX_RATE_LIMIT_RETRIES - retry_number
                if retries_left <= 0:
                    raise
                print(
                    f"Groq token limit reached during reasoning while {description}. Retrying ({retries_left} retries left)...",
                    flush=True,
                )
                time.sleep(2)
                continue
            raise

    raise AssertionError("unreachable")


def _load_existing_cases() -> dict[str, BenchmarkCase]:
    if not OUTPUT.exists():
        return {}
    cases: dict[str, BenchmarkCase] = {}
    for line_number, line in enumerate(OUTPUT.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = BenchmarkCase.model_validate_json(line)
        except ValueError as error:
            raise RuntimeError(f"Invalid JSONL at {OUTPUT}:{line_number}: {error}") from error
        cases[case.case_id] = case
    return cases


def _save_cases(cases: dict[str, BenchmarkCase]) -> None:
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for case_id in sorted(cases):
            handle.write(cases[case_id].model_dump_json() + "\n")
    temporary.replace(OUTPUT)


def _anchor_indices(count: int, needed: int) -> list[int]:
    if count <= 0:
        return []
    if count <= needed:
        return list(range(count)) + [count - 1] * (needed - count)
    # Keep anchors away from extreme edges so each can have context on both sides.
    start, end = 1, max(1, count - 2)
    if needed == 1:
        return [(start + end) // 2]
    return [round(start + (end - start) * i / (needed - 1)) for i in range(needed)]


def _window(chunks, index: int):
    lo = max(0, index - 1)
    hi = min(len(chunks), index + 2)
    return chunks[lo:hi]


def _format_window(window) -> str:
    return "\n\n".join(
        f"[{c.start_time:.1f}s - {c.end_time:.1f}s] {c.text}" for c in window
    )


def _generate_answerable(llm: GroqLLM, window, requested_category: str) -> GeneratedAnswerableCase:
    start = min(c.start_time for c in window)
    end = max(c.end_time for c in window)
    prompt = (
        f"Requested category: {requested_category}\n"
        f"Allowed timestamp range: {start:.1f} to {end:.1f} seconds\n\n"
        f"Transcript window:\n{_format_window(window)}"
    )
    generated = _with_rate_limit_retry(
        lambda: llm.structured(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            GeneratedAnswerableCase,
            name="benchmark_case",
            reasoning_effort="medium",
            max_tokens=2048,
        ),
        "generating an answerable benchmark case",
    )
    # Never trust a generated timestamp outside the supplied evidence window.
    generated.relevant_start = max(start, min(end, generated.relevant_start))
    generated.relevant_end = max(generated.relevant_start, min(end, generated.relevant_end))
    return generated


def _generate_unanswerable(llm: GroqLLM, title: str, chunks) -> list[str]:
    topic_map = "\n".join(
        f"- {c.start_time:.0f}s: {c.topic or c.text[:120]}" for c in chunks
    )
    # Topic map keeps this request small but still describes coverage of the whole video.
    result = _with_rate_limit_retry(
        lambda: llm.structured(
            [
                {"role": "system", "content": UNANSWERABLE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Video title: {title}\n\n"
                        f"Topics/sections covered across the video:\n{topic_map[:50000]}\n\n"
                        f"Return exactly {UNANSWERABLE_PER_VIDEO} questions."
                    ),
                },
            ],
            GeneratedUnanswerableQuestions,
            name="unanswerable_questions",
            reasoning_effort="medium",
            max_tokens=1500,
        ),
        "generating unanswerable benchmark cases",
    )
    return result.questions[:UNANSWERABLE_PER_VIDEO]


def main() -> None:
    repo = VideoRepository()
    llm = GroqLLM()
    manifest = _load_manifest()
    cases = _load_existing_cases()
    service: VideoService | None = None
    if cases:
        print(f"Loaded {len(cases)} existing benchmark cases from {OUTPUT}")

    try:
        for video_number, item in enumerate(manifest, start=1):
            service = _ensure_video(item, service, repo)
            video = repo.get_video(item["video_id"])
            chunks = repo.get_chunks(item["video_id"])
            if not chunks:
                raise RuntimeError(f"No transcript chunks found for {item['video_id']}")

            anchors = _anchor_indices(len(chunks), QUESTIONS_PER_VIDEO)
            for q_index, anchor in enumerate(anchors, start=1):
                case_id = f"v{video_number:02d}-q{q_index:02d}"
                if case_id in cases:
                    continue
                requested_category = CATEGORY_SEQUENCE[q_index - 1]

                generated = _generate_answerable(
                    llm,
                    _window(chunks, anchor),
                    requested_category,
                )

                cases[case_id] = BenchmarkCase(
                    case_id=f"v{video_number:02d}-q{q_index:02d}",
                    video_id=video.video_id,
                    video_url=video.url,
                    video_title=video.title,
                    question=generated.question,
                    category=requested_category,
                    difficulty=DIFFICULTY_BY_CATEGORY[requested_category],
                    answerable=True,
                    reference_answer=generated.reference_answer,
                    relevant_timestamps=[
                        TimeRange(
                            start=generated.relevant_start,
                            end=generated.relevant_end,
                        )
                    ],
                    reviewed=False,
                )
                _save_cases(cases)

            unanswerable_ids = [
                f"v{video_number:02d}-u{uq_index:02d}"
                for uq_index in range(1, UNANSWERABLE_PER_VIDEO + 1)
            ]
            if not all(case_id in cases for case_id in unanswerable_ids):
                questions = _generate_unanswerable(llm, video.title, chunks)
                for uq_index, question in enumerate(questions, start=1):
                    case_id = f"v{video_number:02d}-u{uq_index:02d}"
                    if case_id in cases:
                        continue
                    cases[case_id] = BenchmarkCase(
                        case_id=f"v{video_number:02d}-u{uq_index:02d}",
                        video_id=video.video_id,
                        video_url=video.url,
                        video_title=video.title,
                        question=question,
                        category="unanswerable",
                        difficulty="medium",
                        answerable=False,
                        reference_answer=(
                            "The video does not provide enough information to answer "
                            "this question."
                        ),
                        relevant_timestamps=[],
                        reviewed=False,
                    )
                _save_cases(cases)

            print(f"Prepared {QUESTIONS_PER_VIDEO + UNANSWERABLE_PER_VIDEO} cases for: {video.title}")
    finally:
        if service is not None:
            service.vector_store.close()

    _save_cases(cases)

    answerable = sum(case.answerable for case in cases.values())
    print(f"\nCreated {len(cases)} benchmark cases at {OUTPUT}")
    print(f"Answerable: {answerable} | Unanswerable: {len(cases) - answerable}")
    print("Review the JSONL once, then run: python -m evaluation.run_benchmark")


if __name__ == "__main__":
    try:
        main()
    except GroundTruthRateLimitError as error:
        raise SystemExit(str(error)) from None
