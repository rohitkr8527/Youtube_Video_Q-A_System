from __future__ import annotations

import httpx
import pytest
from groq import RateLimitError

from evaluation import prepare_ground_truth


def _rate_limit_error(message: str, retry_after: str | None = None) -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, headers=headers, request=request)
    return RateLimitError(
        "rate limited",
        response=response,
        body={"error": {"message": message}},
    )


def test_rate_limit_wait_uses_retry_after_header() -> None:
    error = _rate_limit_error("Please try again in 15m43.92s", retry_after="12.5")

    assert prepare_ground_truth._rate_limit_wait_seconds(error) == 12.5


def test_rate_limit_wait_parses_groq_error_message() -> None:
    error = _rate_limit_error("Please try again in 15m43.92s.")

    assert prepare_ground_truth._rate_limit_wait_seconds(error) == pytest.approx(943.92)


def test_rate_limited_operation_retries_after_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _rate_limit_error("Please try again in 2.5s.")
        return "done"

    monkeypatch.setattr(prepare_ground_truth.time, "sleep", sleeps.append)

    assert prepare_ground_truth._with_rate_limit_retry(operation, "testing") == "done"
    assert calls == 2
    assert sleeps == [3.5]
