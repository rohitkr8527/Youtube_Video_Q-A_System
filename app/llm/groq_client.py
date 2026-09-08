from __future__ import annotations

import json
from typing import TypeVar

from groq import BadRequestError, Groq
from pydantic import BaseModel, ValidationError

from app.config import get_settings


T = TypeVar("T", bound=BaseModel)


class LLMConfigurationError(RuntimeError):
    pass


class GroqLLM:
    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.groq_api_key:
            raise LLMConfigurationError(
                "GROQ_API_KEY is missing. Copy .env.example to .env and add your Groq API key."
            )
        self.client = Groq(
            api_key=self.settings.groq_api_key,
            timeout=self.settings.groq_timeout_seconds,
            max_retries=2,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1800,
        reasoning_effort: str = "medium",
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.groq_model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_format="hidden",
        )
        return (response.choices[0].message.content or "").strip()

    def structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        name: str,
        reasoning_effort: str = "low",
        max_tokens: int = 1200,
    ) -> T:
        json_schema = schema.model_json_schema()
        try:
            response = self.client.chat.completions.create(
                model=self.settings.groq_model,
                messages=messages,
                temperature=0,
                max_completion_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                reasoning_format="hidden",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "strict": True,
                        "schema": json_schema,
                    },
                },
            )
            data = json.loads(response.choices[0].message.content or "{}")
            return schema.model_validate(data)
        except (BadRequestError, json.JSONDecodeError, ValidationError):
            # Fallback keeps the product usable if a future schema detail changes.
            response = self.client.chat.completions.create(
                model=self.settings.groq_model,
                messages=messages
                + [
                    {
                        "role": "system",
                        "content": "Return valid JSON only. Do not include markdown fences.",
                    }
                ],
                temperature=0,
                max_completion_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                reasoning_format="hidden",
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content or "{}")
            return schema.model_validate(data)
