"""OpenAI-compatible model client."""

from __future__ import annotations

from collections.abc import Iterable

from src.core.contracts import Message
from src.model.config import ModelConfig


class OpenAIClient:
    """Model adapter for OpenAI-compatible chat clients."""

    def __init__(self, config: ModelConfig, client) -> None:
        self._config = config
        self._client = client

    def complete(self, messages: list[Message]) -> str:
        messages = self._to_openai_messages(messages)
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
        )
        return self._extract_content(response)

    def stream(self, messages: list[Message]) -> Iterable[str]:
        messages = self._to_openai_messages(messages)
        chunks = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            stream=True,
        )
        for chunk in chunks:
            content = self._extract_stream_content(chunk)
            if content is None:
                continue
            yield content

    def _to_openai_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

    def _extract_content(self, response) -> str:
        if not response.choices:
            raise RuntimeError("Model response choices is empty.")
        message = getattr(response.choices[0], "message", None)
        if message is None:
            raise RuntimeError("Model response choice message is missing.")
        return message.content or ""

    def _extract_stream_content(self, chunk) -> str | None:
        if not chunk.choices:
            return None
        delta = getattr(chunk.choices[0], "delta", None)
        if delta is None:
            return None
        content = delta.content
        if not content:
            return None
        return content
