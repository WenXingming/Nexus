"""OpenAI-compatible model client."""

from __future__ import annotations

from src.core.contracts import Message
from src.model.config import ModelConfig


class OpenAIClient:
    """Model adapter for OpenAI-compatible chat clients."""

    def __init__(self, config: ModelConfig, client) -> None:
        self._config = config
        self._client = client

    def complete(self, messages: list[Message]) -> str:
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        )
        content = response.choices[0].message.content
        return content or ""
