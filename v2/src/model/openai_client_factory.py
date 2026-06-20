"""Factory for OpenAI-compatible SDK clients."""

from __future__ import annotations

from src.model.config import ModelConfig


def create_openai_client(config: ModelConfig, openai_cls):
    if not config.api_key:
        raise ValueError("OPENAI_API_KEY is required for openai provider")

    return openai_cls(
        api_key=config.api_key,
        base_url=config.base_url,
    )
