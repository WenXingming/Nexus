"""Factory for the default AgentRuntime."""

from __future__ import annotations

from src.memory.in_memory_session_store import InMemorySessionStore
from src.model.config import ModelConfig
from src.model.fake_model import FakeModel
from src.runtime.agent_runtime import AgentRuntime


def create_runtime(model_config: ModelConfig | None = None) -> AgentRuntime:
    model_config = model_config or ModelConfig.from_env()
    if model_config.provider != "fake":
        raise ValueError(f"Unsupported model provider: {model_config.provider}")

    return AgentRuntime(
        model=FakeModel(),
        session_store=InMemorySessionStore(),
    )
