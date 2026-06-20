"""Factory for the default AgentRuntime."""

from __future__ import annotations

from src.memory.in_memory_session_store import InMemorySessionStore
from src.model.fake_model import FakeModel
from src.runtime.agent_runtime import AgentRuntime


def create_runtime() -> AgentRuntime:
    return AgentRuntime(
        model=FakeModel(),
        session_store=InMemorySessionStore(),
    )
