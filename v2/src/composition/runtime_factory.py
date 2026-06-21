"""Factory for the default AgentRuntime."""

from __future__ import annotations

from src.memory.config import MemoryConfig
from src.memory.file_session_store import FileSessionStore
from src.memory.in_memory_session_store import InMemorySessionStore
from src.model.config import ModelConfig
from src.model.fake_client import FakeClient
from src.model.openai_client import OpenAIClient
from src.model.openai_client_factory import create_openai_client
from src.runtime.agent_runtime import AgentRuntime
from src.runtime.config import AgentConfig


def create_runtime(
    model_config: ModelConfig | None = None,
    memory_config: MemoryConfig | None = None,
    agent_config: AgentConfig | None = None,
    openai_cls=None,
) -> AgentRuntime:
    model_config = model_config or ModelConfig.from_env()
    memory_config = memory_config or MemoryConfig.from_env()

    if model_config.provider == "fake":
        model = FakeClient()
    elif model_config.provider == "openai":
        if openai_cls is None:
            from openai import OpenAI
            openai_cls = OpenAI
        openai_client = create_openai_client(model_config, openai_cls)
        model = OpenAIClient(config=model_config, client=openai_client)
    else:
        raise ValueError(f"Unsupported model provider: {model_config.provider}")

    if memory_config.store == "memory":
        session_store = InMemorySessionStore()
    elif memory_config.store == "file":
        session_store = FileSessionStore(root=memory_config.root)
    else:
        raise ValueError(f"Unsupported memory store: {memory_config.store}")

    return AgentRuntime(
        model=model,
        session_store=session_store,
        config=agent_config,
    )
