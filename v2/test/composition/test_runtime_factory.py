import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.composition.runtime_factory import create_runtime
from src.core.contracts import AgentRequest
from src.memory.config import MemoryConfig
from src.model.config import ModelConfig


class FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse("Echo from openai")


class FakeChat:
    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeOpenAI:
    instances: list["FakeOpenAI"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.completions = FakeCompletions()
        self.chat = FakeChat(self.completions)
        self.instances.append(self)


def test_create_runtime_returns_working_runtime() -> None:
    runtime = create_runtime()

    result = runtime.run(AgentRequest(input="hi"))

    assert result.output == "Echo: hi"
    assert result.session_id == "s1"


def test_create_runtime_accepts_fake_client_config() -> None:
    runtime = create_runtime(ModelConfig(provider="fake"))

    result = runtime.run(AgentRequest(input="hi"))

    assert result.output == "Echo: hi"


def test_create_runtime_accepts_memory_config() -> None:
    runtime = create_runtime(memory_config=MemoryConfig(store="memory"))

    result = runtime.run(AgentRequest(input="hi"))

    assert result.output == "Echo: hi"


def test_create_runtime_accepts_file_memory_config() -> None:
    root = Path("v2/test/.tmp/runtime_factory") / uuid4().hex
    runtime = create_runtime(memory_config=MemoryConfig(store="file", root=root))

    first = runtime.run(AgentRequest(input="first"))
    runtime.run(AgentRequest(input="second", session_id=first.session_id))

    raw = (root / "s1.json").read_text(encoding="utf-8")
    assert json.loads(raw) == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "Echo: first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "Echo: second"},
    ]


def test_create_runtime_rejects_unknown_memory_store() -> None:
    with pytest.raises(ValueError, match="Unsupported memory store: unknown"):
        create_runtime(memory_config=MemoryConfig(store="unknown"))


def test_create_runtime_rejects_unknown_model_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported model provider: unknown"):
        create_runtime(ModelConfig(provider="unknown"))


def test_create_runtime_accepts_openai_model_config() -> None:
    FakeOpenAI.instances = []
    runtime = create_runtime(
        ModelConfig(
            provider="openai",
            api_key="sk-test",
            base_url="https://example.test/v1",
            model="test-model",
        ),
        openai_cls=FakeOpenAI,
    )

    result = runtime.run(AgentRequest(input="hi"))

    assert result.output == "Echo from openai"
    assert FakeOpenAI.instances[0].kwargs == {
        "api_key": "sk-test",
        "base_url": "https://example.test/v1",
    }
    assert FakeOpenAI.instances[0].completions.calls[0]["model"] == "test-model"
