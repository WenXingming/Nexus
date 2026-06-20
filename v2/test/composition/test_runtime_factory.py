import pytest

from src.composition.runtime_factory import create_runtime
from src.core.contracts import AgentRequest
from src.model.config import ModelConfig


def test_create_runtime_returns_working_runtime() -> None:
    runtime = create_runtime()

    result = runtime.run(AgentRequest(input="hi"))

    assert result.output == "Echo: hi"
    assert result.session_id == "s1"


def test_create_runtime_accepts_fake_model_config() -> None:
    runtime = create_runtime(ModelConfig(provider="fake"))

    result = runtime.run(AgentRequest(input="hi"))

    assert result.output == "Echo: hi"


def test_create_runtime_rejects_unknown_model_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported model provider: unknown"):
        create_runtime(ModelConfig(provider="unknown"))
