import pytest

from src.model.config import ModelConfig
from src.model.openai_client_factory import create_openai_client


class FakeOpenAI:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_create_openai_client_requires_api_key() -> None:
    config = ModelConfig(provider="openai", api_key=None)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        create_openai_client(config, FakeOpenAI)


def test_create_openai_client_passes_api_key_and_base_url() -> None:
    config = ModelConfig(
        provider="openai",
        api_key="sk-test",
        base_url="https://example.test/v1",
    )

    client = create_openai_client(config, FakeOpenAI)

    assert client.kwargs == {
        "api_key": "sk-test",
        "base_url": "https://example.test/v1",
    }


def test_create_openai_client_passes_none_base_url() -> None:
    config = ModelConfig(provider="openai", api_key="sk-test", base_url=None)

    client = create_openai_client(config, FakeOpenAI)

    assert client.kwargs == {
        "api_key": "sk-test",
        "base_url": None,
    }
