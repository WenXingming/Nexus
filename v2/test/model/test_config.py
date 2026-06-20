from src.model.config import ModelConfig


def test_model_config_defaults_to_fake(monkeypatch) -> None:
    monkeypatch.delenv("NEXUS_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    config = ModelConfig.from_env()

    assert config.provider == "fake"
    assert config.api_key is None
    assert config.base_url is None
    assert config.model == "gpt-4o-mini"


def test_model_config_reads_values_from_env(monkeypatch) -> None:
    monkeypatch.setenv("NEXUS_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "custom-model")

    config = ModelConfig.from_env()

    assert config.provider == "openai"
    assert config.api_key == "sk-test"
    assert config.base_url == "https://example.test/v1"
    assert config.model == "custom-model"
