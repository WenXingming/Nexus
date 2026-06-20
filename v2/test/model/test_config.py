from src.model.config import ModelConfig


def test_model_config_defaults_to_fake(monkeypatch) -> None:
    monkeypatch.delenv("NEXUS_MODEL_PROVIDER", raising=False)

    config = ModelConfig.from_env()

    assert config.provider == "fake"


def test_model_config_reads_provider_from_env(monkeypatch) -> None:
    monkeypatch.setenv("NEXUS_MODEL_PROVIDER", "openai")

    config = ModelConfig.from_env()

    assert config.provider == "openai"
