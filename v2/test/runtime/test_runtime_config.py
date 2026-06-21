from src.runtime.config import AgentConfig


def test_agent_config_defaults_without_system_prompt() -> None:
    config = AgentConfig()

    assert config.system_prompt is None


def test_agent_config_accepts_system_prompt() -> None:
    config = AgentConfig(system_prompt="You are Nexus.")

    assert config.system_prompt == "You are Nexus."


def test_agent_config_from_env_defaults_without_system_prompt(monkeypatch) -> None:
    monkeypatch.delenv("NEXUS_SYSTEM_PROMPT", raising=False)

    config = AgentConfig.from_env()

    assert config.system_prompt is None


def test_agent_config_from_env_reads_system_prompt(monkeypatch) -> None:
    monkeypatch.setenv("NEXUS_SYSTEM_PROMPT", "You are Nexus.")

    config = AgentConfig.from_env()

    assert config.system_prompt == "You are Nexus."


def test_agent_config_from_env_treats_empty_prompt_as_none(monkeypatch) -> None:
    monkeypatch.setenv("NEXUS_SYSTEM_PROMPT", "")

    config = AgentConfig.from_env()

    assert config.system_prompt is None
