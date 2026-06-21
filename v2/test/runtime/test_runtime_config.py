from src.runtime.config import AgentConfig


def test_agent_config_defaults_without_system_prompt() -> None:
    config = AgentConfig()

    assert config.system_prompt is None


def test_agent_config_accepts_system_prompt() -> None:
    config = AgentConfig(system_prompt="You are Nexus.")

    assert config.system_prompt == "You are Nexus."
