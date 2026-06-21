from pathlib import Path

from src.memory.config import MemoryConfig


def test_memory_config_defaults_to_memory_store(monkeypatch) -> None:
    monkeypatch.delenv("NEXUS_MEMORY_STORE", raising=False)
    monkeypatch.delenv("NEXUS_SESSION_ROOT", raising=False)

    config = MemoryConfig.from_env()

    assert config.store == "memory"
    assert config.root == Path(".nexus-v2/sessions")


def test_memory_config_reads_values_from_env(monkeypatch) -> None:
    monkeypatch.setenv("NEXUS_MEMORY_STORE", "file")
    monkeypatch.setenv("NEXUS_SESSION_ROOT", "custom/sessions")

    config = MemoryConfig.from_env()

    assert config.store == "file"
    assert config.root == Path("custom/sessions")
