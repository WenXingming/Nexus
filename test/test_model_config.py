"""model_config 契约测试。"""

from __future__ import annotations

import pytest

from src.core_contracts.model_config import ModelConfig, RagModelConfig


class TestModelConfigFromEnv:
    def test_requires_openai_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            ModelConfig.from_env()

    def test_uses_safe_default_max_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)

        cfg = ModelConfig.from_env()

        assert cfg.max_tokens == 4096

    def test_rejects_out_of_range_max_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "70000")

        with pytest.raises(ValueError, match="OPENAI_MAX_TOKENS"):
            ModelConfig.from_env()

    def test_rejects_non_integer_max_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "abc")

        with pytest.raises(ValueError, match="OPENAI_MAX_TOKENS"):
            ModelConfig.from_env()


class TestRagModelConfigFromEnv:
    def test_uses_default_when_no_env_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RAG_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)

        cfg = RagModelConfig.from_env()

        assert cfg.embedding_model == "text-embedding-3-small"

    def test_prefers_rag_embedding_model_over_openai_embedding_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAG_EMBEDDING_MODEL", "rag-model")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "openai-embedding-model")

        cfg = RagModelConfig.from_env()

        assert cfg.embedding_model == "rag-model"

    def test_fallbacks_to_openai_embedding_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RAG_EMBEDDING_MODEL", raising=False)
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "openai-embedding-model")

        cfg = RagModelConfig.from_env()

        assert cfg.embedding_model == "openai-embedding-model"

    def test_rejects_blank_rag_embedding_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_EMBEDDING_MODEL", "   ")

        with pytest.raises(ValueError, match="不能为空白字符串"):
            RagModelConfig.from_env()

    def test_rejects_blank_openai_embedding_model_when_rag_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RAG_EMBEDDING_MODEL", raising=False)
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "   ")

        with pytest.raises(ValueError, match="不能为空白字符串"):
            RagModelConfig.from_env()
