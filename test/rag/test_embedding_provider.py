"""OpenAIEmbeddingProvider 单元测试。

通过 patch src.rag.embedding_provider.OpenAI 严格隔离 openai SDK，
验证 embeddings 调用编排、返回解析、异常翻译与模型自动降级策略。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import openai
import pytest

from src.core_contracts.model_config import ModelConfig, RagModelConfig
from src.rag.embedding_provider import OpenAIEmbeddingProvider


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        model_name="gpt-4o-mini",
    )


@pytest.fixture
def rag_config() -> RagModelConfig:
    return RagModelConfig(embedding_model="text-embedding-3-small")


@pytest.fixture
def mock_openai_cls() -> MagicMock:
    with patch("src.rag.embedding_provider.OpenAI") as mock_cls:
        yield mock_cls


@pytest.fixture
def provider(
    model_config: ModelConfig,
    rag_config: RagModelConfig,
    mock_openai_cls: MagicMock,
) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(model_config=model_config, rag_config=rag_config)


class TestInit:
    def test_creates_openai_client_with_correct_args(
        self,
        model_config: ModelConfig,
        rag_config: RagModelConfig,
        mock_openai_cls: MagicMock,
    ) -> None:
        OpenAIEmbeddingProvider(model_config=model_config, rag_config=rag_config)
        mock_openai_cls.assert_called_once_with(
            api_key=model_config.api_key,
            base_url=model_config.base_url,
        )

    def test_raises_on_empty_api_key(self, rag_config: RagModelConfig) -> None:
        with pytest.raises(ValueError, match="api_key"):
            OpenAIEmbeddingProvider(
                model_config=ModelConfig(api_key=""),
                rag_config=rag_config,
            )

    def test_raises_on_empty_embedding_model(self, model_config: ModelConfig) -> None:
        with pytest.raises(ValueError, match="embedding_model"):
            OpenAIEmbeddingProvider(
                model_config=model_config,
                rag_config=RagModelConfig(embedding_model="   "),
            )


class TestEmbedTexts:
    def test_returns_empty_list_for_empty_input(self, provider: OpenAIEmbeddingProvider) -> None:
        assert provider.embed_texts([]) == []

    def test_calls_embeddings_api_with_primary_model_and_input(
        self, provider: OpenAIEmbeddingProvider
    ) -> None:
        response = MagicMock()
        response.data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
        provider._client.embeddings.create.return_value = response

        result = provider.embed_texts(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        provider._client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["hello", "world"],
        )

    def test_fallbacks_to_chat_model_when_primary_model_not_found(
        self, provider: OpenAIEmbeddingProvider
    ) -> None:
        not_found = openai.BadRequestError(
            message="The model `text-embedding-3-small` does not exist or you do not have access to it. code: model_not_found",
            response=MagicMock(),
            body=None,
        )
        fallback_response = MagicMock()
        fallback_response.data = [MagicMock(embedding=[0.6, 0.7])]
        provider._client.embeddings.create.side_effect = [not_found, fallback_response]

        result = provider.embed_texts(["hello"])

        assert result == [[0.6, 0.7]]
        assert provider._client.embeddings.create.call_count == 2
        first_call = provider._client.embeddings.create.call_args_list[0].kwargs
        second_call = provider._client.embeddings.create.call_args_list[1].kwargs
        assert first_call["model"] == "text-embedding-3-small"
        assert second_call["model"] == "gpt-4o-mini"

    def test_raises_combined_error_when_fallback_also_fails(
        self, provider: OpenAIEmbeddingProvider
    ) -> None:
        primary_error = openai.BadRequestError(
            message="model_not_found",
            response=MagicMock(),
            body=None,
        )
        fallback_error = openai.BadRequestError(
            message="model_not_supported",
            response=MagicMock(),
            body=None,
        )
        provider._client.embeddings.create.side_effect = [primary_error, fallback_error]

        with pytest.raises(RuntimeError, match="自动降级失败") as exc_info:
            provider.embed_texts(["hello"])

        assert "primary=text-embedding-3-small" in str(exc_info.value)
        assert "fallback=gpt-4o-mini" in str(exc_info.value)

    def test_raises_on_blank_text(self, provider: OpenAIEmbeddingProvider) -> None:
        with pytest.raises(ValueError, match="空白字符串"):
            provider.embed_texts(["hello", "   "])

    def test_raises_when_response_count_mismatch(self, provider: OpenAIEmbeddingProvider) -> None:
        response = MagicMock()
        response.data = [MagicMock(embedding=[0.1, 0.2])]
        provider._client.embeddings.create.return_value = response

        with pytest.raises(RuntimeError, match="返回数量异常"):
            provider.embed_texts(["hello", "world"])


class TestErrorTranslation:
    def test_authentication_error(self, provider: OpenAIEmbeddingProvider) -> None:
        original = openai.AuthenticationError(message="bad key", response=MagicMock(), body=None)
        provider._client.embeddings.create.side_effect = original
        with pytest.raises(PermissionError):
            provider.embed_texts(["hello"])

    def test_bad_request_error_without_fallback_marker(self, provider: OpenAIEmbeddingProvider) -> None:
        original = openai.BadRequestError(message="bad param", response=MagicMock(), body=None)
        provider._client.embeddings.create.side_effect = original
        with pytest.raises(ValueError):
            provider.embed_texts(["hello"])

    def test_timeout_error(self, provider: OpenAIEmbeddingProvider) -> None:
        original = openai.APITimeoutError(request=MagicMock())
        provider._client.embeddings.create.side_effect = original
        with pytest.raises(TimeoutError):
            provider.embed_texts(["hello"])
