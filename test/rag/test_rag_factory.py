"""RAG 工厂装配测试。

验证 build_rag_gateway 在模块入口内部创建 OpenAIEmbeddingProvider，
调用方无需感知 embeddings provider 具体实现。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.core_contracts.model_config import ModelConfig, RagModelConfig
from src.rag import build_rag_gateway


def test_build_rag_gateway_creates_embedding_provider_internally() -> None:
    model_config = ModelConfig(api_key="sk-test", model_name="gpt-4o-mini")
    rag_config = RagModelConfig(embedding_model="text-embedding-3-small")
    model_client = MagicMock()

    with patch("src.rag.embedding_provider.OpenAIEmbeddingProvider") as mock_provider, \
        patch("src.rag.document_loader.DocumentLoader") as mock_loader:
        gateway = build_rag_gateway(
            model_client=model_client,
            model_config=model_config,
            rag_config=rag_config,
        )

    mock_provider.assert_called_once_with(
        model_config=model_config,
        rag_config=rag_config,
    )
    assert gateway._embedding_provider is mock_provider.return_value
    assert gateway._document_loader is mock_loader.return_value
