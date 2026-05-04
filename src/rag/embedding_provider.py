"""OpenAI Embeddings 适配器。

本模块提供 RAG 所需的最小文本嵌入能力，将 OpenAI SDK 的
embeddings 接口适配为嵌入向量提供者，供 RagGateway 通过依赖注入使用。
"""

from __future__ import annotations

import openai
from openai import OpenAI

from src.core_contracts.model_config import ModelConfig, RagModelConfig
from src.core_contracts.rag_contracts import RagChunk, RagEmbedding


class OpenAIEmbeddingProvider:
    """基于 OpenAI Embeddings API 的文本嵌入提供者。

    公开接口：
      - embed_chunks: 索引路径，list[RagChunk] → list[RagEmbedding]
      - embed_query:  检索路径，单条查询 → 单个向量
    """

    _BATCH_SIZE = 128

    def __init__(
        self,
        model_config: ModelConfig,
        rag_config: RagModelConfig,
    ) -> None:
        """初始化嵌入提供者并创建底层 OpenAI 客户端。"""
        if not model_config.api_key or not model_config.api_key.strip():
            raise ValueError("ModelConfig.api_key 不能为空字符串。")
        if not rag_config.embedding_model.strip():
            raise ValueError("RagModelConfig.embedding_model 不能为空字符串。")
        if not model_config.model_name.strip():
            raise ValueError("ModelConfig.model_name 不能为空字符串。")
        self._embedding_model = rag_config.embedding_model
        self._fallback_model = model_config.model_name
        self._client = OpenAI(
            api_key=model_config.api_key,
            base_url=model_config.base_url,
        )

    # ── 公有接口 ────────────────────────────────────────────────────────────

    def embed_chunks(self, chunks: list[RagChunk]) -> list[RagEmbedding]:
        """索引路径：将分块列表嵌入并绑定为 RagEmbedding 列表。

        Args:
            chunks (list[RagChunk]): 待嵌入的分块列表。
        Returns:
            list[RagEmbedding]: 与输入等长的嵌入向量列表，每个元素绑定其原始分块。
        Raises:
            ValueError: chunks 为空或存在空白内容时抛出。
            RuntimeError: OpenAI 接口异常或响应数量不匹配时抛出。
        """
        if not chunks:
            return []
        texts = [chunk.content for chunk in chunks]
        vectors = self._embed_texts(texts)
        return [
            RagEmbedding(chunk=chunk, vector=vector)
            for chunk, vector in zip(chunks, vectors)
        ]

    def embed_query(self, query: str) -> list[float]:
        """检索路径：将单条查询文本嵌入为向量。

        Args:
            query (str): 查询文本。
        Returns:
            list[float]: 查询嵌入向量。
        Raises:
            ValueError: query 为空白字符串时抛出。
            RuntimeError: OpenAI 接口异常时抛出。
        """
        return self._embed_texts([query])[0]

    # ── 私有方法 ────────────────────────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """底层：将文本列表批量转换为嵌入向量列表。

        内部自动分批，降低峰值内存与第三方 API 限频风险。
        """
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("texts 中不能包含空白字符串。")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[start:start + self._BATCH_SIZE]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """单批嵌入：调用 OpenAI API 并校验返回数量。"""
        try:
            response = self._create_embeddings(model=self._embedding_model, texts=texts)
        except openai.OpenAIError as exc:
            if self._should_fallback_to_chat_model(exc):
                try:
                    response = self._create_embeddings(model=self._fallback_model, texts=texts)
                except openai.OpenAIError as fallback_exc:
                    raise RuntimeError(
                        "embeddings 模型自动降级失败："
                        f"primary={self._embedding_model} err={exc}; "
                        f"fallback={self._fallback_model} err={fallback_exc}"
                    ) from fallback_exc
            else:
                raise self._translate_error(exc) from exc

        if len(response.data) != len(texts):
            raise RuntimeError(
                "Embeddings API 返回数量异常："
                f"期望 {len(texts)} 条，实际收到 {len(response.data)} 条。"
            )
        return [item.embedding for item in response.data]

    def _create_embeddings(self, model: str, texts: list[str]) -> object:
        """调用 OpenAI embeddings.create 接口。"""
        return self._client.embeddings.create(model=model, input=texts)

    def _should_fallback_to_chat_model(self, error: openai.OpenAIError) -> bool:
        """判断是否应从 embeddings 模型降级到聊天模型。"""
        if self._fallback_model == self._embedding_model:
            return False

        msg = str(error).lower()
        model_not_found_markers = (
            "model_not_found",
            "does not exist",
            "do not have access",
        )
        if any(marker in msg for marker in model_not_found_markers):
            return True

        not_found_error_cls = getattr(openai, "NotFoundError", None)
        if not_found_error_cls is not None and isinstance(error, not_found_error_cls):
            return True

        return False

    def _translate_error(self, error: openai.OpenAIError) -> Exception:
        """将 OpenAI SDK 异常翻译为 Python 内置异常。"""
        msg = str(error)
        if isinstance(error, openai.AuthenticationError):
            return PermissionError(msg)
        if isinstance(error, openai.RateLimitError):
            return RuntimeError(msg)
        if isinstance(error, openai.BadRequestError):
            return ValueError(msg)
        if isinstance(error, openai.APITimeoutError):
            return TimeoutError(msg)
        if isinstance(error, openai.APIConnectionError):
            return RuntimeError(msg)
        return RuntimeError(msg)
