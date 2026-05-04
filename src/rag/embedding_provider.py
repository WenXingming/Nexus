"""OpenAI Embeddings 适配器。

本模块提供 RAG 所需的最小文本嵌入能力，将 OpenAI SDK 的
embeddings 接口适配为 src.core_contracts.rag_contracts.EmbeddingProvider
协议，供 RagGateway 通过依赖注入使用。
"""

from __future__ import annotations

import openai
from openai import OpenAI

from src.core_contracts.model_config import ModelConfig, RagModelConfig


class OpenAIEmbeddingProvider:
    """基于 OpenAI Embeddings API 的文本嵌入提供者。"""

    def __init__(
        self,
        model_config: ModelConfig,
        rag_config: RagModelConfig,
    ) -> None:
        """初始化嵌入提供者并创建底层 OpenAI 客户端。

        Args:
            model_config (ModelConfig): 包含 API 凭据与聊天模型名的配置。
            rag_config (RagModelConfig): RAG embeddings 配置。
        Raises:
            ValueError: API key 或 embeddings/chat 模型名为空字符串时抛出。
        """
        if not model_config.api_key or not model_config.api_key.strip():
            raise ValueError("ModelConfig.api_key 不能为空字符串。")
        if not rag_config.embedding_model.strip():
            raise ValueError("RagModelConfig.embedding_model 不能为空字符串。")
        if not model_config.model_name.strip():
            raise ValueError("ModelConfig.model_name 不能为空字符串。")
        self._embedding_model = rag_config.embedding_model  # str：首选 embeddings 模型名称。
        self._fallback_model = model_config.model_name      # str：降级使用的聊天模型名称。
        self._client = OpenAI(  # OpenAI：底层 OpenAI SDK 客户端。
            api_key=model_config.api_key,
            base_url=model_config.base_url,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """将文本列表批量转换为嵌入向量列表。

        Args:
            texts (list[str]): 待嵌入的文本列表。
        Returns:
            list[list[float]]: 与输入文本等长的嵌入向量列表。
        Raises:
            ValueError: texts 中存在空白字符串时抛出。
            RuntimeError: OpenAI 接口异常或响应结构异常时抛出。
            TimeoutError: 请求超时时抛出。
            PermissionError: API 认证失败时抛出。
        """
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("texts 中不能包含空白字符串。")

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
        """调用 OpenAI embeddings.create 接口。

        Args:
            model (str): 本次调用的模型名称。
            texts (list[str]): 待嵌入文本列表。
        Returns:
            object: OpenAI SDK 返回的 embeddings 响应对象。
        Raises:
            openai.OpenAIError: 底层 SDK 抛出的调用异常。
        """
        return self._client.embeddings.create(model=model, input=texts)

    def _should_fallback_to_chat_model(self, error: openai.OpenAIError) -> bool:
        """判断是否应从 embeddings 模型降级到聊天模型。

        Args:
            error (openai.OpenAIError): 首选 embeddings 模型调用异常。
        Returns:
            bool: 仅当错误为模型不存在/无权限且降级模型与首选模型不同才返回 True。
        Raises:
            None
        """
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
        """将 OpenAI SDK 异常翻译为 Python 内置异常。

        Args:
            error (openai.OpenAIError): 原始 OpenAI SDK 异常。
        Returns:
            Exception: 翻译后的标准异常对象。
        Raises:
            None
        """
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
