"""模型配置契约定义。

本文件聚焦于模型相关配置：
1. 通用聊天模型配置（ModelConfig）
2. RAG embeddings 配置（RagModelConfig）
"""

from __future__ import annotations

import os
from typing import ClassVar
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """LLM 客户端连接与默认行为配置。

    Attributes:
        api_key: API 密钥。
        base_url: 自定义 API 基础地址，为 None 时使用官方默认地址。
        model_name: 默认模型名称（如 "gpt-4o"）。
        temperature: 默认采样温度，范围 [0, 2]。
        max_tokens: 默认最大生成 Token 数。
    """

    MIN_MAX_TOKENS: ClassVar[int] = 1
    MAX_MAX_TOKENS: ClassVar[int] = 65_536
    DEFAULT_MAX_TOKENS: ClassVar[int] = 4_096

    api_key: str
    base_url: str | None = None
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        """校验配置边界。"""
        if not (self.MIN_MAX_TOKENS <= self.max_tokens <= self.MAX_MAX_TOKENS):
            raise ValueError(
                f"ModelConfig.max_tokens 必须在 [{self.MIN_MAX_TOKENS}, {self.MAX_MAX_TOKENS}] 范围内，当前值: {self.max_tokens}"
            )

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """从环境变量读取配置并创建 ModelConfig 实例。

        读取的环境变量：
            OPENAI_API_KEY: API 密钥（必需）。
            OPENAI_BASE_URL: 自定义 API 基础地址（可选）。
            OPENAI_MODEL: 默认模型名称（可选，默认 "gpt-4o"）。
            OPENAI_TEMPERATURE: 默认采样温度（可选，默认 0.7）。
            OPENAI_MAX_TOKENS: 默认最大生成 Token 数（可选，默认 4096）。

        Returns:
            ModelConfig: 从环境变量构建的配置实例。

        Raises:
            ValueError: 当 OPENAI_API_KEY 未设置时抛出。
        """
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key.strip():
            raise ValueError("环境变量 OPENAI_API_KEY 未设置或为空")

        raw_max_tokens = os.environ.get("OPENAI_MAX_TOKENS", str(cls.DEFAULT_MAX_TOKENS))
        try:
            max_tokens = int(raw_max_tokens)
        except ValueError as exc:
            raise ValueError("环境变量 OPENAI_MAX_TOKENS 必须为整数。") from exc
        if not (cls.MIN_MAX_TOKENS <= max_tokens <= cls.MAX_MAX_TOKENS):
            raise ValueError(
                f"环境变量 OPENAI_MAX_TOKENS 必须在 [{cls.MIN_MAX_TOKENS}, {cls.MAX_MAX_TOKENS}] 范围内，当前值: {max_tokens}"
            )

        return cls(
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            model_name=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=max_tokens,
        )


@dataclass(frozen=True)
class RagModelConfig:
    """RAG embeddings 相关配置。

    Attributes:
        embedding_model: 首选 embeddings 模型名称。
    """

    embedding_model: str = "text-embedding-3-small"

    @classmethod
    def from_env(cls) -> "RagModelConfig":
        """从环境变量读取 RAG embeddings 配置。

        环境变量优先级：
            1. RAG_EMBEDDING_MODEL
            2. OPENAI_EMBEDDING_MODEL
            3. 默认值 text-embedding-3-small

        Returns:
            RagModelConfig: 从环境变量构建的 embeddings 配置实例。

        Raises:
            ValueError: 环境变量存在但为空白字符串时抛出。
        """
        raw = os.environ.get("RAG_EMBEDDING_MODEL")
        if raw is None:
            raw = os.environ.get("OPENAI_EMBEDDING_MODEL")

        if raw is None:
            return cls()

        model = raw.strip()
        if not model:
            raise ValueError("RAG_EMBEDDING_MODEL / OPENAI_EMBEDDING_MODEL 不能为空白字符串")
        return cls(embedding_model=model)
