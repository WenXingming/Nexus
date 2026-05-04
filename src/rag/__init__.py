"""RAG（检索增强生成）模块公开入口。

外部代码只允许从本文件导入，不得直接导入内部实现类。

公开导出：
    - RagGateway           : RAG 模块的唯一门面，处理索引与检索流水线编排。
    - build_rag_gateway    : 标准装配工厂，接收外部依赖并完成全链路注入。

所有请求/结果契约与异常类型均定义在 src.core_contracts.rag_contracts，
请直接从该模块导入，无需经过本模块转发。
"""

from __future__ import annotations

from core_contracts.model_config import ModelConfig, RagModelConfig
from src.rag.document_chunker import DocumentChunker as _DocumentChunker
from src.rag.document_loader import DocumentLoader as _DocumentLoader
from src.rag.embedding_provider import OpenAIEmbeddingProvider as _OpenAIEmbeddingProvider
from src.rag.vector_store import VectorStore as _VectorStore
from src.rag.prompt_builder import PromptBuilder as _PromptBuilder
from src.rag.rag_gateway import RagGateway


__all__ = ['RagGateway', 'build_rag_gateway']


def build_rag_gateway(
    *,
    model_config: 'ModelConfig',
    rag_config: 'RagModelConfig',
) -> RagGateway:
    """标准工厂函数：装配并返回一个开箱即用的 RagGateway 实例。

    调用方只需提供基础配置，工厂负责构造 embeddings provider
    及全部内部组件并完成依赖注入；外部代码无需感知 DocumentChunker /
    VectorStore / OpenAIEmbeddingProvider 的存在。

    Args:
        model_config (ModelConfig): 用于创建底层 OpenAI 客户端的聊天模型配置。
        rag_config (RagModelConfig): RAG embeddings 相关配置。
    Returns:
        RagGateway: 已完整装配的 RAG 门面网关实例。
    """

    document_loader = _DocumentLoader()
    document_chunker = _DocumentChunker()
    embedding_provider = _OpenAIEmbeddingProvider(
        model_config=model_config,
        rag_config=rag_config,
    )
    vector_store = _VectorStore()
    prompt_builder = _PromptBuilder()

    return RagGateway(
        embedding_provider=embedding_provider,
        chunker=document_chunker,
        vector_store=vector_store,
        document_loader=document_loader,
        prompt_builder=prompt_builder,
    )
