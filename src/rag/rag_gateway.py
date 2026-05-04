"""RAG 模块唯一公开门面（Facade）。

外部调用者只依赖：
    - RagGateway（本文件）
    - src.core_contracts.rag_contracts 中的请求/结果契约与异常

网关职责：接收外部契约、依赖注入分发、调度内部组件、翻译异常。
网关本身不包含任何分块、向量运算或提示词构建逻辑。
"""

from __future__ import annotations

import time

from src.core_contracts.model_contracts import Message
from src.core_contracts.rag_contracts import (
    RagChunk,
    RagDocument,
    RagError,
    RagIndexRequest,
    RagIndexResult,
    RagRetrieveRequest,
    RagRetrieveResult,
    RagRetrievedChunk,
)
from src.rag.document_chunker import DocumentChunker
from src.rag.document_loader import DocumentLoader
from src.rag.embedding_provider import OpenAIEmbeddingProvider
from src.rag.prompt_builder import PromptBuilder
from src.rag.vector_store import VectorStore


class RagGateway:
    """RAG 模块门面与流水线编排器。

    核心工作流：
      index                    → 切分 → embed_chunks → 写入向量存储
      retrieve                 → embed_query → 余弦相似度检索 → 返回分块
      retrieve_and_build_messages → retrieve + PromptBuilder 组合
    """

    def __init__(
        self,
        *,
        embedding_provider: OpenAIEmbeddingProvider,
        chunker: DocumentChunker,
        vector_store: VectorStore,
        document_loader: DocumentLoader | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        """初始化 RagGateway，注入全部流水线依赖。

        Args:
            embedding_provider: 文本嵌入向量提供者。
            chunker: 文档滑动窗口切分器。
            vector_store: 基于余弦相似度的内存向量存储。
            document_loader: 可选文档装载器，用于 source_path 闭环索引。
        """
        self._embedding_provider = embedding_provider
        self._chunker = chunker
        self._vector_store = vector_store
        self._document_loader = document_loader
        self._prompt_builder = prompt_builder

    # ── 公有接口 ────────────────────────────────────────────────────────────

    def index(self, request: RagIndexRequest) -> RagIndexResult:
        """索引一批文档：切分 → 嵌入 → 写入向量存储。

        Args:
            request (RagIndexRequest): 包含文档列表与索引参数的标准请求契约。
        Returns:
            RagIndexResult: 包含索引数量统计与耗时的结果契约。
        Raises:
            ValueError: request.documents 为空时抛出。
            RagError: 分块、嵌入或写入任一环节失败时抛出。
        """
        documents = self._resolve_index_documents(request)
        t_start = time.monotonic()
        try:
            all_chunks = self._chunk_documents(
                documents=documents,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
            )
            if all_chunks:
                embeddings = self._embedding_provider.embed_chunks(all_chunks)
                self._vector_store.upsert(
                    name=request.collection_name,
                    embeddings=embeddings,
                )
            return RagIndexResult(
                collection_name=request.collection_name,
                docs_indexed=len(documents),
                chunks_created=len(all_chunks),
                duration_s=time.monotonic() - t_start,
            )
        except RagError:
            raise
        except Exception as exc:
            raise RagError(f"索引操作意外失败: {exc}") from exc

    def retrieve(self, request: RagRetrieveRequest) -> RagRetrieveResult:
        """在指定集合中执行纯向量相似度检索，不调用模型生成。

        Args:
            request (RagRetrieveRequest): 包含查询文本与检索参数的标准请求契约。
        Returns:
            RagRetrieveResult: 包含命中分块列表（按相似度降序）与耗时的结果契约。
        Raises:
            ValueError: request.query 为空字符串时抛出。
            RagError: 指定集合不存在或嵌入/检索失败时抛出。
        """
        if not request.query.strip():
            raise ValueError("RagRetrieveRequest.query 不能为空字符串。")
        t_start = time.monotonic()
        try:
            retrieved_chunks = self._search_collection(
                query=request.query,
                collection_name=request.collection_name,
                top_k=request.top_k,
            )
            return RagRetrieveResult(
                query=request.query,
                collection_name=request.collection_name,
                retrieved_chunks=tuple(retrieved_chunks),
                duration_s=time.monotonic() - t_start,
            )
        except RagError:
            raise
        except Exception as exc:
            raise RagError(f"检索操作意外失败: {exc}") from exc
    
    def retrieve_and_build_messages(
        self,
        query: str,
        collection_name: str = 'default',
        top_k: int = 5,
        max_tokens: int = 1024,
        system_override: str | None = None,
    ) -> list[Message]:
        """检索并构建 LLM-ready Message 列表（组合 retrieve + PromptBuilder）。

        Args:
            query (str): 用户的原始问题文本。
            collection_name (str): 目标向量集合名称。
            top_k (int): 检索返回的分块数量上限。
            max_tokens (int): 回答所允许的最大 token 数。
            system_override (str | None): 覆盖默认系统提示词。
        Returns:
            list[Message]: 包含 system 和 user 两条消息的列表。
        Raises:
            ValueError: query 为空字符串时抛出。
            RagError: 检索或嵌入失败时抛出。
        """
        result = self.retrieve(
            RagRetrieveRequest(query=query, collection_name=collection_name, top_k=top_k)
        )
        return self._prompt_builder.build_messages(
            query=query,
            chunks=list(result.retrieved_chunks),
            max_tokens=max_tokens,
            system_override=system_override,
        )

    def drop_collection(self, collection_name: str) -> None:
        """删除指定名称的向量集合及其全部分块数据。"""
        self._vector_store.drop(collection_name)

    def list_collections(self) -> list[str]:
        """列出当前向量存储中所有已建立的集合名称。"""
        return self._vector_store.list_names()
    
    # ── 私有辅助方法 ───────────────────────────────────────────────────────

    def _chunk_documents(
        self,
        documents: list[RagDocument],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[RagChunk]:
        """将文档列表批量切分为分块扁平列表。"""
        result: list[RagChunk] = []
        for doc in documents:
            result.extend(
                self._chunker.chunk(
                    document=doc,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
        return result

    def _resolve_index_documents(self, request: RagIndexRequest) -> list[RagDocument]:
        """根据 documents 或 source_path 解析本次索引的文档列表。"""
        if request.documents:
            return list(request.documents)

        source_path = (request.source_path or '').strip()
        if not source_path:
            raise ValueError('RagIndexRequest.documents 不能为空，或提供非空 source_path。')
        if self._document_loader is None:
            raise RagError('RagGateway 未配置文档装载器，无法处理 source_path。')

        try:
            documents = self._document_loader.load(source_path)
        except RagError:
            raise
        except Exception as exc:
            raise RagError(f'文档加载失败: {exc}') from exc

        if not documents:
            raise ValueError('RagIndexRequest.documents 不能为空，至少需要包含一篇文档。')
        return list(documents)

    def _search_collection(
        self,
        query: str,
        collection_name: str,
        top_k: int,
    ) -> list[RagRetrievedChunk]:
        """嵌入查询文本并在指定集合中检索最相似分块。"""
        try:
            query_vector = self._embedding_provider.embed_query(query)
        except Exception as exc:
            raise RagError(f"查询嵌入失败: {exc}") from exc

        try:
            return self._vector_store.search(
                name=collection_name,
                query_vector=query_vector,
                top_k=top_k,
            )
        except RagError:
            raise
        except Exception as exc:
            raise RagError(f"向量检索失败: {exc}") from exc
