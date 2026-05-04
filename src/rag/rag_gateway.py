"""RAG 模块唯一公开门面（Facade）。

外部调用者只依赖：
    - RagGateway（本文件）
    - src.core_contracts.rag_contracts 中的请求/结果契约与异常

网关职责：接收外部契约、依赖注入分发、调度内部组件、翻译异常。
网关本身不包含任何分块、向量运算或提示词构建逻辑。
"""

from __future__ import annotations

import time

from src.core_contracts.rag_contracts import (
    EmbeddingProvider,
    RagChunk,
    RagCollectionNotFoundError,
    RagDocument,
    RagDocumentLoader,
    RagError,
    RagIndexError,
    RagIndexRequest,
    RagIndexResult,
    RagQueryError,
    RagQueryRequest,
    RagQueryResult,
    RagRetrieveError,
    RagRetrieveRequest,
    RagRetrieveResult,
    RagRetrievedChunk,
)
from src.rag.answer_generator import AnswerGenerator
from src.rag.chunker import DocumentChunker
from src.rag.vector_store import VectorStore


class RagGateway:
    """RAG 模块门面与流水线编排器。

    核心工作流：
      index_documents → 切分 → 嵌入 → 写入向量存储
      retrieve        → 嵌入查询 → 余弦相似度检索 → 返回分块
      query           → retrieve → 构建提示词 → 调用模型 → 返回回答
    """

    _EMBED_BATCH_SIZE = 128

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        chunker: DocumentChunker,
        vector_store: VectorStore,
        answer_generator: AnswerGenerator,
        document_loader: RagDocumentLoader | None = None,
    ) -> None:
        """初始化 RagGateway，注入全部流水线依赖。

        Args:
            embedding_provider (EmbeddingProvider): 文本嵌入向量提供者，须实现 EmbeddingProvider 协议。
            chunker (DocumentChunker): 文档滑动窗口切分器。
            vector_store (VectorStore): 基于余弦相似度的内存向量存储。
            answer_generator (AnswerGenerator): 提示词构建与模型调用回答生成器。
        """
        self._embedding_provider = embedding_provider  # EmbeddingProvider：注入的嵌入向量提供者。
        self._chunker = chunker                        # DocumentChunker：注入的文档切分器。
        self._vector_store = vector_store              # VectorStore：注入的向量存储实例。
        self._answer_generator = answer_generator      # AnswerGenerator：注入的回答生成器。
        self._document_loader = document_loader        # RagDocumentLoader | None：可选文档装载器，用于 source_path 闭环索引。

    # ── 公有接口 ────────────────────────────────────────────────────────────

    def index_documents(self, request: RagIndexRequest) -> RagIndexResult:
        """索引一批文档：切分 → 嵌入 → 写入向量存储。

        Args:
            request (RagIndexRequest): 包含文档列表与索引参数的标准请求契约。
        Returns:
            RagIndexResult: 包含索引数量统计与耗时的结果契约。
        Raises:
            ValueError: request.documents 为空时抛出。
            RagIndexError: 分块、嵌入或写入任一环节失败时抛出。
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
                vectors = self._embed_texts([chunk.content for chunk in all_chunks])
                self._vector_store.upsert(
                    name=request.collection_name,
                    chunks=all_chunks,
                    vectors=vectors,
                )
            return RagIndexResult(
                collection_name=request.collection_name,
                docs_indexed=len(documents),
                chunks_created=len(all_chunks),
                duration_s=time.monotonic() - t_start,
            )
        except RagIndexError:
            raise
        except RagError:
            raise
        except Exception as exc:
            raise RagIndexError(f"索引操作意外失败: {exc}") from exc

    def retrieve(self, request: RagRetrieveRequest) -> RagRetrieveResult:
        """在指定集合中执行纯向量相似度检索，不调用模型生成。

        Args:
            request (RagRetrieveRequest): 包含查询文本与检索参数的标准请求契约。
        Returns:
            RagRetrieveResult: 包含命中分块列表（按相似度降序）与耗时的结果契约。
        Raises:
            ValueError: request.query 为空字符串时抛出。
            RagCollectionNotFoundError: 指定集合尚未通过 index_documents 建立时抛出。
            RagRetrieveError: 嵌入查询或向量检索失败时抛出。
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
        except (RagCollectionNotFoundError, RagRetrieveError):
            raise
        except RagError:
            raise
        except Exception as exc:
            raise RagRetrieveError(f"检索操作意外失败: {exc}") from exc

    def query(self, request: RagQueryRequest) -> RagQueryResult:
        """执行完整 RAG 流水线：检索 → 构建上下文 → 调用模型生成回答。

        Args:
            request (RagQueryRequest): 包含用户问题与生成参数的标准请求契约。
        Returns:
            RagQueryResult: 包含检索上下文、生成回答及 token 统计的结果契约。
        Raises:
            ValueError: request.query 为空字符串时抛出。
            RagCollectionNotFoundError: 指定集合不存在时抛出。
            RagQueryError: 检索或模型调用失败时抛出。
        """
        if not request.query.strip():
            raise ValueError("RagQueryRequest.query 不能为空字符串。")
        t_start = time.monotonic()
        try:
            retrieve_result = self.retrieve(
                RagRetrieveRequest(
                    query=request.query,
                    collection_name=request.collection_name,
                    top_k=request.top_k,
                )
            )
            answer, prompt_tokens, completion_tokens = self._answer_generator.generate(
                query=request.query,
                chunks=list(retrieve_result.retrieved_chunks),
                max_tokens=request.answer_max_tokens,
                system_override=request.system_prompt_override,
            )
            return RagQueryResult(
                query=request.query,
                collection_name=request.collection_name,
                retrieved_chunks=retrieve_result.retrieved_chunks,
                answer=answer,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_s=time.monotonic() - t_start,
            )
        except (RagCollectionNotFoundError, RagQueryError):
            raise
        except RagError:
            raise
        except Exception as exc:
            raise RagQueryError(f"RAG 问答流水线意外失败: {exc}") from exc

    def drop_collection(self, collection_name: str) -> None:
        """删除指定名称的向量集合及其全部分块数据。

        Args:
            collection_name (str): 要删除的集合名称。
        Raises:
            RagCollectionNotFoundError: 集合不存在时抛出。
        """
        self._vector_store.drop(collection_name)

    def list_collections(self) -> list[str]:
        """列出当前向量存储中所有已建立的集合名称。

        Returns:
            list[str]: 集合名称列表；若尚未索引任何文档则返回空列表。
        """
        return self._vector_store.list_names()

    # ── 私有辅助方法（深度优先）──────────────────────────────────────────────

    def _chunk_documents(
        self,
        documents: list[RagDocument],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[RagChunk]:
        """将文档列表批量切分为分块扁平列表。

        Args:
            documents (list[RagDocument]): 待切分的原始文档列表。
            chunk_size (int): 每个分块的最大字符数。
            chunk_overlap (int): 相邻分块间的重叠字符数。
        Returns:
            list[RagChunk]: 所有文档切分结果的扁平化列表。
        """
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
            raise RagIndexError('RagGateway 未配置文档装载器，无法处理 source_path。')

        try:
            documents = self._document_loader.load(source_path)
        except RagIndexError:
            raise
        except Exception as exc:
            raise RagIndexError(f'文档加载失败: {exc}') from exc

        if not documents:
            raise ValueError('RagIndexRequest.documents 不能为空，至少需要包含一篇文档。')
        return list(documents)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """调用嵌入提供者执行批量文本向量化。

        Args:
            texts (list[str]): 待嵌入的文本列表。
        Returns:
            list[list[float]]: 与 texts 等长的嵌入向量列表。
        Raises:
            RagIndexError: 嵌入服务调用失败时抛出。
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        try:
            # 当前先采用固定条数分批，优先降低峰值内存与第三方嵌入 API 的限频风险。
            # 若后续接入更大规模语料，建议演进为基于 token 或字符预算的自适应批处理。
            for start in range(0, len(texts), self._EMBED_BATCH_SIZE):
                batch = texts[start:start + self._EMBED_BATCH_SIZE]
                batch_vectors = self._embedding_provider.embed_texts(batch)
                if len(batch_vectors) != len(batch):
                    raise RagIndexError(
                        "嵌入提供者返回数量异常："
                        f"期望 {len(batch)} 条向量，实际收到 {len(batch_vectors)} 条。"
                    )
                vectors.extend(batch_vectors)
            return vectors
        except RagIndexError:
            raise
        except Exception as exc:
            raise RagIndexError(f"嵌入向量生成失败: {exc}") from exc

    def _search_collection(
        self,
        query: str,
        collection_name: str,
        top_k: int,
    ) -> list[RagRetrievedChunk]:
        """嵌入查询文本并在指定集合中检索最相似分块。

        Args:
            query (str): 用户的查询文本。
            collection_name (str): 目标向量集合名称。
            top_k (int): 返回的最大分块数量。
        Returns:
            list[RagRetrievedChunk]: 按余弦相似度降序排列的检索命中列表。
        Raises:
            RagRetrieveError: 嵌入查询失败时抛出。
            RagCollectionNotFoundError: 集合不存在时透传抛出。
        """
        try:
            query_vectors = self._embed_texts([query])
        except RagIndexError as exc:
            raise RagRetrieveError(f"查询嵌入失败: {exc}") from exc

        try:
            return self._vector_store.search(
                name=collection_name,
                query_vector=query_vectors[0],
                top_k=top_k,
            )
        except RagCollectionNotFoundError:
            raise
        except Exception as exc:
            raise RagRetrieveError(f"向量检索失败: {exc}") from exc
