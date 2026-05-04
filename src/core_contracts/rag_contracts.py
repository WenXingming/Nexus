"""RAG（检索增强生成）领域契约。

集中定义 RAG 模块的全部跨域数据契约，保证任何外部调用者
与 RAG 模块的交互**只依赖本文件中的纯数据类与协议**：

  - 嵌入提供者协议   (EmbeddingProvider)
  - 模型客户端协议   (RagModelClient)
  - 文档 / 分块 DTO  (RagDocument, RagChunk, RagRetrievedChunk)
  - 索引请求 / 结果  (RagIndexRequest, RagIndexResult)
  - 检索请求 / 结果  (RagRetrieveRequest, RagRetrieveResult)
  - 查询请求 / 结果  (RagQueryRequest, RagQueryResult)
  - 领域专属异常层次 (RagError → RagIndexError / RagRetrieveError /
                       RagQueryError / RagCollectionNotFoundError)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.core_contracts.client_contracts import LlmRequest, LlmResult

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonDict = dict[str, JsonValue]


# ── 私有强制类型转换辅助函数 ──────────────────────────────────────────────────

def _as_str(v: object, default: str = '') -> str:
    """将任意值安全转换为字符串；None 时返回 default。"""
    return str(v) if v is not None else default


def _as_int(v: object, default: int = 0) -> int:
    """将任意值安全转换为整数；失败时返回 default。"""
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_float(v: object, default: float = 0.0) -> float:
    """将任意值安全转换为浮点数；失败时返回 default。"""
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_dict(v: object) -> dict:  # type: ignore[type-arg]
    """将任意值安全转换为字典；非字典时返回空字典。"""
    return dict(v) if isinstance(v, dict) else {}  # type: ignore[arg-type]


# ── 领域专属异常 ─────────────────────────────────────────────────────────────

class RagError(RuntimeError):
    """RAG 领域所有异常的统一基类。"""


class RagIndexError(RagError):
    """文档索引流程（分块、嵌入、写入向量库）中发生的错误。"""


class RagRetrieveError(RagError):
    """向量相似度检索流程中发生的错误。"""


class RagQueryError(RagError):
    """检索增强生成（检索 + 模型调用）流程中发生的错误。"""


class RagCollectionNotFoundError(RagError):
    """请求的向量集合在存储中不存在。

    Attributes:
        collection_name (str): 未找到的集合名称。
    """

    def __init__(self, collection_name: str) -> None:
        """初始化集合未找到异常。

        Args:
            collection_name (str): 触发异常的集合名称。
        """
        super().__init__(f"RAG 向量集合不存在: {collection_name!r}")
        self.collection_name = collection_name  # str：未找到的集合名称。


# ── 嵌入提供者协议 ──────────────────────────────────────────────────────────

@runtime_checkable
class EmbeddingProvider(Protocol):
    """文本嵌入向量提供者协议。

    任何满足此协议的对象均可作为嵌入能力注入 RagGateway；
    可以是本地模型、OpenAI Embeddings API 的适配器等。
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """将文本批量转换为等长的嵌入向量列表。

        Args:
            texts (list[str]): 待嵌入的文本批次，不可为空。
        Returns:
            list[list[float]]: 与 texts 等长的嵌入向量列表；
                               同一批次内所有向量的维度必须相同。
        Raises:
            RagIndexError: 嵌入服务调用失败或返回异常数据时抛出。
        """
        ...  # Protocol stub — 具体实现由外部注入。


# ── 模型客户端协议 ──────────────────────────────────────────────────────────

@runtime_checkable
class RagModelClient(Protocol):
    """RAG 模块所需的最小模型调用协议。

    ClientGateway 天然满足此协议，可直接注入；
    测试时可用 Mock 对象替代，无需引入真实 LLM 客户端。
    """

    def chat(self, request: 'LlmRequest') -> 'LlmResult':
        """执行一次非流式聊天补全调用并返回完整结果。

        Args:
            request (LlmRequest): 标准化的聊天补全请求契约。
        Returns:
            LlmResult: 包含生成文本、用量及完成原因的标准化结果。
        Raises:
            RuntimeError: 网络连接失败或其他底层客户端异常。
        """
        ...  # Protocol stub — 具体实现由外部注入。


@runtime_checkable
class RagDocumentLoader(Protocol):
    """RAG 文档源加载协议。

    任何满足此协议的对象均可作为文档装载器注入 RagGateway；
    例如文件系统 DocumentLoader 或后续的对象存储装载器。
    """

    def load(self, source: str) -> list['RagDocument']:
        """从给定 source 载入一组可索引文档。"""
        ...


# ── 文档与分块 DTO ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RagDocument:
    """表示一个待索引的原始文档。

    文档是索引操作的最小输入单元；引擎会将其切分为多个 RagChunk
    后才进行嵌入与向量存储。
    """

    doc_id: str                                       # str：文档唯一标识符，由调用方保证全局唯一。
    content: str                                      # str：文档正文全文。
    metadata: JsonDict = field(default_factory=dict)  # JsonDict：文档附加元数据（来源路径、标题等）。

    def to_dict(self) -> JsonDict:
        """把文档对象序列化为可 JSON 化的字典。

        Returns:
            JsonDict: 包含 doc_id、content、metadata 的字典。
        """
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'metadata': dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: JsonDict) -> 'RagDocument':
        """从字典恢复文档对象。

        Args:
            payload (JsonDict): 原始字典，同时兼容 camelCase 键名。
        Returns:
            RagDocument: 恢复后的文档对象。
        """
        data = _as_dict(payload)
        return cls(
            doc_id=_as_str(data.get('doc_id', data.get('docId')), ''),
            content=_as_str(data.get('content'), ''),
            metadata=_as_dict(data.get('metadata')),
        )


@dataclass(frozen=True)
class RagChunk:
    """文档被滑动窗口切分后的单个文本分块，是向量索引的最小单元。"""

    chunk_id: str                                     # str：分块唯一标识符，格式约定为 "{doc_id}#{position}"。
    doc_id: str                                       # str：所属文档的唯一标识符，与 RagDocument.doc_id 对应。
    content: str                                      # str：分块的文本内容。
    position: int                                     # int：该分块在原文档中的序号（0-based）。
    metadata: JsonDict = field(default_factory=dict)  # JsonDict：继承自父文档的元数据。

    def to_dict(self) -> JsonDict:
        """把分块对象序列化为字典。

        Returns:
            JsonDict: 包含 chunk_id、doc_id、content、position、metadata 的字典。
        """
        return {
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'content': self.content,
            'position': self.position,
            'metadata': dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: JsonDict) -> 'RagChunk':
        """从字典恢复分块对象。

        Args:
            payload (JsonDict): 原始字典，兼容 camelCase 键名。
        Returns:
            RagChunk: 恢复后的分块对象。
        """
        data = _as_dict(payload)
        return cls(
            chunk_id=_as_str(data.get('chunk_id', data.get('chunkId')), ''),
            doc_id=_as_str(data.get('doc_id', data.get('docId')), ''),
            content=_as_str(data.get('content'), ''),
            position=_as_int(data.get('position'), 0),
            metadata=_as_dict(data.get('metadata')),
        )


@dataclass(frozen=True)
class RagRetrievedChunk:
    """一次检索中命中的分块，附带与查询向量的余弦相似度得分。"""

    chunk: RagChunk  # RagChunk：命中的原始分块对象。
    score: float     # float：与查询向量的余弦相似度，范围 [0.0, 1.0]，越高越相关。

    def to_dict(self) -> JsonDict:
        """把检索命中分块序列化为字典。

        Returns:
            JsonDict: 包含 chunk（嵌套字典）和 score 的字典。
        """
        return {
            'chunk': self.chunk.to_dict(),
            'score': self.score,
        }

    @classmethod
    def from_dict(cls, payload: JsonDict) -> 'RagRetrievedChunk':
        """从字典恢复检索命中分块。

        Args:
            payload (JsonDict): 原始字典。
        Returns:
            RagRetrievedChunk: 恢复后的检索命中分块对象。
        """
        data = _as_dict(payload)
        return cls(
            chunk=RagChunk.from_dict(_as_dict(data.get('chunk'))),
            score=_as_float(data.get('score'), 0.0),
        )


# ── 索引请求 / 结果 ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RagIndexRequest:
    """向 RAG 模块提交文档索引任务的标准请求契约。

    调用方可以直接传入 documents，或传入 source_path 由 RagGateway
    在内部装载文档后完成索引；当两者同时提供时，documents 优先。
    """

    documents: tuple[RagDocument, ...] = ()  # tuple[RagDocument, ...]：待索引的文档列表；为空时可改用 source_path。
    source_path: str | None = None           # str | None：待装载的文件或目录路径；由 RagGateway 内部调用 document_loader 处理。
    collection_name: str = 'default'         # str：目标向量集合名称；不存在时自动创建，已存在时追加覆盖。
    chunk_size: int = 512                    # int：每个分块的最大字符数（基于 Unicode 字符计数）。
    chunk_overlap: int = 64                  # int：相邻分块间的重叠字符数，用于保留跨块上下文。

    def to_dict(self) -> JsonDict:
        """把索引请求序列化为字典。

        Returns:
            JsonDict: 包含 documents、collection_name、chunk_size、chunk_overlap 的字典。
        """
        return {
            'documents': [doc.to_dict() for doc in self.documents],
            'source_path': self.source_path,
            'collection_name': self.collection_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
        }

    @classmethod
    def from_dict(cls, payload: JsonDict) -> 'RagIndexRequest':
        """从字典恢复索引请求。

        Args:
            payload (JsonDict): 原始字典，兼容 camelCase 键名。
        Returns:
            RagIndexRequest: 恢复后的索引请求对象。
        """
        data = _as_dict(payload)
        docs_raw = data.get('documents', [])
        if not isinstance(docs_raw, list):
            docs_raw = []
        return cls(
            documents=tuple(RagDocument.from_dict(d) for d in docs_raw if isinstance(d, dict)),
            source_path=_as_str(data.get('source_path', data.get('sourcePath'))) or None,
            collection_name=_as_str(data.get('collection_name', data.get('collectionName')), 'default'),
            chunk_size=_as_int(data.get('chunk_size', data.get('chunkSize')), 512),
            chunk_overlap=_as_int(data.get('chunk_overlap', data.get('chunkOverlap')), 64),
        )


@dataclass(frozen=True)
class RagIndexResult:
    """文档索引操作完成后的标准结果契约。"""

    collection_name: str  # str：被写入的目标集合名称。
    docs_indexed: int     # int：本次成功处理的文档数量。
    chunks_created: int   # int：本次生成并嵌入到向量存储中的分块总数。
    duration_s: float     # float：从请求到写入完成的总耗时（秒）。

    def to_dict(self) -> JsonDict:
        """把索引结果序列化为字典。

        Returns:
            JsonDict: 包含 collection_name、docs_indexed、chunks_created、duration_s 的字典。
        """
        return {
            'collection_name': self.collection_name,
            'docs_indexed': self.docs_indexed,
            'chunks_created': self.chunks_created,
            'duration_s': self.duration_s,
        }

    @classmethod
    def from_dict(cls, payload: JsonDict) -> 'RagIndexResult':
        """从字典恢复索引结果。

        Args:
            payload (JsonDict): 原始字典，兼容 camelCase 键名。
        Returns:
            RagIndexResult: 恢复后的索引结果对象。
        """
        data = _as_dict(payload)
        return cls(
            collection_name=_as_str(data.get('collection_name', data.get('collectionName')), ''),
            docs_indexed=_as_int(data.get('docs_indexed', data.get('docsIndexed')), 0),
            chunks_created=_as_int(data.get('chunks_created', data.get('chunksCreated')), 0),
            duration_s=_as_float(data.get('duration_s', data.get('durationS')), 0.0),
        )


# ── 检索请求 / 结果（纯向量检索，不含生成）──────────────────────────────────

@dataclass(frozen=True)
class RagRetrieveRequest:
    """向 RAG 模块提交纯向量相似度检索（不触发模型生成）的标准请求契约。

    适用于只需要检索相关分块、由上层自行决定如何使用上下文的场景。
    """

    query: str                         # str：用于相似度检索的查询文本，引擎会将其嵌入后与向量库对比。
    collection_name: str = 'default'   # str：目标向量集合名称，须已通过 index_documents 建立。
    top_k: int = 5                     # int：返回相似度最高的分块数量上限（实际数量不超过集合大小）。

    def to_dict(self) -> JsonDict:
        """把检索请求序列化为字典。

        Returns:
            JsonDict: 包含 query、collection_name、top_k 的字典。
        """
        return {
            'query': self.query,
            'collection_name': self.collection_name,
            'top_k': self.top_k,
        }

    @classmethod
    def from_dict(cls, payload: JsonDict) -> 'RagRetrieveRequest':
        """从字典恢复检索请求。

        Args:
            payload (JsonDict): 原始字典，兼容 camelCase 键名。
        Returns:
            RagRetrieveRequest: 恢复后的检索请求对象。
        """
        data = _as_dict(payload)
        return cls(
            query=_as_str(data.get('query'), ''),
            collection_name=_as_str(data.get('collection_name', data.get('collectionName')), 'default'),
            top_k=_as_int(data.get('top_k', data.get('topK')), 5),
        )


@dataclass(frozen=True)
class RagRetrieveResult:
    """纯向量检索操作完成后的标准结果契约。"""

    query: str                                       # str：原始查询文本（回显）。
    collection_name: str                             # str：被查询的目标集合名称。
    retrieved_chunks: tuple[RagRetrievedChunk, ...]  # tuple[RagRetrievedChunk, ...]：按相似度降序排列的命中分块列表。
    duration_s: float                                # float：检索操作总耗时（秒）。

    def to_dict(self) -> JsonDict:
        """把检索结果序列化为字典。

        Returns:
            JsonDict: 包含 query、collection_name、retrieved_chunks、duration_s 的字典。
        """
        return {
            'query': self.query,
            'collection_name': self.collection_name,
            'retrieved_chunks': [c.to_dict() for c in self.retrieved_chunks],
            'duration_s': self.duration_s,
        }

    @classmethod
    def from_dict(cls, payload: JsonDict) -> 'RagRetrieveResult':
        """从字典恢复检索结果。

        Args:
            payload (JsonDict): 原始字典，兼容 camelCase 键名。
        Returns:
            RagRetrieveResult: 恢复后的检索结果对象。
        """
        data = _as_dict(payload)
        raw_chunks = data.get('retrieved_chunks', data.get('retrievedChunks', []))
        if not isinstance(raw_chunks, list):
            raw_chunks = []
        return cls(
            query=_as_str(data.get('query'), ''),
            collection_name=_as_str(data.get('collection_name', data.get('collectionName')), ''),
            retrieved_chunks=tuple(
                RagRetrievedChunk.from_dict(c) for c in raw_chunks if isinstance(c, dict)
            ),
            duration_s=_as_float(data.get('duration_s', data.get('durationS')), 0.0),
        )


# ── 查询请求 / 结果（RAG = 检索 + 生成）────────────────────────────────────

@dataclass(frozen=True)
class RagQueryRequest:
    """向 RAG 模块提交检索增强生成（RAG）问答的标准请求契约。

    引擎将执行完整 RAG 流水线：
    1. 将 query 嵌入并在向量库中检索 top_k 个相关分块；
    2. 将检索到的分块拼装为上下文，连同 query 构建提示词；
    3. 调用注入的模型客户端生成自然语言回答。
    """

    query: str                                  # str：用户提问的自然语言文本。
    collection_name: str = 'default'            # str：用于检索的目标向量集合名称。
    top_k: int = 5                              # int：检索阶段取回的分块数量上限。
    answer_max_tokens: int = 1024               # int：期望生成回答的最大 token 数（提示词中约束，受模型能力限制）。
    system_prompt_override: str | None = None   # str | None：覆盖默认系统提示词；为 None 时使用内置 RAG 模板。

    def to_dict(self) -> JsonDict:
        """把查询请求序列化为字典。

        Returns:
            JsonDict: 包含 query、collection_name、top_k、answer_max_tokens、system_prompt_override 的字典。
        """
        return {
            'query': self.query,
            'collection_name': self.collection_name,
            'top_k': self.top_k,
            'answer_max_tokens': self.answer_max_tokens,
            'system_prompt_override': self.system_prompt_override,
        }

    @classmethod
    def from_dict(cls, payload: JsonDict) -> 'RagQueryRequest':
        """从字典恢复查询请求。

        Args:
            payload (JsonDict): 原始字典，兼容 camelCase 键名。
        Returns:
            RagQueryRequest: 恢复后的查询请求对象。
        """
        data = _as_dict(payload)
        override_raw = data.get('system_prompt_override', data.get('systemPromptOverride'))
        return cls(
            query=_as_str(data.get('query'), ''),
            collection_name=_as_str(data.get('collection_name', data.get('collectionName')), 'default'),
            top_k=_as_int(data.get('top_k', data.get('topK')), 5),
            answer_max_tokens=_as_int(
                data.get('answer_max_tokens', data.get('answerMaxTokens')), 1024
            ),
            system_prompt_override=_as_str(override_raw) if override_raw is not None else None,
        )


@dataclass(frozen=True)
class RagQueryResult:
    """完整 RAG 问答流水线完成后的标准结果契约。"""

    query: str                                       # str：原始用户问题（回显）。
    collection_name: str                             # str：被查询的目标集合名称。
    retrieved_chunks: tuple[RagRetrievedChunk, ...]  # tuple[RagRetrievedChunk, ...]：检索阶段命中分块（按相似度降序）。
    answer: str                                      # str：模型基于检索上下文生成的回答文本。
    prompt_tokens: int                               # int：本次调用消耗的提示词 token 数。
    completion_tokens: int                           # int：本次调用生成回答消耗的 token 数。
    duration_s: float                                # float：完整流水线（检索 + 生成）的总耗时（秒）。

    def to_dict(self) -> JsonDict:
        """把查询结果序列化为字典。

        Returns:
            JsonDict: 包含所有字段的字典。
        """
        return {
            'query': self.query,
            'collection_name': self.collection_name,
            'retrieved_chunks': [c.to_dict() for c in self.retrieved_chunks],
            'answer': self.answer,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'duration_s': self.duration_s,
        }

    @classmethod
    def from_dict(cls, payload: JsonDict) -> 'RagQueryResult':
        """从字典恢复查询结果。

        Args:
            payload (JsonDict): 原始字典，兼容 camelCase 键名。
        Returns:
            RagQueryResult: 恢复后的查询结果对象。
        """
        data = _as_dict(payload)
        raw_chunks = data.get('retrieved_chunks', data.get('retrievedChunks', []))
        if not isinstance(raw_chunks, list):
            raw_chunks = []
        return cls(
            query=_as_str(data.get('query'), ''),
            collection_name=_as_str(data.get('collection_name', data.get('collectionName')), ''),
            retrieved_chunks=tuple(
                RagRetrievedChunk.from_dict(c) for c in raw_chunks if isinstance(c, dict)
            ),
            answer=_as_str(data.get('answer'), ''),
            prompt_tokens=_as_int(data.get('prompt_tokens', data.get('promptTokens')), 0),
            completion_tokens=_as_int(data.get('completion_tokens', data.get('completionTokens')), 0),
            duration_s=_as_float(data.get('duration_s', data.get('durationS')), 0.0),
        )
