"""RAG（检索增强生成）领域契约。

集中定义 RAG 模块的全部跨域数据契约与异常，保证任何外部调用者
与 RAG 模块的交互**只依赖本文件中的纯数据类**：

  - 文档 / 分块 DTO  (RagDocument, RagChunk, RagEmbedding, RagRetrievedChunk)
  - 索引请求 / 结果  (RagIndexRequest, RagIndexResult)
  - 检索请求 / 结果  (RagRetrieveRequest, RagRetrieveResult)
  - 领域异常         (RagError)
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── 领域异常 ─────────────────────────────────────────────────────────────────

class RagError(RuntimeError):
    """RAG 领域统一异常。"""


# ── 文档与分块 DTO ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RagDocument:
    """待索引的原始文档，是索引操作的最小输入单元。"""

    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RagChunk:
    """文档被滑动窗口切分后的单个文本分块，是向量索引的最小单元。"""

    chunk_id: str
    doc_id: str
    content: str
    position: int
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RagEmbedding:
    """RagChunk 与其嵌入向量的绑定契约，消除并行列表的运行时校验。"""

    chunk: RagChunk
    vector: list[float]


@dataclass(frozen=True)
class RagRetrievedChunk:
    """检索命中的分块，附带与查询向量的余弦相似度得分。"""

    chunk: RagChunk
    score: float


# ── 索引请求 / 结果 ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RagIndexRequest:
    """文档索引任务的标准请求契约。

    documents 与 source_path 至少提供一个；同时提供时 documents 优先。
    """

    documents: tuple[RagDocument, ...] = ()
    source_path: str | None = None
    collection_name: str = 'default'
    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass(frozen=True)
class RagIndexResult:
    """索引操作完成后的标准结果契约。"""

    collection_name: str
    docs_indexed: int
    chunks_created: int
    duration_s: float


# ── 检索请求 / 结果 ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RagRetrieveRequest:
    """纯向量相似度检索的标准请求契约（不触发模型生成）。"""

    query: str
    collection_name: str = 'default'
    top_k: int = 5


@dataclass(frozen=True)
class RagRetrieveResult:
    """纯向量检索操作完成后的标准结果契约。"""

    query: str
    collection_name: str
    retrieved_chunks: tuple[RagRetrievedChunk, ...]
    duration_s: float
