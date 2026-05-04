"""基于余弦相似度的纯 Python 内存向量存储（RAG 模块内部实现）。

职责单一：管理命名集合的嵌入向量，提供写入与线性检索能力。
不依赖任何 I/O 或模型调用，可独立单元测试。
"""

from __future__ import annotations

import heapq
import math

from src.core_contracts.rag_contracts import (
    RagEmbedding,
    RagError,
    RagRetrievedChunk,
)


class VectorStore:
    """基于余弦相似度的纯 Python 内存向量存储。

    内部以命名集合（collection）为单位组织数据，每个集合存储
    一组 RagEmbedding。检索时对集合内所有向量执行线性扫描，
    通过堆排序返回 top-k 最相似分块。
    """

    def __init__(self) -> None:
        """初始化空向量存储，不预分配任何集合。"""
        # dict[str, list[RagEmbedding]]
        self._store: dict[str, list[RagEmbedding]] = {}

    # ── 公有接口 ────────────────────────────────────────────────────────────

    def upsert(self, name: str, embeddings: list[RagEmbedding]) -> None:
        """将嵌入向量写入（或更新）指定集合。

        Args:
            name (str): 目标集合名称；不存在时自动创建。
            embeddings (list[RagEmbedding]): 待写入的嵌入向量列表。
        Raises:
            RagError: 同批或跨批向量维度不一致时抛出。
        """
        if name not in self._store:
            self._store[name] = []

        existing = self._store[name]
        self._validate_upsert_dimensions(name, existing, embeddings)

        positions_by_chunk_id = {
            emb.chunk.chunk_id: index for index, emb in enumerate(existing)
        }
        for emb in embeddings:
            existing_index = positions_by_chunk_id.get(emb.chunk.chunk_id)
            if existing_index is None:
                existing.append(emb)
                positions_by_chunk_id[emb.chunk.chunk_id] = len(existing) - 1
            else:
                existing[existing_index] = emb

    def search(
        self,
        name: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[RagRetrievedChunk]:
        """在指定集合中检索与查询向量最相似的分块（线性扫描）。

        Args:
            name (str): 目标集合名称，须已通过 upsert 创建。
            query_vector (list[float]): 查询嵌入向量。
            top_k (int): 返回相似度最高的分块数量上限。
        Returns:
            list[RagRetrievedChunk]: 按余弦相似度降序排列的检索命中分块列表。
        Raises:
            RagError: 指定集合不存在时抛出。
        """
        if name not in self._store:
            raise RagError(name)
        embeddings = self._store[name]
        if not embeddings or top_k <= 0:
            return []

        scored = (
            RagRetrievedChunk(
                chunk=emb.chunk,
                score=self._cosine_similarity(query_vector, emb.vector),
            )
            for emb in embeddings
        )
        return heapq.nlargest(top_k, scored, key=lambda item: item.score)

    def drop(self, name: str) -> None:
        """删除指定集合及其全部数据。

        Args:
            name (str): 要删除的集合名称。
        Raises:
            RagError: 集合不存在时抛出。
        """
        if name not in self._store:
            raise RagError(name)
        del self._store[name]

    def list_names(self) -> list[str]:
        """返回当前存储中所有集合的名称列表。

        Returns:
            list[str]: 所有集合名称列表；尚未建立任何集合时返回空列表。
        """
        return list(self._store.keys())

    # ── 私有辅助方法 ────────────────────────────────────────────────────────

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _validate_upsert_dimensions(
        self,
        name: str,
        existing: list[RagEmbedding],
        incoming: list[RagEmbedding],
    ) -> None:
        """校验同批及与现有集合间的向量维度一致性。"""
        if not incoming:
            return

        expected_dim = len(incoming[0].vector)
        for emb in incoming[1:]:
            if len(emb.vector) != expected_dim:
                raise RagError(
                    "同一批写入的向量维度不一致："
                    f"期望维度 {expected_dim}，实际收到 {len(emb.vector)}。"
                )

        if not existing:
            return

        existing_dim = len(existing[0].vector)
        if expected_dim != existing_dim:
            raise RagError(
                f"集合 {name!r} 的向量维度为 {existing_dim}，"
                f"本次写入维度为 {expected_dim}。"
            )
