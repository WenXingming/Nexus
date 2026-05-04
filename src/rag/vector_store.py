"""基于余弦相似度的纯 Python 内存向量存储（RAG 模块内部实现）。

职责单一：管理命名集合的分块与嵌入向量，提供写入与线性检索能力。
不依赖任何 I/O 或模型调用，可独立单元测试。
"""

from __future__ import annotations

import heapq
import math

from src.core_contracts.rag_contracts import (
    RagChunk,
    RagCollectionNotFoundError,
    RagIndexError,
    RagRetrievedChunk,
)


class VectorStore:
    """基于余弦相似度的纯 Python 内存向量存储。

    内部以命名集合（collection）为单位组织数据，每个集合存储
    一组 (RagChunk, embedding_vector) 配对。检索时对集合内所有
    向量执行线性扫描，通过堆排序返回 top-k 最相似分块。
    """

    def __init__(self) -> None:
        """初始化空向量存储，不预分配任何集合。"""
        # dict[str, tuple[list[RagChunk], list[list[float]]]]
        # 键为集合名称，值为 (分块列表, 嵌入向量列表) 二元组。
        self._store: dict[str, tuple[list[RagChunk], list[list[float]]]] = {}

    # ── 公有接口 ────────────────────────────────────────────────────────────

    def upsert(
        self,
        name: str,
        chunks: list[RagChunk],
        vectors: list[list[float]],
    ) -> None:
        """将分块及其嵌入向量写入（或更新）指定集合。

        Args:
            name (str): 目标集合名称；不存在时自动创建。
            chunks (list[RagChunk]): 待写入的分块列表。
            vectors (list[list[float]]): 与 chunks 一一对应的嵌入向量列表。
        Raises:
            RagIndexError: 数量不一致或向量维度非法时抛出。
        """
        if len(chunks) != len(vectors):
            raise RagIndexError(
                f"分块数量 ({len(chunks)}) 与向量数量 ({len(vectors)}) 不一致。"
            )
        if name not in self._store:
            self._store[name] = ([], [])

        existing_chunks, existing_vectors = self._store[name]
        self._validate_vector_dimensions(name, existing_vectors, vectors)

        positions_by_chunk_id = {
            chunk.chunk_id: index for index, chunk in enumerate(existing_chunks)
        }
        for chunk, vector in zip(chunks, vectors):
            existing_index = positions_by_chunk_id.get(chunk.chunk_id)
            if existing_index is None:
                existing_chunks.append(chunk)
                existing_vectors.append(vector)
                positions_by_chunk_id[chunk.chunk_id] = len(existing_chunks) - 1
                continue

            existing_chunks[existing_index] = chunk
            existing_vectors[existing_index] = vector

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
            RagCollectionNotFoundError: 指定集合不存在时抛出。
        """
        if name not in self._store:
            raise RagCollectionNotFoundError(name)
        chunks, vectors = self._store[name]
        if not chunks or top_k <= 0:
            return []

        scored = (
            RagRetrievedChunk(chunk=chunk, score=self._cosine_similarity(query_vector, vec))
            for chunk, vec in zip(chunks, vectors)
        )
        return heapq.nlargest(top_k, scored, key=lambda item: item.score)

    def drop(self, name: str) -> None:
        """删除指定集合及其全部数据。

        Args:
            name (str): 要删除的集合名称。
        Raises:
            RagCollectionNotFoundError: 集合不存在时抛出。
        """
        if name not in self._store:
            raise RagCollectionNotFoundError(name)
        del self._store[name]

    def list_names(self) -> list[str]:
        """返回当前存储中所有集合的名称列表。

        Returns:
            list[str]: 所有集合名称列表；尚未建立任何集合时返回空列表。
        """
        return list(self._store.keys())

    # ── 私有辅助方法 ────────────────────────────────────────────────────────

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度。

        Args:
            a (list[float]): 第一个嵌入向量。
            b (list[float]): 第二个嵌入向量，必须与 a 等长。
        Returns:
            float: 余弦相似度，范围 [-1.0, 1.0]；任一向量为零向量时返回 0.0。
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _validate_vector_dimensions(
        self,
        name: str,
        existing_vectors: list[list[float]],
        new_vectors: list[list[float]],
    ) -> None:
        """校验同一批写入及现有集合中的向量维度一致性。

        Args:
            name (str): 目标集合名称。
            existing_vectors (list[list[float]]): 集合中已存在的向量列表。
            new_vectors (list[list[float]]): 本次待写入的向量列表。
        Returns:
            None: 仅执行校验，无返回值。
        Raises:
            RagIndexError: 同批向量维度不一致，或与现有集合维度冲突时抛出。
        """
        if not new_vectors:
            return

        expected_new_dim = len(new_vectors[0])
        for vector in new_vectors[1:]:
            if len(vector) != expected_new_dim:
                raise RagIndexError(
                    "同一批写入的向量维度不一致："
                    f"期望维度 {expected_new_dim}，实际收到 {len(vector)}。"
                )

        if not existing_vectors:
            return

        existing_dim = len(existing_vectors[0])
        if expected_new_dim != existing_dim:
            raise RagIndexError(
                f"集合 {name!r} 的向量维度为 {existing_dim}，"
                f"本次写入维度为 {expected_new_dim}。"
            )
