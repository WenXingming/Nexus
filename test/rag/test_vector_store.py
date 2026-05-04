"""VectorStore 单元测试。

验证集合管理（upsert/drop/list）、余弦相似度检索及边界行为。
所有测试为纯内存计算，无外部依赖。
"""

from __future__ import annotations

import math

import pytest

from src.core_contracts.rag_contracts import (
    RagChunk,
    RagCollectionNotFoundError,
    RagIndexError,
)
from src.rag.vector_store import VectorStore


# =============================================================================
# 测试夹具
# =============================================================================

@pytest.fixture
def store() -> VectorStore:
    return VectorStore()


def make_chunk(chunk_id: str = 'c1', doc_id: str = 'd1', content: str = 'text') -> RagChunk:
    return RagChunk(chunk_id=chunk_id, doc_id=doc_id, content=content, position=0)


def unit_vector(dim: int, hot_index: int) -> list[float]:
    """创建指定维度的单位向量（仅 hot_index 位为 1.0）。"""
    v = [0.0] * dim
    v[hot_index] = 1.0
    return v


# =============================================================================
# upsert
# =============================================================================

class TestUpsert:
    def test_creates_new_collection(self, store: VectorStore) -> None:
        chunk = make_chunk()
        store.upsert('col', [chunk], [[0.1, 0.2]])
        assert 'col' in store.list_names()

    def test_appends_to_existing_collection(self, store: VectorStore) -> None:
        c1 = make_chunk('c1')
        c2 = make_chunk('c2')
        store.upsert('col', [c1], [[1.0, 0.0]])
        store.upsert('col', [c2], [[0.0, 1.0]])
        result = store.search('col', [1.0, 0.0], top_k=10)
        assert len(result) == 2

    def test_raises_when_chunk_and_vector_counts_differ(self, store: VectorStore) -> None:
        with pytest.raises(RagIndexError, match='不一致'):
            store.upsert('col', [make_chunk()], [[1.0], [0.5]])

    def test_accepts_empty_lists(self, store: VectorStore) -> None:
        # 空列表不应报错，集合被创建（空）
        store.upsert('empty', [], [])
        assert 'empty' in store.list_names()

    def test_upsert_replaces_existing_chunk_with_same_chunk_id(self, store: VectorStore) -> None:
        original = make_chunk('c1', content='old')
        replacement = make_chunk('c1', content='new')
        store.upsert('col', [original], [[1.0, 0.0]])
        store.upsert('col', [replacement], [[0.0, 1.0]])

        result = store.search('col', [0.0, 1.0], top_k=10)
        assert len(result) == 1
        assert result[0].chunk.content == 'new'

    def test_upsert_same_batch_duplicate_chunk_id_keeps_last_value(self, store: VectorStore) -> None:
        first = make_chunk('c1', content='first')
        last = make_chunk('c1', content='last')

        store.upsert('col', [first, last], [[1.0, 0.0], [0.0, 1.0]])

        result = store.search('col', [0.0, 1.0], top_k=10)
        assert len(result) == 1
        assert result[0].chunk.content == 'last'

    def test_upsert_raises_when_batch_vector_dimensions_mismatch(self, store: VectorStore) -> None:
        with pytest.raises(RagIndexError, match='同一批写入的向量维度不一致'):
            store.upsert('col', [make_chunk('c1'), make_chunk('c2')], [[1.0], [1.0, 2.0]])

    def test_upsert_raises_when_vector_dimensions_conflict_with_existing_collection(
        self, store: VectorStore
    ) -> None:
        store.upsert('col', [make_chunk('c1')], [[1.0, 0.0]])

        with pytest.raises(RagIndexError, match='向量维度为 2'):
            store.upsert('col', [make_chunk('c2')], [[1.0, 0.0, 0.0]])


# =============================================================================
# search
# =============================================================================

class TestSearch:
    def test_raises_when_collection_not_found(self, store: VectorStore) -> None:
        with pytest.raises(RagCollectionNotFoundError):
            store.search('nonexistent', [1.0, 0.0], top_k=1)

    def test_returns_empty_list_for_empty_collection(self, store: VectorStore) -> None:
        store.upsert('empty', [], [])
        result = store.search('empty', [1.0, 0.0], top_k=5)
        assert result == []

    def test_returns_empty_list_when_top_k_is_zero(self, store: VectorStore) -> None:
        store.upsert('col', [make_chunk()], [[1.0, 0.0]])
        result = store.search('col', [1.0, 0.0], top_k=0)
        assert result == []

    def test_most_similar_chunk_returned_first(self, store: VectorStore) -> None:
        c_best = make_chunk('best', content='best match')
        c_weak = make_chunk('weak', content='weak match')
        # c_best 与查询向量 [1,0,0] 完全一致（相似度=1.0）
        store.upsert('col', [c_best, c_weak], [unit_vector(3, 0), unit_vector(3, 1)])
        result = store.search('col', unit_vector(3, 0), top_k=2)
        assert result[0].chunk.chunk_id == 'best'
        assert result[0].score == pytest.approx(1.0)

    def test_scores_are_cosine_similarity(self, store: VectorStore) -> None:
        chunk = make_chunk()
        # 两个相同向量，余弦相似度 = 1.0
        v = [3.0, 4.0]
        store.upsert('col', [chunk], [v])
        result = store.search('col', v, top_k=1)
        assert result[0].score == pytest.approx(1.0)

    def test_top_k_limits_results(self, store: VectorStore) -> None:
        chunks = [make_chunk(f'c{i}') for i in range(10)]
        vectors = [unit_vector(10, i) for i in range(10)]
        store.upsert('col', chunks, vectors)
        result = store.search('col', unit_vector(10, 0), top_k=3)
        assert len(result) == 3

    def test_results_are_sorted_descending_by_score(self, store: VectorStore) -> None:
        chunks = [make_chunk(f'c{i}') for i in range(5)]
        # 人工构造不同相似度
        query = [1.0, 0.0, 0.0]
        vectors = [
            [0.2, 0.8, 0.0],
            [0.9, 0.1, 0.0],
            [0.5, 0.5, 0.0],
            [0.1, 0.9, 0.0],
            [0.7, 0.3, 0.0],
        ]
        # 归一化
        def norm(v: list[float]) -> list[float]:
            n = math.sqrt(sum(x*x for x in v))
            return [x/n for x in v]
        store.upsert('col', chunks, [norm(v) for v in vectors])
        result = store.search('col', norm(query), top_k=5)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_zero_vector_returns_zero_score(self, store: VectorStore) -> None:
        chunk = make_chunk()
        store.upsert('col', [chunk], [[0.0, 0.0, 0.0]])
        result = store.search('col', [0.0, 0.0, 0.0], top_k=1)
        assert result[0].score == pytest.approx(0.0)


# =============================================================================
# drop
# =============================================================================

class TestDrop:
    def test_drops_existing_collection(self, store: VectorStore) -> None:
        store.upsert('col', [make_chunk()], [[1.0]])
        store.drop('col')
        assert 'col' not in store.list_names()

    def test_raises_when_dropping_nonexistent_collection(self, store: VectorStore) -> None:
        with pytest.raises(RagCollectionNotFoundError):
            store.drop('ghost')

    def test_dropped_collection_not_searchable(self, store: VectorStore) -> None:
        store.upsert('col', [make_chunk()], [[1.0]])
        store.drop('col')
        with pytest.raises(RagCollectionNotFoundError):
            store.search('col', [1.0], top_k=1)


# =============================================================================
# list_names
# =============================================================================

class TestListNames:
    def test_empty_store_returns_empty_list(self, store: VectorStore) -> None:
        assert store.list_names() == []

    def test_lists_all_collection_names(self, store: VectorStore) -> None:
        store.upsert('a', [make_chunk()], [[1.0]])
        store.upsert('b', [make_chunk()], [[1.0]])
        names = store.list_names()
        assert set(names) == {'a', 'b'}
