"""DocumentChunker 单元测试。

验证滑动窗口切分策略、参数校验、分块元数据继承及边界行为。
所有测试为纯内存计算，无外部依赖。
"""

from __future__ import annotations

import pytest

from src.core_contracts.rag_contracts import RagDocument, RagError
from src.rag.document_chunker import DocumentChunker


# =============================================================================
# 测试夹具
# =============================================================================

@pytest.fixture
def chunker() -> DocumentChunker:
    return DocumentChunker()


@pytest.fixture
def simple_doc() -> RagDocument:
    return RagDocument(
        doc_id='doc-001',
        content='Hello world. This is a test document for chunking purposes.',
        metadata={'source': 'test'},
    )


# =============================================================================
# 参数校验
# =============================================================================

class TestChunkValidation:
    def test_raises_when_chunk_size_is_zero(self, chunker: DocumentChunker, simple_doc: RagDocument) -> None:
        with pytest.raises(RagError, match='chunk_size'):
            chunker.chunk(simple_doc, chunk_size=0, chunk_overlap=0)

    def test_raises_when_chunk_size_is_negative(self, chunker: DocumentChunker, simple_doc: RagDocument) -> None:
        with pytest.raises(RagError, match='chunk_size'):
            chunker.chunk(simple_doc, chunk_size=-1, chunk_overlap=0)

    def test_raises_when_overlap_equals_chunk_size(self, chunker: DocumentChunker, simple_doc: RagDocument) -> None:
        with pytest.raises(RagError, match='chunk_overlap'):
            chunker.chunk(simple_doc, chunk_size=10, chunk_overlap=10)

    def test_raises_when_overlap_exceeds_chunk_size(self, chunker: DocumentChunker, simple_doc: RagDocument) -> None:
        with pytest.raises(RagError, match='chunk_overlap'):
            chunker.chunk(simple_doc, chunk_size=10, chunk_overlap=15)


# =============================================================================
# 空内容
# =============================================================================

class TestEmptyContent:
    def test_empty_string_returns_empty_list(self, chunker: DocumentChunker) -> None:
        doc = RagDocument(doc_id='empty', content='')
        result = chunker.chunk(doc, chunk_size=100, chunk_overlap=0)
        assert result == []

    def test_whitespace_only_returns_empty_list(self, chunker: DocumentChunker) -> None:
        doc = RagDocument(doc_id='ws', content='   \n\t  ')
        result = chunker.chunk(doc, chunk_size=100, chunk_overlap=0)
        assert result == []


# =============================================================================
# 分块核心行为
# =============================================================================

class TestChunkBehavior:
    def test_single_chunk_when_content_fits(self, chunker: DocumentChunker) -> None:
        doc = RagDocument(doc_id='small', content='Short text.')
        result = chunker.chunk(doc, chunk_size=512, chunk_overlap=0)
        assert len(result) == 1
        assert result[0].content == 'Short text.'

    def test_multiple_chunks_for_long_content(self, chunker: DocumentChunker) -> None:
        content = 'A' * 200
        doc = RagDocument(doc_id='long', content=content)
        result = chunker.chunk(doc, chunk_size=50, chunk_overlap=0)
        assert len(result) > 1

    def test_chunk_ids_follow_convention(self, chunker: DocumentChunker, simple_doc: RagDocument) -> None:
        result = chunker.chunk(simple_doc, chunk_size=512, chunk_overlap=0)
        for idx, chunk in enumerate(result):
            assert chunk.chunk_id == f'doc-001#{idx}'

    def test_chunk_positions_are_sequential(self, chunker: DocumentChunker) -> None:
        doc = RagDocument(doc_id='seq', content='x' * 300)
        result = chunker.chunk(doc, chunk_size=50, chunk_overlap=0)
        positions = [c.position for c in result]
        assert positions == list(range(len(result)))

    def test_doc_id_propagated_to_all_chunks(self, chunker: DocumentChunker) -> None:
        doc = RagDocument(doc_id='my-doc', content='x' * 200)
        result = chunker.chunk(doc, chunk_size=50, chunk_overlap=0)
        assert all(c.doc_id == 'my-doc' for c in result)

    def test_metadata_inherited_by_all_chunks(self, chunker: DocumentChunker) -> None:
        meta = {'source': 'wiki', 'lang': 'zh'}
        doc = RagDocument(doc_id='meta-doc', content='x' * 200, metadata=meta)
        result = chunker.chunk(doc, chunk_size=50, chunk_overlap=0)
        assert all(c.metadata == meta for c in result)

    def test_metadata_is_copied_not_shared(self, chunker: DocumentChunker) -> None:
        """每个分块的 metadata 是独立副本，而非共享引用。"""
        meta: dict = {'key': 'val'}
        doc = RagDocument(doc_id='copy-test', content='x' * 200, metadata=meta)  # type: ignore[arg-type]
        result = chunker.chunk(doc, chunk_size=50, chunk_overlap=0)
        assert result[0].metadata is not result[1].metadata

    def test_overlap_produces_more_chunks_than_no_overlap(self, chunker: DocumentChunker) -> None:
        content = 'A' * 300
        doc = RagDocument(doc_id='overlap', content=content)
        no_overlap = chunker.chunk(doc, chunk_size=100, chunk_overlap=0)
        with_overlap = chunker.chunk(doc, chunk_size=100, chunk_overlap=50)
        assert len(with_overlap) >= len(no_overlap)

    def test_no_chunk_content_is_empty_string(self, chunker: DocumentChunker) -> None:
        doc = RagDocument(doc_id='nonempty', content='Hello world test.')
        result = chunker.chunk(doc, chunk_size=5, chunk_overlap=0)
        assert all(c.content != '' for c in result)


# =============================================================================
# 断点查找策略
# =============================================================================

class TestBreakPointStrategy:
    def test_prefers_newline_over_sentence_break(self, chunker: DocumentChunker) -> None:
        # 换行符在句末标点之前，应优先在换行处断开
        content = 'Hello\nWorld. More text here.'
        doc = RagDocument(doc_id='nl', content=content)
        # chunk_size=7 覆盖 'Hello\n'
        result = chunker.chunk(doc, chunk_size=7, chunk_overlap=0)
        assert result[0].content == 'Hello'

    def test_breaks_at_sentence_punctuation(self, chunker: DocumentChunker) -> None:
        content = 'First sentence. Second sentence. Third sentence.'
        doc = RagDocument(doc_id='sent', content=content)
        result = chunker.chunk(doc, chunk_size=20, chunk_overlap=0)
        # 至少切出多块
        assert len(result) > 1
        # 第一块应以句末标点结尾（或不含后续内容）
        assert result[0].content.endswith('.')

    def test_natural_break_does_not_skip_following_text(self, chunker: DocumentChunker) -> None:
        content = '123456789.abcdefghij'
        doc = RagDocument(doc_id='coverage', content=content)

        result = chunker.chunk(doc, chunk_size=12, chunk_overlap=0)

        assert ''.join(chunk.content for chunk in result) == content
