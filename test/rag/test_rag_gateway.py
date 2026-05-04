"""RagGateway 单元测试。

通过 Mock 严格隔离全部注入依赖（EmbeddingProvider、DocumentChunker、
VectorStore、AnswerGenerator），验证：
  - index_documents / retrieve / query / drop_collection / list_collections
    各接口的主流程、参数透传、结果契约封装；
  - 空输入快速失败（ValueError）；
  - 各环节异常的正确翻译与透传。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core_contracts.rag_contracts import (
    RagChunk,
    RagCollectionNotFoundError,
    RagDocument,
    RagIndexError,
    RagIndexRequest,
    RagQueryError,
    RagQueryRequest,
    RagRetrieveError,
    RagRetrieveRequest,
    RagRetrievedChunk,
)
from src.rag.rag_gateway import RagGateway


# =============================================================================
# 工具函数
# =============================================================================

def make_chunk(chunk_id: str = 'c1', doc_id: str = 'd1', content: str = 'content') -> RagChunk:
    return RagChunk(chunk_id=chunk_id, doc_id=doc_id, content=content, position=0)


def make_retrieved(chunk: RagChunk, score: float = 0.9) -> RagRetrievedChunk:
    return RagRetrievedChunk(chunk=chunk, score=score)


def make_doc(doc_id: str = 'd1', content: str = 'some document content') -> RagDocument:
    return RagDocument(doc_id=doc_id, content=content)


# =============================================================================
# 测试夹具
# =============================================================================

@pytest.fixture
def mock_embedding_provider() -> MagicMock:
    ep = MagicMock()
    ep.embed_texts.return_value = [[0.1, 0.2, 0.3]]
    return ep


@pytest.fixture
def mock_chunker() -> MagicMock:
    chunker = MagicMock()
    chunker.chunk.return_value = [make_chunk()]
    return chunker


@pytest.fixture
def mock_vector_store() -> MagicMock:
    vs = MagicMock()
    vs.search.return_value = [make_retrieved(make_chunk())]
    vs.list_names.return_value = []
    return vs


@pytest.fixture
def mock_answer_generator() -> MagicMock:
    ag = MagicMock()
    ag.generate.return_value = ('测试回答', 50, 20)
    return ag


@pytest.fixture
def mock_document_loader() -> MagicMock:
    loader = MagicMock()
    loader.load.return_value = [make_doc()]
    return loader


@pytest.fixture
def gateway(
    mock_embedding_provider: MagicMock,
    mock_chunker: MagicMock,
    mock_vector_store: MagicMock,
    mock_answer_generator: MagicMock,
    mock_document_loader: MagicMock,
) -> RagGateway:
    return RagGateway(
        embedding_provider=mock_embedding_provider,
        chunker=mock_chunker,
        vector_store=mock_vector_store,
        answer_generator=mock_answer_generator,
        document_loader=mock_document_loader,
    )


# =============================================================================
# index_documents
# =============================================================================

class TestIndexDocuments:
    def test_raises_value_error_for_empty_documents(self, gateway: RagGateway) -> None:
        request = RagIndexRequest(documents=())
        with pytest.raises(ValueError, match='不能为空'):
            gateway.index_documents(request)

    def test_loads_documents_from_source_path_when_documents_missing(
        self,
        gateway: RagGateway,
        mock_document_loader: MagicMock,
    ) -> None:
        request = RagIndexRequest(source_path='docs/')

        gateway.index_documents(request)

        mock_document_loader.load.assert_called_once_with('docs/')

    def test_calls_chunker_for_each_document(
        self, gateway: RagGateway, mock_chunker: MagicMock
    ) -> None:
        docs = (make_doc('d1'), make_doc('d2'))
        request = RagIndexRequest(documents=docs, chunk_size=100, chunk_overlap=10)
        gateway._embedding_provider.embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]
        gateway.index_documents(request)
        assert mock_chunker.chunk.call_count == 2

    def test_passes_chunk_size_and_overlap_to_chunker(
        self, gateway: RagGateway, mock_chunker: MagicMock
    ) -> None:
        doc = make_doc()
        request = RagIndexRequest(documents=(doc,), chunk_size=256, chunk_overlap=32)
        gateway.index_documents(request)
        _, kwargs = mock_chunker.chunk.call_args
        assert kwargs['chunk_size'] == 256
        assert kwargs['chunk_overlap'] == 32

    def test_calls_embed_texts_with_chunk_contents(
        self, gateway: RagGateway, mock_embedding_provider: MagicMock, mock_chunker: MagicMock
    ) -> None:
        chunk = make_chunk(content='hello world')
        mock_chunker.chunk.return_value = [chunk]
        gateway.index_documents(RagIndexRequest(documents=(make_doc(),)))
        mock_embedding_provider.embed_texts.assert_called_once_with(['hello world'])

    def test_calls_vector_store_upsert(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        request = RagIndexRequest(documents=(make_doc(),), collection_name='my-col')
        gateway.index_documents(request)
        mock_vector_store.upsert.assert_called_once()
        _, kwargs = mock_vector_store.upsert.call_args
        assert kwargs['name'] == 'my-col'

    def test_returns_correct_index_result(
        self, gateway: RagGateway, mock_chunker: MagicMock
    ) -> None:
        mock_chunker.chunk.return_value = [make_chunk('c1'), make_chunk('c2')]
        docs = (make_doc('d1'), make_doc('d2'))
        request = RagIndexRequest(documents=docs, collection_name='test-col')
        gateway._embedding_provider.embed_texts.return_value = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
        ]
        result = gateway.index_documents(request)
        assert result.collection_name == 'test-col'
        assert result.docs_indexed == 2
        assert result.chunks_created == 4  # 2 docs × 2 chunks
        assert result.duration_s >= 0.0

    def test_index_batches_embedding_requests_in_fixed_chunks(
        self, gateway: RagGateway, mock_chunker: MagicMock, mock_embedding_provider: MagicMock
    ) -> None:
        chunks = [make_chunk(f'c{i}', content=f'chunk-{i}') for i in range(257)]
        mock_chunker.chunk.return_value = chunks
        mock_embedding_provider.embed_texts.side_effect = [
            [[0.1, 0.2]] * 128,
            [[0.1, 0.2]] * 128,
            [[0.1, 0.2]],
        ]

        gateway.index_documents(RagIndexRequest(documents=(make_doc(),)))

        assert mock_embedding_provider.embed_texts.call_count == 3
        first_batch = mock_embedding_provider.embed_texts.call_args_list[0].args[0]
        second_batch = mock_embedding_provider.embed_texts.call_args_list[1].args[0]
        third_batch = mock_embedding_provider.embed_texts.call_args_list[2].args[0]
        assert len(first_batch) == 128
        assert len(second_batch) == 128
        assert len(third_batch) == 1

    def test_raises_when_embedding_provider_returns_wrong_vector_count(
        self, gateway: RagGateway, mock_embedding_provider: MagicMock
    ) -> None:
        mock_embedding_provider.embed_texts.return_value = []

        with pytest.raises(RagIndexError, match='返回数量异常'):
            gateway.index_documents(RagIndexRequest(documents=(make_doc(),)))

    def test_skips_upsert_when_all_docs_produce_no_chunks(
        self, gateway: RagGateway, mock_chunker: MagicMock, mock_vector_store: MagicMock
    ) -> None:
        mock_chunker.chunk.return_value = []
        gateway.index_documents(RagIndexRequest(documents=(make_doc(),)))
        mock_vector_store.upsert.assert_not_called()

    def test_wraps_unexpected_exception_as_rag_index_error(
        self, gateway: RagGateway, mock_chunker: MagicMock
    ) -> None:
        mock_chunker.chunk.side_effect = RuntimeError('unexpected')
        with pytest.raises(RagIndexError, match='意外失败'):
            gateway.index_documents(RagIndexRequest(documents=(make_doc(),)))

    def test_wraps_source_loading_failure_as_rag_index_error(
        self,
        gateway: RagGateway,
        mock_document_loader: MagicMock,
    ) -> None:
        mock_document_loader.load.side_effect = RuntimeError('missing source')

        with pytest.raises(RagIndexError, match='文档加载失败'):
            gateway.index_documents(RagIndexRequest(source_path='docs/'))

    def test_transparently_passes_rag_index_error(
        self, gateway: RagGateway, mock_embedding_provider: MagicMock
    ) -> None:
        mock_embedding_provider.embed_texts.side_effect = RagIndexError('embed failed')
        with pytest.raises(RagIndexError, match='embed failed'):
            gateway.index_documents(RagIndexRequest(documents=(make_doc(),)))


# =============================================================================
# retrieve
# =============================================================================

class TestRetrieve:
    def test_raises_value_error_for_empty_query(self, gateway: RagGateway) -> None:
        with pytest.raises(ValueError, match='不能为空'):
            gateway.retrieve(RagRetrieveRequest(query='   '))

    def test_returns_retrieve_result_contract(self, gateway: RagGateway) -> None:
        result = gateway.retrieve(RagRetrieveRequest(query='test query', collection_name='col'))
        assert result.query == 'test query'
        assert result.collection_name == 'col'
        assert len(result.retrieved_chunks) == 1
        assert result.duration_s >= 0.0

    def test_calls_embed_texts_with_query(
        self, gateway: RagGateway, mock_embedding_provider: MagicMock
    ) -> None:
        gateway.retrieve(RagRetrieveRequest(query='my query'))
        mock_embedding_provider.embed_texts.assert_called_once_with(['my query'])

    def test_calls_vector_store_search_with_correct_params(
        self, gateway: RagGateway, mock_vector_store: MagicMock, mock_embedding_provider: MagicMock
    ) -> None:
        query_vec = [0.5, 0.5, 0.0]
        mock_embedding_provider.embed_texts.return_value = [query_vec]
        gateway.retrieve(RagRetrieveRequest(query='q', collection_name='col', top_k=3))
        mock_vector_store.search.assert_called_once_with(
            name='col', query_vector=query_vec, top_k=3
        )

    def test_transparently_passes_collection_not_found_error(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.search.side_effect = RagCollectionNotFoundError('col')
        with pytest.raises(RagCollectionNotFoundError):
            gateway.retrieve(RagRetrieveRequest(query='q', collection_name='col'))

    def test_wraps_embed_failure_as_retrieve_error(
        self, gateway: RagGateway, mock_embedding_provider: MagicMock
    ) -> None:
        mock_embedding_provider.embed_texts.side_effect = Exception('embed down')
        with pytest.raises(RagRetrieveError, match='查询嵌入失败'):
            gateway.retrieve(RagRetrieveRequest(query='q'))

    def test_wraps_unexpected_search_exception_as_retrieve_error(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.search.side_effect = RuntimeError('search crashed')
        with pytest.raises(RagRetrieveError, match='向量检索失败'):
            gateway.retrieve(RagRetrieveRequest(query='q'))


# =============================================================================
# query
# =============================================================================

class TestQuery:
    def test_raises_value_error_for_empty_query(self, gateway: RagGateway) -> None:
        with pytest.raises(ValueError, match='不能为空'):
            gateway.query(RagQueryRequest(query=''))

    def test_returns_full_query_result_contract(self, gateway: RagGateway) -> None:
        result = gateway.query(RagQueryRequest(query='什么是 RAG？', collection_name='col'))
        assert result.query == '什么是 RAG？'
        assert result.collection_name == 'col'
        assert result.answer == '测试回答'
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 20
        assert result.duration_s >= 0.0
        assert len(result.retrieved_chunks) == 1

    def test_calls_answer_generator_with_correct_args(
        self, gateway: RagGateway, mock_answer_generator: MagicMock, mock_vector_store: MagicMock
    ) -> None:
        chunk = make_retrieved(make_chunk(content='context'))
        mock_vector_store.search.return_value = [chunk]
        gateway.query(RagQueryRequest(
            query='my question',
            collection_name='col',
            top_k=3,
            answer_max_tokens=512,
            system_prompt_override='custom system',
        ))
        mock_answer_generator.generate.assert_called_once_with(
            query='my question',
            chunks=[chunk],
            max_tokens=512,
            system_override='custom system',
        )

    def test_transparently_passes_collection_not_found_error(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.search.side_effect = RagCollectionNotFoundError('col')
        with pytest.raises(RagCollectionNotFoundError):
            gateway.query(RagQueryRequest(query='q', collection_name='col'))

    def test_transparently_passes_rag_query_error(
        self, gateway: RagGateway, mock_answer_generator: MagicMock
    ) -> None:
        mock_answer_generator.generate.side_effect = RagQueryError('gen failed')
        with pytest.raises(RagQueryError, match='gen failed'):
            gateway.query(RagQueryRequest(query='q'))

    def test_wraps_unexpected_exception_as_rag_query_error(
        self, gateway: RagGateway, mock_answer_generator: MagicMock
    ) -> None:
        mock_answer_generator.generate.side_effect = RuntimeError('boom')
        with pytest.raises(RagQueryError, match='意外失败'):
            gateway.query(RagQueryRequest(query='q'))


# =============================================================================
# drop_collection / list_collections
# =============================================================================

class TestCollectionManagement:
    def test_drop_collection_delegates_to_vector_store(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        gateway.drop_collection('my-col')
        mock_vector_store.drop.assert_called_once_with('my-col')

    def test_drop_collection_propagates_not_found_error(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.drop.side_effect = RagCollectionNotFoundError('ghost')
        with pytest.raises(RagCollectionNotFoundError):
            gateway.drop_collection('ghost')

    def test_list_collections_returns_vector_store_names(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.list_names.return_value = ['col-a', 'col-b']
        result = gateway.list_collections()
        assert result == ['col-a', 'col-b']
