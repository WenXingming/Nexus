"""RagGateway 单元测试。

通过 Mock 严格隔离全部注入依赖（OpenAIEmbeddingProvider、DocumentChunker、
VectorStore、DocumentLoader），验证：
  - index / retrieve / retrieve_and_build_messages / drop_collection / list_collections
    各接口的主流程、参数透传、结果契约封装；
  - 空输入快速失败（ValueError）；
  - 各环节异常的正确翻译与透传。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core_contracts.model_contracts import Message
from src.core_contracts.rag_contracts import (
    RagChunk,
    RagDocument,
    RagEmbedding,
    RagError,
    RagIndexRequest,
    RagRetrieveRequest,
    RagRetrievedChunk,
)
from src.rag.rag_gateway import RagGateway


# =============================================================================
# 工具函数
# =============================================================================

def make_chunk(chunk_id: str = 'c1', doc_id: str = 'd1', content: str = 'content') -> RagChunk:
    return RagChunk(chunk_id=chunk_id, doc_id=doc_id, content=content, position=0)


def make_embedding(chunk: RagChunk, vector: list[float] | None = None) -> RagEmbedding:
    return RagEmbedding(chunk=chunk, vector=vector or [0.1, 0.2, 0.3])


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
    ep.embed_chunks.return_value = [make_embedding(make_chunk())]
    ep.embed_query.return_value = [0.1, 0.2, 0.3]
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
def mock_document_loader() -> MagicMock:
    loader = MagicMock()
    loader.load.return_value = [make_doc()]
    return loader


@pytest.fixture
def gateway(
    mock_embedding_provider: MagicMock,
    mock_chunker: MagicMock,
    mock_vector_store: MagicMock,
    mock_document_loader: MagicMock,
) -> RagGateway:
    return RagGateway(
        embedding_provider=mock_embedding_provider,
        chunker=mock_chunker,
        vector_store=mock_vector_store,
        document_loader=mock_document_loader,
    )


# =============================================================================
# index
# =============================================================================

class TestIndex:
    def test_raises_value_error_for_empty_documents(self, gateway: RagGateway) -> None:
        request = RagIndexRequest(documents=())
        with pytest.raises(ValueError, match='不能为空'):
            gateway.index(request)

    def test_loads_documents_from_source_path_when_documents_missing(
        self,
        gateway: RagGateway,
        mock_document_loader: MagicMock,
    ) -> None:
        request = RagIndexRequest(source_path='docs/')
        gateway.index(request)
        mock_document_loader.load.assert_called_once_with('docs/')

    def test_calls_chunker_for_each_document(
        self, gateway: RagGateway, mock_chunker: MagicMock
    ) -> None:
        docs = (make_doc('d1'), make_doc('d2'))
        request = RagIndexRequest(documents=docs, chunk_size=100, chunk_overlap=10)
        gateway.index(request)
        assert mock_chunker.chunk.call_count == 2

    def test_passes_chunk_size_and_overlap_to_chunker(
        self, gateway: RagGateway, mock_chunker: MagicMock
    ) -> None:
        doc = make_doc()
        request = RagIndexRequest(documents=(doc,), chunk_size=256, chunk_overlap=32)
        gateway.index(request)
        _, kwargs = mock_chunker.chunk.call_args
        assert kwargs['chunk_size'] == 256
        assert kwargs['chunk_overlap'] == 32

    def test_calls_embed_chunks_with_all_chunks(
        self, gateway: RagGateway,
        mock_chunker: MagicMock,
        mock_embedding_provider: MagicMock,
    ) -> None:
        c1 = make_chunk('c1', content='hello')
        c2 = make_chunk('c2', content='world')
        mock_chunker.chunk.return_value = [c1, c2]
        gateway.index(RagIndexRequest(documents=(make_doc(),)))
        mock_embedding_provider.embed_chunks.assert_called_once_with([c1, c2])

    def test_calls_vector_store_upsert_with_embeddings(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        request = RagIndexRequest(documents=(make_doc(),), collection_name='my-col')
        gateway.index(request)
        mock_vector_store.upsert.assert_called_once()
        _, kwargs = mock_vector_store.upsert.call_args
        assert kwargs['name'] == 'my-col'

    def test_returns_correct_index_result(
        self, gateway: RagGateway, mock_chunker: MagicMock
    ) -> None:
        mock_chunker.chunk.return_value = [make_chunk('c1'), make_chunk('c2')]
        docs = (make_doc('d1'), make_doc('d2'))
        request = RagIndexRequest(documents=docs, collection_name='test-col')
        gateway._embedding_provider.embed_chunks.return_value = [
            make_embedding(make_chunk('c1'), [0.1, 0.2]),
            make_embedding(make_chunk('c2'), [0.3, 0.4]),
            make_embedding(make_chunk('c3'), [0.5, 0.6]),
            make_embedding(make_chunk('c4'), [0.7, 0.8]),
        ]
        result = gateway.index(request)
        assert result.collection_name == 'test-col'
        assert result.docs_indexed == 2
        assert result.chunks_created == 4
        assert result.duration_s >= 0.0

    def test_skips_upsert_when_all_docs_produce_no_chunks(
        self, gateway: RagGateway, mock_chunker: MagicMock, mock_vector_store: MagicMock
    ) -> None:
        mock_chunker.chunk.return_value = []
        gateway.index(RagIndexRequest(documents=(make_doc(),)))
        mock_vector_store.upsert.assert_not_called()

    def test_wraps_unexpected_exception_as_rag_error(
        self, gateway: RagGateway, mock_chunker: MagicMock
    ) -> None:
        mock_chunker.chunk.side_effect = RuntimeError('unexpected')
        with pytest.raises(RagError, match='索引操作意外失败'):
            gateway.index(RagIndexRequest(documents=(make_doc(),)))

    def test_wraps_source_loading_failure_as_rag_error(
        self,
        gateway: RagGateway,
        mock_document_loader: MagicMock,
    ) -> None:
        mock_document_loader.load.side_effect = RuntimeError('missing source')
        with pytest.raises(RagError, match='文档加载失败'):
            gateway.index(RagIndexRequest(source_path='docs/'))

    def test_transparently_passes_rag_error(
        self, gateway: RagGateway, mock_embedding_provider: MagicMock
    ) -> None:
        mock_embedding_provider.embed_chunks.side_effect = RagError('embed failed')
        with pytest.raises(RagError, match='embed failed'):
            gateway.index(RagIndexRequest(documents=(make_doc(),)))


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

    def test_calls_embed_query_with_query(
        self, gateway: RagGateway, mock_embedding_provider: MagicMock
    ) -> None:
        gateway.retrieve(RagRetrieveRequest(query='my query'))
        mock_embedding_provider.embed_query.assert_called_once_with('my query')

    def test_calls_vector_store_search_with_correct_params(
        self, gateway: RagGateway, mock_vector_store: MagicMock, mock_embedding_provider: MagicMock
    ) -> None:
        mock_embedding_provider.embed_query.return_value = [0.5, 0.5, 0.0]
        gateway.retrieve(RagRetrieveRequest(query='q', collection_name='col', top_k=3))
        mock_vector_store.search.assert_called_once_with(
            name='col', query_vector=[0.5, 0.5, 0.0], top_k=3
        )

    def test_transparently_passes_rag_error_from_search(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.search.side_effect = RagError('col')
        with pytest.raises(RagError):
            gateway.retrieve(RagRetrieveRequest(query='q', collection_name='col'))

    def test_wraps_embed_failure_as_rag_error(
        self, gateway: RagGateway, mock_embedding_provider: MagicMock
    ) -> None:
        mock_embedding_provider.embed_query.side_effect = Exception('embed down')
        with pytest.raises(RagError, match='查询嵌入失败'):
            gateway.retrieve(RagRetrieveRequest(query='q'))

    def test_wraps_unexpected_search_exception_as_rag_error(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.search.side_effect = RuntimeError('search crashed')
        with pytest.raises(RagError, match='向量检索失败'):
            gateway.retrieve(RagRetrieveRequest(query='q'))


# =============================================================================
# retrieve_and_build_messages
# =============================================================================

class TestRetrieveAndBuildMessages:
    def test_returns_list_of_messages(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        chunk = make_chunk(content='检索增强生成的原理。')
        mock_vector_store.search.return_value = [make_retrieved(chunk)]
        messages = gateway.retrieve_and_build_messages(query='什么是 RAG', collection_name='col')
        assert len(messages) == 2
        assert all(isinstance(m, Message) for m in messages)

    def test_roles_are_system_and_user(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.search.return_value = [make_retrieved(make_chunk())]
        messages = gateway.retrieve_and_build_messages(query='q', collection_name='col')
        assert messages[0].role == 'system'
        assert messages[1].role == 'user'

    def test_user_message_contains_query(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.search.return_value = [make_retrieved(make_chunk())]
        messages = gateway.retrieve_and_build_messages(query='什么是 RAG', collection_name='col')
        assert '什么是 RAG' in messages[1].content

    def test_user_message_contains_retrieved_chunk_content(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        chunk = make_chunk(content='RAG 是检索增强生成技术。')
        mock_vector_store.search.return_value = [make_retrieved(chunk)]
        messages = gateway.retrieve_and_build_messages(query='q', collection_name='col')
        assert 'RAG 是检索增强生成技术。' in messages[1].content

    def test_empty_retrieval_shows_placeholder(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.search.return_value = []
        messages = gateway.retrieve_and_build_messages(query='q', collection_name='col')
        assert '未找到相关参考资料' in messages[1].content

    def test_uses_system_override(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.search.return_value = [make_retrieved(make_chunk())]
        messages = gateway.retrieve_and_build_messages(
            query='q', collection_name='col', system_override='自定义提示词',
        )
        assert messages[0].content == '自定义提示词'

    def test_uses_default_system_prompt_when_override_is_none(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.search.return_value = [make_retrieved(make_chunk())]
        messages = gateway.retrieve_and_build_messages(query='q', collection_name='col')
        assert '知识问答助手' in messages[0].content

    def test_includes_max_tokens_hint(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.search.return_value = [make_retrieved(make_chunk())]
        messages = gateway.retrieve_and_build_messages(
            query='q', collection_name='col', max_tokens=512,
        )
        assert '512 tokens' in messages[1].content

    def test_passes_top_k_to_retrieve(
        self, gateway: RagGateway, mock_vector_store: MagicMock,
        mock_embedding_provider: MagicMock,
    ) -> None:
        mock_embedding_provider.embed_query.return_value = [0.1, 0.2]
        mock_vector_store.search.return_value = [make_retrieved(make_chunk())]
        gateway.retrieve_and_build_messages(query='q', collection_name='col', top_k=10)
        mock_vector_store.search.assert_called_once_with(
            name='col', query_vector=[0.1, 0.2], top_k=10,
        )

    def test_raises_value_error_for_empty_query(self, gateway: RagGateway) -> None:
        with pytest.raises(ValueError, match='不能为空'):
            gateway.retrieve_and_build_messages(query='   ', collection_name='col')

    def test_wraps_retrieve_failure_as_rag_error(
        self, gateway: RagGateway, mock_embedding_provider: MagicMock
    ) -> None:
        mock_embedding_provider.embed_query.side_effect = Exception('down')
        with pytest.raises(RagError, match='查询嵌入失败'):
            gateway.retrieve_and_build_messages(query='q', collection_name='col')


# =============================================================================
# drop_collection / list_collections
# =============================================================================

class TestCollectionManagement:
    def test_drop_collection_delegates_to_vector_store(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        gateway.drop_collection('my-col')
        mock_vector_store.drop.assert_called_once_with('my-col')

    def test_drop_collection_propagates_rag_error(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.drop.side_effect = RagError('ghost')
        with pytest.raises(RagError):
            gateway.drop_collection('ghost')

    def test_list_collections_returns_vector_store_names(
        self, gateway: RagGateway, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.list_names.return_value = ['col-a', 'col-b']
        result = gateway.list_collections()
        assert result == ['col-a', 'col-b']
