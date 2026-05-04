"""AnswerGenerator 单元测试。

通过 Mock 严格隔离 RagModelClient，验证提示词构建逻辑、
正常生成路径及异常翻译行为。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core_contracts.client_contracts import LlmRequest, LlmResult
from src.core_contracts.model_contracts import Message, TokenUsage
from src.core_contracts.rag_contracts import (
    RagChunk,
    RagQueryError,
    RagRetrievedChunk,
)
from src.rag.answer_generator import AnswerGenerator


# =============================================================================
# 测试夹具
# =============================================================================

@pytest.fixture
def mock_client() -> MagicMock:
    """返回满足 RagModelClient 协议的 Mock 对象，默认返回有效回答。"""
    client = MagicMock()
    client.chat.return_value = LlmResult(
        content='这是一个测试回答。',
        model='test-model',
        finish_reason='stop',
        usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
    )
    return client


@pytest.fixture
def generator(mock_client: MagicMock) -> AnswerGenerator:
    return AnswerGenerator(model_client=mock_client)


def make_retrieved_chunk(
    chunk_id: str = 'c1',
    doc_id: str = 'd1',
    content: str = '相关内容片段',
    score: float = 0.9,
) -> RagRetrievedChunk:
    chunk = RagChunk(chunk_id=chunk_id, doc_id=doc_id, content=content, position=0)
    return RagRetrievedChunk(chunk=chunk, score=score)


# =============================================================================
# 正常生成路径
# =============================================================================

class TestGenerateSuccess:
    def test_returns_answer_and_token_counts(
        self, generator: AnswerGenerator, mock_client: MagicMock
    ) -> None:
        chunks = [make_retrieved_chunk()]
        answer, prompt_tokens, completion_tokens = generator.generate(
            query='什么是 RAG？',
            chunks=chunks,
            max_tokens=512,
            system_override=None,
        )
        assert answer == '这是一个测试回答。'
        assert prompt_tokens == 50
        assert completion_tokens == 20

    def test_calls_model_client_chat_once(
        self, generator: AnswerGenerator, mock_client: MagicMock
    ) -> None:
        generator.generate(
            query='测试问题',
            chunks=[make_retrieved_chunk()],
            max_tokens=256,
            system_override=None,
        )
        mock_client.chat.assert_called_once()

    def test_passes_llm_request_to_chat(
        self, generator: AnswerGenerator, mock_client: MagicMock
    ) -> None:
        generator.generate(
            query='测试',
            chunks=[],
            max_tokens=128,
            system_override=None,
        )
        call_args = mock_client.chat.call_args
        request: LlmRequest = call_args[0][0] if call_args[0] else call_args[1]['request']
        assert isinstance(request, LlmRequest)
        assert request.max_tokens == 128

    def test_messages_contain_system_and_user(
        self, generator: AnswerGenerator, mock_client: MagicMock
    ) -> None:
        generator.generate(
            query='问题',
            chunks=[],
            max_tokens=256,
            system_override=None,
        )
        request: LlmRequest = mock_client.chat.call_args[0][0]
        roles = [m.role for m in request.messages]
        assert roles == ['system', 'user']

    def test_system_override_replaces_default_prompt(
        self, generator: AnswerGenerator, mock_client: MagicMock
    ) -> None:
        custom_system = '自定义系统提示词'
        generator.generate(
            query='问题',
            chunks=[],
            max_tokens=256,
            system_override=custom_system,
        )
        request: LlmRequest = mock_client.chat.call_args[0][0]
        system_msg = next(m for m in request.messages if m.role == 'system')
        assert system_msg.content == custom_system

    def test_default_system_prompt_used_when_override_is_none(
        self, generator: AnswerGenerator, mock_client: MagicMock
    ) -> None:
        generator.generate(
            query='问题',
            chunks=[],
            max_tokens=256,
            system_override=None,
        )
        request: LlmRequest = mock_client.chat.call_args[0][0]
        system_msg = next(m for m in request.messages if m.role == 'system')
        assert '知识问答助手' in system_msg.content  # type: ignore[operator]

    def test_user_message_contains_query(
        self, generator: AnswerGenerator, mock_client: MagicMock
    ) -> None:
        generator.generate(
            query='什么是向量数据库',
            chunks=[],
            max_tokens=256,
            system_override=None,
        )
        request: LlmRequest = mock_client.chat.call_args[0][0]
        user_msg = next(m for m in request.messages if m.role == 'user')
        assert '什么是向量数据库' in user_msg.content  # type: ignore[operator]

    def test_user_message_contains_chunk_content(
        self, generator: AnswerGenerator, mock_client: MagicMock
    ) -> None:
        chunks = [make_retrieved_chunk(content='向量检索的核心原理')]
        generator.generate(
            query='问题',
            chunks=chunks,
            max_tokens=256,
            system_override=None,
        )
        request: LlmRequest = mock_client.chat.call_args[0][0]
        user_msg = next(m for m in request.messages if m.role == 'user')
        assert '向量检索的核心原理' in user_msg.content  # type: ignore[operator]

    def test_empty_chunks_shows_placeholder_in_user_message(
        self, generator: AnswerGenerator, mock_client: MagicMock
    ) -> None:
        generator.generate(
            query='问题',
            chunks=[],
            max_tokens=256,
            system_override=None,
        )
        request: LlmRequest = mock_client.chat.call_args[0][0]
        user_msg = next(m for m in request.messages if m.role == 'user')
        assert '未找到相关参考资料' in user_msg.content  # type: ignore[operator]

    def test_strips_whitespace_from_answer(self, mock_client: MagicMock) -> None:
        mock_client.chat.return_value = LlmResult(
            content='  回答内容  \n',
            model='m',
            finish_reason='stop',
            usage=TokenUsage(10, 5, 15),
        )
        gen = AnswerGenerator(model_client=mock_client)
        answer, _, _ = gen.generate('q', [], 256, None)
        assert answer == '回答内容'


# =============================================================================
# 异常翻译
# =============================================================================

class TestGenerateErrors:
    def test_raises_rag_query_error_when_client_raises(
        self, mock_client: MagicMock
    ) -> None:
        mock_client.chat.side_effect = RuntimeError('network error')
        gen = AnswerGenerator(model_client=mock_client)
        with pytest.raises(RagQueryError, match='模型调用失败'):
            gen.generate('q', [], 256, None)

    def test_raises_rag_query_error_when_answer_is_empty(
        self, mock_client: MagicMock
    ) -> None:
        mock_client.chat.return_value = LlmResult(
            content='   ',
            model='m',
            finish_reason='stop',
            usage=TokenUsage(10, 0, 10),
        )
        gen = AnswerGenerator(model_client=mock_client)
        with pytest.raises(RagQueryError, match='空回答'):
            gen.generate('q', [], 256, None)

    def test_original_exception_chained_as_cause(
        self, mock_client: MagicMock
    ) -> None:
        original = ValueError('original cause')
        mock_client.chat.side_effect = original
        gen = AnswerGenerator(model_client=mock_client)
        with pytest.raises(RagQueryError) as exc_info:
            gen.generate('q', [], 256, None)
        assert exc_info.value.__cause__ is original
