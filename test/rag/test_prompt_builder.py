"""PromptBuilder 单元测试。

验证 build_messages() 的提示词构建逻辑：
  - system/user 消息结构
  - 上下文格式化
  - system_override 行为
  - 空 chunk 占位提示
"""

from __future__ import annotations

import pytest

from src.core_contracts.model_contracts import Message
from src.core_contracts.rag_contracts import (
    RagChunk,
    RagRetrievedChunk,
)
from src.rag.prompt_builder import PromptBuilder


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


def make_retrieved_chunk(
    chunk_id: str = 'c1',
    doc_id: str = 'd1',
    content: str = '相关内容片段',
    score: float = 0.9,
) -> RagRetrievedChunk:
    chunk = RagChunk(chunk_id=chunk_id, doc_id=doc_id, content=content, position=0)
    return RagRetrievedChunk(chunk=chunk, score=score)


class TestBuildMessages:
    def test_returns_two_messages(self, builder: PromptBuilder) -> None:
        messages = builder.build_messages(
            query='测试问题',
            chunks=[],
            max_tokens=256,
            system_override=None,
        )
        assert len(messages) == 2
        assert all(isinstance(m, Message) for m in messages)

    def test_roles_are_system_and_user(self, builder: PromptBuilder) -> None:
        messages = builder.build_messages('q', [], 256, None)
        assert messages[0].role == 'system'
        assert messages[1].role == 'user'

    def test_default_system_prompt_used_when_override_is_none(self, builder: PromptBuilder) -> None:
        messages = builder.build_messages('q', [], 256, None)
        assert '知识问答助手' in messages[0].content

    def test_system_override_replaces_default_prompt(self, builder: PromptBuilder) -> None:
        custom = '自定义系统提示词'
        messages = builder.build_messages('q', [], 256, custom)
        assert messages[0].content == custom

    def test_user_message_contains_query(self, builder: PromptBuilder) -> None:
        messages = builder.build_messages('什么是向量数据库', [], 256, None)
        assert '什么是向量数据库' in messages[1].content

    def test_user_message_contains_chunk_content(self, builder: PromptBuilder) -> None:
        chunks = [make_retrieved_chunk(content='向量检索的核心原理')]
        messages = builder.build_messages('问题', chunks, 256, None)
        assert '向量检索的核心原理' in messages[1].content

    def test_empty_chunks_shows_placeholder(self, builder: PromptBuilder) -> None:
        messages = builder.build_messages('问题', [], 256, None)
        assert '未找到相关参考资料' in messages[1].content

    def test_user_message_contains_max_tokens_hint(self, builder: PromptBuilder) -> None:
        messages = builder.build_messages('问题', [], 512, None)
        assert '512 tokens' in messages[1].content

    def test_multiple_chunks_are_separated(self, builder: PromptBuilder) -> None:
        chunks = [
            make_retrieved_chunk('c1', content='第一段'),
            make_retrieved_chunk('c2', content='第二段'),
        ]
        messages = builder.build_messages('问题', chunks, 256, None)
        assert '第一段' in messages[1].content
        assert '第二段' in messages[1].content
        assert '---' in messages[1].content

    def test_chunk_metadata_source_displayed(self, builder: PromptBuilder) -> None:
        chunk = RagChunk(
            chunk_id='c1', doc_id='d1', content='正文',
            position=0, metadata={'source': '/path/to/doc.md'},
        )
        retrieved = RagRetrievedChunk(chunk=chunk, score=0.95)
        messages = builder.build_messages('问题', [retrieved], 256, None)
        assert '/path/to/doc.md' in messages[1].content
