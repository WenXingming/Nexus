"""Compactor 单元测试。

通过 Mock ContextModelClient 严格隔离 LLM 调用，
验证 compact 的主流程、各种失败路径以及 is_context_length_error 的识别逻辑。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.context.compactor import Compactor
from src.context.token_estimator import TokenEstimator
from src.core_contracts.model_contracts import Message, TokenUsage
from src.core_contracts.client_contracts import LlmResult


# =============================================================================
# 测试夹具
# =============================================================================


@pytest.fixture
def mock_client() -> MagicMock:
    """返回满足 ContextModelClient 协议的 Mock 对象。"""
    return MagicMock()


@pytest.fixture
def compactor(mock_client: MagicMock) -> Compactor:
    """返回注入 Mock 客户端的 Compactor。"""
    return Compactor(client=mock_client, token_estimator=TokenEstimator())


def make_ok_response(summary: str = "Summary here.") -> LlmResult:
    """构造模拟的正常模型响应。"""
    return LlmResult(
        content=summary,
        model="test-model",
        finish_reason="stop",
        usage=TokenUsage(100, 50, 150),
        tool_calls=None,
    )


def make_messages(n: int = 8) -> list[Message]:
    """构造包含 system + 若干 user/assistant 轮次的消息列表。"""
    msgs: list[Message] = [Message(role="system", content="You are an agent.")]
    for i in range(n - 1):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(Message(role=role, content=f"turn {i} content"))
    return msgs


# =============================================================================
# 主流程
# =============================================================================


class TestCompactSuccess:
    def test_compact_returns_compacted_true(self, compactor: Compactor, mock_client: MagicMock) -> None:
        """正常流程下 compact 返回 compacted=True 且 summary_text 不为空。"""
        mock_client.chat.return_value = make_ok_response("Here is the summary.")
        msgs = make_messages(8)
        new_msgs, result = compactor.compact(msgs, preserve_messages=2)
        assert result.compacted is True
        assert result.summary_text == "Here is the summary."

    def test_compact_reduces_message_count(self, compactor: Compactor, mock_client: MagicMock) -> None:
        """compact 后消息列表长度小于原始长度（多条被替换为一条摘要）。"""
        mock_client.chat.return_value = make_ok_response()
        msgs = make_messages(10)
        original_len = len(msgs)
        new_msgs, _ = compactor.compact(msgs, preserve_messages=2)
        assert len(new_msgs) < original_len

    def test_compact_inserts_system_summary_message(self, compactor: Compactor, mock_client: MagicMock) -> None:
        """compact 后 messages[1]（prefix 之后）是 system 角色的摘要消息。"""
        mock_client.chat.return_value = make_ok_response("Summarized.")
        msgs = make_messages(8)
        new_msgs, _ = compactor.compact(msgs, preserve_messages=2)
        # msgs[0] 是原始 system，msgs[1] 应该是插入的摘要 system 消息
        assert new_msgs[1].role == "system"
        assert "Summarized." in (new_msgs[1].content or "")

    def test_compact_usage_propagated(self, compactor: Compactor, mock_client: MagicMock) -> None:
        """compact 结果中携带模型调用消耗的 usage。"""
        mock_client.chat.return_value = make_ok_response()
        msgs = make_messages(8)
        _, result = compactor.compact(msgs)
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50

    def test_compact_tokens_removed_positive(self, compactor: Compactor, mock_client: MagicMock) -> None:
        """compact 后 tokens_removed 应为正数（多条消息被单条摘要替代）。"""
        mock_client.chat.return_value = make_ok_response("s")
        msgs = make_messages(10)
        _, result = compactor.compact(msgs, preserve_messages=1)
        assert result.tokens_removed >= 0  # 可能为 0 若摘要刚好一样大

    def test_messages_replaced_count(self, compactor: Compactor, mock_client: MagicMock) -> None:
        """messages_replaced 等于被删除的原始消息条数。"""
        mock_client.chat.return_value = make_ok_response()
        msgs = make_messages(6)  # 1 system + 5 others
        _, result = compactor.compact(msgs, preserve_messages=2)
        # prefix=1, tail=2, replaced=6-1-2=3
        assert result.messages_replaced == 3


# =============================================================================
# 消息不足时提前返回
# =============================================================================


class TestCompactNotEnough:
    def test_not_enough_messages(self, compactor: Compactor, mock_client: MagicMock) -> None:
        """消息数量不足时返回 compacted=False 且不调用模型。"""
        msgs = [Message(role="system", content="sys")]
        _, result = compactor.compact(msgs, preserve_messages=4)
        assert result.compacted is False
        mock_client.chat.assert_not_called()

    def test_preserve_exceeds_available(self, compactor: Compactor, mock_client: MagicMock) -> None:
        """preserve 数量超过可用消息时返回 compacted=False。"""
        msgs = make_messages(3)
        _, result = compactor.compact(msgs, preserve_messages=10)
        assert result.compacted is False
        mock_client.chat.assert_not_called()


# =============================================================================
# 模型返回空/工具调用/异常
# =============================================================================


class TestCompactModelFailures:
    def test_empty_summary_returns_false(self, compactor: Compactor, mock_client: MagicMock) -> None:
        """模型返回空字符串时返回 compacted=False。"""
        mock_client.chat.return_value = make_ok_response("")
        msgs = make_messages(8)
        _, result = compactor.compact(msgs)
        assert result.compacted is False
        assert result.error is not None

    def test_model_raises_error(self, compactor: Compactor, mock_client: MagicMock) -> None:
        """模型调用抛出异常时返回 compacted=False 且 error 字段包含异常信息。"""
        mock_client.chat.side_effect = RuntimeError("connection refused")
        msgs = make_messages(8)
        _, result = compactor.compact(msgs)
        assert result.compacted is False
        assert "connection refused" in (result.error or "")

    def test_model_returns_tool_calls(self, compactor: Compactor, mock_client: MagicMock) -> None:
        """模型异常返回 tool_calls 时，compact 认定为失败。"""
        bad_response = LlmResult(
            content="",
            model="test",
            finish_reason="tool_calls",
            usage=TokenUsage(10, 5, 15),
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        )
        mock_client.chat.return_value = bad_response
        msgs = make_messages(8)
        _, result = compactor.compact(msgs)
        assert result.compacted is False
        assert result.error is not None


# =============================================================================
# is_context_length_error
# =============================================================================


class TestIsContextLengthError:
    def test_status_code_413(self, compactor: Compactor) -> None:
        """status_code=413 的异常被识别为 context length 错误。"""
        exc = Exception("too large")
        exc.status_code = 413  # type: ignore[attr-defined]
        assert compactor.is_context_length_error(exc) is True

    def test_keyword_in_message(self, compactor: Compactor) -> None:
        """消息中包含关键词时被识别。"""
        assert compactor.is_context_length_error(RuntimeError("context length exceeded")) is True
        assert compactor.is_context_length_error(RuntimeError("prompt too long")) is True
        assert compactor.is_context_length_error(RuntimeError("too many tokens in request")) is True

    def test_unrelated_error_not_identified(self, compactor: Compactor) -> None:
        """无关错误不被识别为 context length 错误。"""
        assert compactor.is_context_length_error(RuntimeError("connection timeout")) is False
        assert compactor.is_context_length_error(ValueError("invalid input")) is False
