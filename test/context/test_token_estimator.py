"""TokenEstimator 单元测试。

覆盖 estimate_message、estimate_messages、estimate_tools
以及底层 _count_chars 的边界行为。
"""

from __future__ import annotations

import pytest

from src.context.token_estimator import TokenEstimator
from src.core_contracts.model_contracts import Message


# =============================================================================
# 测试夹具
# =============================================================================


@pytest.fixture
def estimator() -> TokenEstimator:
    """返回使用默认参数的 TokenEstimator 实例。"""
    return TokenEstimator()


# =============================================================================
# estimate_message
# =============================================================================


class TestEstimateMessage:
    def test_simple_user_message(self, estimator: TokenEstimator) -> None:
        """普通用户文本消息估算结果大于固定开销。"""
        msg = Message(role="user", content="Hello world")
        result = estimator.estimate_message(msg)
        assert result > estimator._message_overhead

    def test_none_content_returns_overhead_plus_role(self, estimator: TokenEstimator) -> None:
        """content 为 None 时只计 role + 消息开销。"""
        msg = Message(role="assistant", content=None)
        expected = estimator._count_chars("assistant") + estimator._message_overhead
        assert estimator.estimate_message(msg) == expected

    def test_empty_content_string(self, estimator: TokenEstimator) -> None:
        """空字符串 content 与 None content 估算相同。"""
        msg_none = Message(role="user", content=None)
        msg_empty = Message(role="user", content="")
        assert estimator.estimate_message(msg_none) == estimator.estimate_message(msg_empty)

    def test_tool_calls_add_extra_tokens(self, estimator: TokenEstimator) -> None:
        """携带 tool_calls 的消息 token 数大于同等无 tool_calls 的消息。"""
        tool_calls = [{"id": "c1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}]
        msg_with_tc = Message(role="assistant", content=None, tool_calls=tool_calls)
        msg_without_tc = Message(role="assistant", content=None)
        assert estimator.estimate_message(msg_with_tc) > estimator.estimate_message(msg_without_tc)

    def test_tool_role_includes_tool_call_id(self, estimator: TokenEstimator) -> None:
        """tool 角色消息能正确被估算（tool_call_id 不计入 token，只有 content/role 计）。"""
        msg = Message(role="tool", content="result text", tool_call_id="call_abc", name="my_tool")
        result = estimator.estimate_message(msg)
        assert result > estimator._message_overhead

    def test_longer_content_gives_more_tokens(self, estimator: TokenEstimator) -> None:
        """较长内容的估算 token 数多于较短内容。"""
        short_msg = Message(role="user", content="Hi")
        long_msg = Message(role="user", content="A" * 400)
        assert estimator.estimate_message(long_msg) > estimator.estimate_message(short_msg)


# =============================================================================
# estimate_messages
# =============================================================================


class TestEstimateMessages:
    def test_empty_list_returns_chat_base(self, estimator: TokenEstimator) -> None:
        """空列表返回 chat_base_tokens。"""
        assert estimator.estimate_messages([]) == estimator._chat_base

    def test_single_message(self, estimator: TokenEstimator) -> None:
        """单条消息总量 = chat_base + estimate_message(msg)。"""
        msg = Message(role="user", content="test")
        expected = estimator._chat_base + estimator.estimate_message(msg)
        assert estimator.estimate_messages([msg]) == expected

    def test_multiple_messages_accumulated(self, estimator: TokenEstimator) -> None:
        """多条消息总量等于各条加总再加 chat_base。"""
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there"),
        ]
        expected = estimator._chat_base + sum(estimator.estimate_message(m) for m in msgs)
        assert estimator.estimate_messages(msgs) == expected


# =============================================================================
# estimate_tools
# =============================================================================


class TestEstimateTools:
    def test_empty_tools_returns_zero(self, estimator: TokenEstimator) -> None:
        """空工具列表返回 0。"""
        assert estimator.estimate_tools([]) == 0

    def test_single_tool_returns_positive(self, estimator: TokenEstimator) -> None:
        """单个工具 schema 返回正数 token 估算。"""
        tool = {"type": "function", "function": {"name": "run_shell", "parameters": {}}}
        assert estimator.estimate_tools([tool]) > 0

    def test_more_tools_means_more_tokens(self, estimator: TokenEstimator) -> None:
        """更多工具 schema 给出更大 token 估算。"""
        tool = {"type": "function", "function": {"name": "x", "description": "desc"}}
        assert estimator.estimate_tools([tool, tool]) > estimator.estimate_tools([tool])


# =============================================================================
# _count_chars（边界条件）
# =============================================================================


class TestCountChars:
    def test_single_char_returns_one(self, estimator: TokenEstimator) -> None:
        """单个字符估算结果为 1（最小值）。"""
        assert estimator._count_chars("a") == 1

    def test_exact_multiple(self, estimator: TokenEstimator) -> None:
        """恰好整除时无进位。"""
        # 默认 4 chars/token，8 字符 = 2 token
        assert estimator._count_chars("a" * 8) == 2

    def test_rounds_up(self, estimator: TokenEstimator) -> None:
        """不足整除时向上取整。"""
        # 5 字符 / 4 = 1.25 → 向上取整为 2
        assert estimator._count_chars("a" * 5) == 2

    def test_custom_chars_per_token(self) -> None:
        """自定义 chars_per_token 生效。"""
        e = TokenEstimator(chars_per_token=2)
        assert e._count_chars("ab") == 1
        assert e._count_chars("abc") == 2
