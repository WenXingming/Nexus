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


# =============================================================================
# project_budget
# =============================================================================


class TestProjectBudget:
    def test_no_limit_returns_all_false(self, estimator: TokenEstimator) -> None:
        """不设 max_input_tokens 时两个 over 标记均为 False。"""
        msgs = [Message(role="system", content="You are helpful."), Message(role="user", content="Hello")]
        result = estimator.project_budget(msgs)
        assert result.is_hard_over is False
        assert result.is_soft_over is False
        assert result.hard_input_limit is None
        assert result.soft_input_limit is None

    def test_no_limit_projected_positive(self, estimator: TokenEstimator) -> None:
        """不设限制时 projected_input_tokens 依然为正数。"""
        msgs = [Message(role="system", content="You are helpful."), Message(role="user", content="Hello")]
        result = estimator.project_budget(msgs)
        assert result.projected_input_tokens > 0

    def test_output_reserve_reflected(self, estimator: TokenEstimator) -> None:
        """output_reserve_tokens 按传入值返回。"""
        msgs = [Message(role="user", content="Hello")]
        result = estimator.project_budget(msgs, output_reserve_tokens=1024)
        assert result.output_reserve_tokens == 1024

    def test_within_limits(self, estimator: TokenEstimator) -> None:
        """投影远小于上限时两个 over 均为 False。"""
        msgs = [Message(role="user", content="Hello")]
        result = estimator.project_budget(msgs, max_input_tokens=100_000)
        assert result.is_hard_over is False
        assert result.is_soft_over is False

    def test_hard_limit_set(self, estimator: TokenEstimator) -> None:
        """设了上限后 hard_input_limit 与 soft_input_limit 均被计算。"""
        msgs = [Message(role="user", content="Hello")]
        result = estimator.project_budget(msgs, max_input_tokens=10_000)
        assert result.hard_input_limit == 10_000
        assert result.soft_input_limit is not None

    def test_soft_over_triggered(self) -> None:
        """当 projected > soft_limit 但 <= usable 时，is_soft_over=True，is_hard_over=False。"""
        from unittest.mock import MagicMock
        mock_estimator = MagicMock(spec=TokenEstimator)
        mock_estimator.estimate_messages.return_value = 50
        mock_estimator.estimate_tools.return_value = 0
        # 直接调用 project_budget 方法（使用真实实现）
        mock_estimator.project_budget = TokenEstimator.project_budget.__get__(mock_estimator)
        msgs = [Message(role="user", content="hi")]
        result = mock_estimator.project_budget(msgs, max_input_tokens=200, output_reserve_tokens=100, soft_buffer_tokens=90)
        # usable=200-100=100, soft_limit=max(0,100-90)=10 → 50 > 10 soft_over
        assert result.is_soft_over is True
        assert result.is_hard_over is False

    def test_hard_over_triggered(self) -> None:
        """当 projected > usable 时，is_hard_over=True。"""
        from unittest.mock import MagicMock
        mock_estimator = MagicMock(spec=TokenEstimator)
        mock_estimator.estimate_messages.return_value = 200
        mock_estimator.estimate_tools.return_value = 0
        mock_estimator.project_budget = TokenEstimator.project_budget.__get__(mock_estimator)
        msgs = [Message(role="user", content="hi")]
        result = mock_estimator.project_budget(msgs, max_input_tokens=100, output_reserve_tokens=50)
        # usable=100-50=50, projected=200 → hard_over
        assert result.is_hard_over is True
        assert result.is_soft_over is True  # hard_over 必然也 soft_over

    def test_tools_add_to_projected(self, estimator: TokenEstimator) -> None:
        """携带工具 schema 时 projected_input_tokens 大于不携带时的值。"""
        msgs = [Message(role="user", content="Hello")]
        tools = [{"type": "function", "function": {"name": "run", "parameters": {}}}]
        result_no_tools = estimator.project_budget(msgs)
        result_with_tools = estimator.project_budget(msgs, tools=tools)
        assert result_with_tools.projected_input_tokens > result_no_tools.projected_input_tokens

    def test_none_tools_treated_as_empty(self, estimator: TokenEstimator) -> None:
        """tools=None 与 tools=[] 投影结果相同。"""
        msgs = [Message(role="user", content="Hello")]
        result_none = estimator.project_budget(msgs, tools=None)
        result_empty = estimator.project_budget(msgs, tools=[])
        assert result_none.projected_input_tokens == result_empty.projected_input_tokens
