"""BudgetProjector 单元测试。

覆盖不设限制、未超限、soft-over、hard-over 四种投影场景，
以及自定义 output_reserve 和 soft_buffer 覆盖参数。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.context.budget_projector import BudgetProjector
from src.context.token_estimator import TokenEstimator
from src.core_contracts.model_contracts import Message


# =============================================================================
# 测试夹具
# =============================================================================


@pytest.fixture
def estimator() -> TokenEstimator:
    return TokenEstimator()


@pytest.fixture
def projector(estimator: TokenEstimator) -> BudgetProjector:
    return BudgetProjector(token_estimator=estimator)


@pytest.fixture
def messages() -> list[Message]:
    return [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
    ]


# =============================================================================
# 无限制场景
# =============================================================================


class TestProjectUnlimited:
    def test_no_limit_returns_all_false(self, projector: BudgetProjector, messages: list[Message]) -> None:
        """不设 max_input_tokens 时两个 over 标记均为 False。"""
        result = projector.project(messages)
        assert result.is_hard_over is False
        assert result.is_soft_over is False
        assert result.hard_input_limit is None
        assert result.soft_input_limit is None

    def test_no_limit_projected_positive(self, projector: BudgetProjector, messages: list[Message]) -> None:
        """不设限制时 projected_input_tokens 依然为正数。"""
        result = projector.project(messages)
        assert result.projected_input_tokens > 0

    def test_output_reserve_reflected(self, projector: BudgetProjector, messages: list[Message]) -> None:
        """output_reserve_tokens 按传入值返回。"""
        result = projector.project(messages, output_reserve_tokens=1024)
        assert result.output_reserve_tokens == 1024


# =============================================================================
# 未超限场景
# =============================================================================


class TestProjectWithinLimits:
    def test_within_limits(self, projector: BudgetProjector, messages: list[Message]) -> None:
        """投影远小于上限时两个 over 均为 False。"""
        result = projector.project(messages, max_input_tokens=100_000)
        assert result.is_hard_over is False
        assert result.is_soft_over is False

    def test_hard_limit_set(self, projector: BudgetProjector, messages: list[Message]) -> None:
        """设了上限后 hard_input_limit 与 soft_input_limit 均被计算。"""
        result = projector.project(messages, max_input_tokens=10_000)
        assert result.hard_input_limit == 10_000
        assert result.soft_input_limit is not None


# =============================================================================
# soft-over 场景
# =============================================================================


class TestProjectSoftOver:
    def test_soft_over_triggered(self, projector: BudgetProjector) -> None:
        """当 projected > soft_limit 但 <= usable 时，is_soft_over=True，is_hard_over=False。"""
        # 构造：估算 50 token，soft_limit 很小（max_input=200，reserve=100，soft_buffer=90 → soft=10）
        mock_estimator = MagicMock()
        mock_estimator.estimate_messages.return_value = 50
        mock_estimator.estimate_tools.return_value = 0
        proj = BudgetProjector(
            token_estimator=mock_estimator,
            default_output_reserve_tokens=100,
            default_soft_buffer_tokens=90,
        )
        msgs = [Message(role="user", content="hi")]
        result = proj.project(msgs, max_input_tokens=200)
        # usable=200-100=100, soft_limit=max(0,100-90)=10 → 50 > 10 soft_over
        assert result.is_soft_over is True
        assert result.is_hard_over is False


# =============================================================================
# hard-over 场景
# =============================================================================


class TestProjectHardOver:
    def test_hard_over_triggered(self, projector: BudgetProjector) -> None:
        """当 projected > usable 时，is_hard_over=True。"""
        mock_estimator = MagicMock()
        mock_estimator.estimate_messages.return_value = 200
        mock_estimator.estimate_tools.return_value = 0
        proj = BudgetProjector(token_estimator=mock_estimator)
        msgs = [Message(role="user", content="hi")]
        result = proj.project(msgs, max_input_tokens=100, output_reserve_tokens=50)
        # usable=100-50=50, projected=200 → hard_over
        assert result.is_hard_over is True
        assert result.is_soft_over is True  # hard_over 必然也 soft_over


# =============================================================================
# 工具 token 纳入计算
# =============================================================================


class TestProjectWithTools:
    def test_tools_add_to_projected(self, projector: BudgetProjector, messages: list[Message]) -> None:
        """携带工具 schema 时 projected_input_tokens 大于不携带时的值。"""
        tools = [{"type": "function", "function": {"name": "run", "parameters": {}}}]
        result_no_tools = projector.project(messages)
        result_with_tools = projector.project(messages, tools=tools)
        assert result_with_tools.projected_input_tokens > result_no_tools.projected_input_tokens

    def test_none_tools_treated_as_empty(self, projector: BudgetProjector, messages: list[Message]) -> None:
        """tools=None 与 tools=[] 投影结果相同。"""
        result_none = projector.project(messages, tools=None)
        result_empty = projector.project(messages, tools=[])
        assert result_none.projected_input_tokens == result_empty.projected_input_tokens
