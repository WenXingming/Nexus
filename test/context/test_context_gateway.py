"""ContextGateway 单元测试。

通过 Mock 严格隔离 TokenEstimator、Snipper、Compactor、
PreModelBudgetGuard 以及 ContextModelClient，
验证三条公有路径的编排逻辑、事件产出与 run_state 状态更新。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, call

import pytest

from src.context.context_gateway import ContextGateway, _add_usage
from src.core_contracts.model_contracts import Message, TokenUsage
from src.core_contracts.client_contracts import LlmResult
from src.core_contracts.context_contracts import (
    BudgetConfig,
    BudgetProjection,
    CompactionResult,
    ContextPolicy,
    SnipResult,
)


# =============================================================================
# 辅助类：满足 ContextRunState 协议的可变数据类
# =============================================================================


@dataclass
class FakeRunState:
    """满足 ContextRunState 协议的测试用具体实现。"""

    session_messages: list[Message] = field(default_factory=list)
    turn_index: int = 0
    usage_delta: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0, 0))
    model_call_count: int = 0
    turns_offset: int = 0
    turns_this_run: int = 0
    token_budget_snapshot: BudgetProjection | None = None


# =============================================================================
# 辅助函数
# =============================================================================


def make_snapshot(
    projected: int = 100,
    *,
    soft_over: bool = False,
    hard_over: bool = False,
    max_input: int = 10_000,
) -> BudgetProjection:
    return BudgetProjection(
        projected_input_tokens=projected,
        output_reserve_tokens=1000,
        hard_input_limit=max_input,
        soft_input_limit=max_input - 1000,
        is_hard_over=hard_over,
        is_soft_over=soft_over,
    )


def make_llm_result(content: str = "ok") -> LlmResult:
    return LlmResult(
        content=content,
        model="test-model",
        finish_reason="stop",
        usage=TokenUsage(10, 5, 15),
        tool_calls=None,
    )


def make_compact_result(compacted: bool = True, replaced: int = 3) -> CompactionResult:
    return CompactionResult(
        compacted=compacted,
        summary_text="summary" if compacted else "",
        messages_replaced=replaced if compacted else 0,
        tokens_removed=200 if compacted else 0,
        pre_tokens=500,
        post_tokens=300,
        preserve_messages_used=4,
        usage=TokenUsage(20, 10, 30),
        error=None if compacted else "not enough",
    )


# =============================================================================
# 测试夹具
# =============================================================================


@pytest.fixture
def mock_estimator() -> MagicMock:
    est = MagicMock()
    est.project_budget.return_value = make_snapshot()
    return est


@pytest.fixture
def mock_snipper() -> MagicMock:
    snpr = MagicMock()
    snpr.snip.return_value = SnipResult(snipped_count=0, tokens_removed=0)
    return snpr


@pytest.fixture
def mock_compactor() -> MagicMock:
    cmp = MagicMock()
    cmp.compact.return_value = make_compact_result(compacted=False)
    cmp.is_context_length_error.return_value = False
    return cmp


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_guard() -> MagicMock:
    guard = MagicMock()
    guard.check_pre_model.return_value = None  # 默认允许继续
    return guard


@pytest.fixture
def gateway(
    mock_estimator: MagicMock,
    mock_snipper: MagicMock,
    mock_compactor: MagicMock,
) -> ContextGateway:
    return ContextGateway(
        token_estimator=mock_estimator,
        snipper=mock_snipper,
        compactor=mock_compactor,
    )


@pytest.fixture
def gateway_no_client(mock_estimator: MagicMock, mock_snipper: MagicMock) -> ContextGateway:
    return ContextGateway(token_estimator=mock_estimator, snipper=mock_snipper)


@pytest.fixture
def run_state() -> FakeRunState:
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="q"),
    ]
    return FakeRunState(session_messages=msgs)


@pytest.fixture
def budget_config() -> BudgetConfig:
    return BudgetConfig(max_input_tokens=10_000)


@pytest.fixture
def policy() -> ContextPolicy:
    return ContextPolicy(compact_preserve_messages=4, auto_compact_threshold_tokens=None)


# =============================================================================
# project_budget
# =============================================================================


class TestProjectBudget:
    def test_delegates_to_estimator(
        self,
        gateway: ContextGateway,
        mock_estimator: MagicMock,
        run_state: FakeRunState,
    ) -> None:
        """project_budget 将调用委托给 TokenEstimator。"""
        gateway.project_budget(run_state.session_messages)
        mock_estimator.project_budget.assert_called_once()

    def test_returns_projection(self, gateway: ContextGateway, run_state: FakeRunState) -> None:
        """project_budget 返回 BudgetProjection 对象。"""
        result = gateway.project_budget(run_state.session_messages)
        assert isinstance(result, BudgetProjection)

    def test_none_config_uses_defaults(
        self,
        gateway: ContextGateway,
        mock_estimator: MagicMock,
        run_state: FakeRunState,
    ) -> None:
        """budget_config=None 时使用 BudgetConfig 默认值（不设硬限）。"""
        gateway.project_budget(run_state.session_messages, budget_config=None)
        _, kwargs = mock_estimator.project_budget.call_args
        assert kwargs["max_input_tokens"] is None


# =============================================================================
# run_pre_model_cycle — 正常路径
# =============================================================================


class TestRunPreModelCycleNormal:
    def test_returns_outcome_with_no_stop(
        self,
        gateway: ContextGateway,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        policy: ContextPolicy,
        mock_guard: MagicMock,
    ) -> None:
        """guard 返回 None 时 pre_model_stop 为 None。"""
        result = gateway.run_pre_model_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            guard=mock_guard,
            tools=[],
        )
        assert result.pre_model_stop is None

    def test_budget_event_always_emitted(
        self,
        gateway: ContextGateway,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        policy: ContextPolicy,
        mock_guard: MagicMock,
    ) -> None:
        """每次循环都产出 token_budget 事件。"""
        result = gateway.run_pre_model_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            guard=mock_guard,
            tools=[],
        )
        types = [e["type"] for e in result.events]
        assert "token_budget" in types

    def test_snapshot_written_to_run_state(
        self,
        gateway: ContextGateway,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        policy: ContextPolicy,
        mock_guard: MagicMock,
    ) -> None:
        """run_state.token_budget_snapshot 被更新。"""
        gateway.run_pre_model_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            guard=mock_guard,
            tools=[],
        )
        assert run_state.token_budget_snapshot is not None


# =============================================================================
# run_pre_model_cycle — snip 触发
# =============================================================================


class TestRunPreModelCycleSnip:
    def test_snip_triggered_when_soft_over(
        self,
        gateway: ContextGateway,
        mock_estimator: MagicMock,
        mock_snipper: MagicMock,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        policy: ContextPolicy,
        mock_guard: MagicMock,
    ) -> None:
        """is_soft_over=True 时 snipper.snip 被调用。"""
        mock_estimator.project_budget.return_value = make_snapshot(soft_over=True)
        mock_snipper.snip.return_value = SnipResult(snipped_count=0, tokens_removed=0)
        gateway.run_pre_model_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            guard=mock_guard,
            tools=[],
        )
        mock_snipper.snip.assert_called_once()

    def test_snip_event_emitted_when_snipped(
        self,
        gateway: ContextGateway,
        mock_estimator: MagicMock,
        mock_snipper: MagicMock,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        policy: ContextPolicy,
        mock_guard: MagicMock,
    ) -> None:
        """snip_count > 0 时 snip_boundary 事件被产出。"""
        mock_estimator.project_budget.side_effect = [
            make_snapshot(soft_over=True),  # 第一次投影
            make_snapshot(soft_over=False),  # snip 后重新投影
        ]
        mock_snipper.snip.return_value = SnipResult(snipped_count=2, tokens_removed=300)
        result = gateway.run_pre_model_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            guard=mock_guard,
            tools=[],
        )
        types = [e["type"] for e in result.events]
        assert "snip_boundary" in types

    def test_no_snip_event_when_nothing_snipped(
        self,
        gateway: ContextGateway,
        mock_estimator: MagicMock,
        mock_snipper: MagicMock,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        policy: ContextPolicy,
        mock_guard: MagicMock,
    ) -> None:
        """snipped_count=0 时不产出 snip_boundary 事件。"""
        mock_estimator.project_budget.return_value = make_snapshot(soft_over=True)
        mock_snipper.snip.return_value = SnipResult(snipped_count=0, tokens_removed=0)
        result = gateway.run_pre_model_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            guard=mock_guard,
            tools=[],
        )
        types = [e["type"] for e in result.events]
        assert "snip_boundary" not in types


# =============================================================================
# run_pre_model_cycle — guard 拦截
# =============================================================================


class TestRunPreModelCycleGuardStop:
    def test_guard_stop_returned(
        self,
        gateway: ContextGateway,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        policy: ContextPolicy,
    ) -> None:
        """guard.check_pre_model 返回非 None 时 pre_model_stop 被传递出去。"""
        guard = MagicMock()
        guard.check_pre_model.return_value = "budget_exceeded"
        result = gateway.run_pre_model_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            guard=guard,
            tools=[],
        )
        assert result.pre_model_stop == "budget_exceeded"


# =============================================================================
# run_pre_model_cycle — auto-compact 触发
# =============================================================================


class TestRunPreModelCycleAutoCompact:
    def test_auto_compact_triggered(
        self,
        gateway: ContextGateway,
        mock_estimator: MagicMock,
        mock_compactor: MagicMock,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        mock_guard: MagicMock,
    ) -> None:
        """projected 超过 auto_compact_threshold 时 compactor.compact 被调用。"""
        mock_estimator.project_budget.return_value = make_snapshot(projected=9000)
        mock_compactor.compact.return_value = make_compact_result(compacted=True)
        policy = ContextPolicy(compact_preserve_messages=4, auto_compact_threshold_tokens=8000)
        gateway.run_pre_model_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            guard=mock_guard,
            tools=[],
        )
        mock_compactor.compact.assert_called_once()

    def test_compact_event_emitted_on_success(
        self,
        gateway: ContextGateway,
        mock_estimator: MagicMock,
        mock_compactor: MagicMock,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        mock_guard: MagicMock,
    ) -> None:
        """auto-compact 成功时产出 compact_boundary 事件。"""
        mock_estimator.project_budget.return_value = make_snapshot(projected=9000)
        mock_compactor.compact.return_value = make_compact_result(compacted=True)
        policy = ContextPolicy(compact_preserve_messages=4, auto_compact_threshold_tokens=8000)
        result = gateway.run_pre_model_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            guard=mock_guard,
            tools=[],
        )
        types = [e["type"] for e in result.events]
        assert "compact_boundary" in types

    def test_compact_failed_event_emitted(
        self,
        gateway: ContextGateway,
        mock_estimator: MagicMock,
        mock_compactor: MagicMock,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        mock_guard: MagicMock,
    ) -> None:
        """auto-compact 失败时产出 compact_failed 事件。"""
        mock_estimator.project_budget.return_value = make_snapshot(projected=9000)
        failed = CompactionResult(compacted=False, error="no messages to compact")
        mock_compactor.compact.return_value = failed
        policy = ContextPolicy(compact_preserve_messages=4, auto_compact_threshold_tokens=8000)
        result = gateway.run_pre_model_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            guard=mock_guard,
            tools=[],
        )
        types = [e["type"] for e in result.events]
        assert "compact_failed" in types

    def test_auto_compact_updates_usage_and_call_count(
        self,
        gateway: ContextGateway,
        mock_estimator: MagicMock,
        mock_compactor: MagicMock,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        mock_guard: MagicMock,
    ) -> None:
        """auto-compact 成功后 run_state.model_call_count += 1 且 usage_delta 增加。"""
        mock_estimator.project_budget.return_value = make_snapshot(projected=9000)
        compact_res = make_compact_result(compacted=True)
        mock_compactor.compact.return_value = compact_res
        policy = ContextPolicy(compact_preserve_messages=4, auto_compact_threshold_tokens=8000)
        gateway.run_pre_model_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            guard=mock_guard,
            tools=[],
        )
        assert run_state.model_call_count == 1
        assert run_state.usage_delta.total_tokens == compact_res.usage.total_tokens

    def test_no_auto_compact_without_client(
        self,
        gateway_no_client: ContextGateway,
        mock_estimator: MagicMock,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        mock_guard: MagicMock,
    ) -> None:
        """未配置 compactor 时，即使超过阈值也不触发 compact。"""
        mock_estimator.project_budget.return_value = make_snapshot(projected=9000)
        policy = ContextPolicy(compact_preserve_messages=4, auto_compact_threshold_tokens=8000)
        # 不应抛出，应正常完成
        result = gateway_no_client.run_pre_model_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            guard=mock_guard,
            tools=[],
        )
        assert result.pre_model_stop is None


# =============================================================================
# run_reactive_compact_cycle — reactive compact 恢复步骤
# =============================================================================


class TestRunReactiveCompactCycle:
    def test_context_error_compacted_then_retryable(
        self,
        gateway: ContextGateway,
        mock_compactor: MagicMock,
        mock_estimator: MagicMock,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        policy: ContextPolicy,
        mock_guard: MagicMock,
    ) -> None:
        """context 错误且 compact 成功时，返回 retry_model_call=True。"""
        mock_compactor.is_context_length_error.return_value = True
        mock_compactor.compact.return_value = make_compact_result(compacted=True)
        mock_estimator.project_budget.return_value = make_snapshot()

        result = gateway.run_reactive_compact_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            tools=[],
            guard=mock_guard,
            error=RuntimeError("context length exceeded"),
            attempt=1,
        )
        assert result.retry_model_call is True
        types = [e["type"] for e in result.events]
        assert "compact_boundary" in types
        assert "reactive_compact_retry" in types

    def test_non_context_error_not_retryable(
        self,
        gateway: ContextGateway,
        mock_compactor: MagicMock,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        policy: ContextPolicy,
        mock_guard: MagicMock,
    ) -> None:
        """非 context 错误直接返回不可重试。"""
        mock_compactor.is_context_length_error.return_value = False

        result = gateway.run_reactive_compact_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            tools=[],
            guard=mock_guard,
            error=RuntimeError("auth failed"),
            attempt=1,
        )
        assert result.retry_model_call is False
        types = [e["type"] for e in result.events]
        assert "backend_error" in types
        mock_compactor.compact.assert_not_called()

    def test_attempt_exceeded_not_retryable(
        self,
        gateway: ContextGateway,
        mock_compactor: MagicMock,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        policy: ContextPolicy,
        mock_guard: MagicMock,
    ) -> None:
        """超过最大重试序号后不可重试。"""
        mock_compactor.is_context_length_error.return_value = True

        result = gateway.run_reactive_compact_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            tools=[],
            guard=mock_guard,
            error=RuntimeError("context length exceeded"),
            attempt=3,
        )
        assert result.retry_model_call is False
        types = [e["type"] for e in result.events]
        assert "backend_error" in types

    def test_compact_failed_not_retryable(
        self,
        gateway: ContextGateway,
        mock_compactor: MagicMock,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        policy: ContextPolicy,
        mock_guard: MagicMock,
    ) -> None:
        """compact 失败时返回不可重试并产出失败事件。"""
        mock_compactor.is_context_length_error.return_value = True
        mock_compactor.compact.return_value = CompactionResult(compacted=False, error="no progress")

        result = gateway.run_reactive_compact_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            tools=[],
            guard=mock_guard,
            error=RuntimeError("context length exceeded"),
            attempt=1,
        )
        assert result.retry_model_call is False
        types = [e["type"] for e in result.events]
        assert "reactive_compact_retry" in types
        assert "backend_error" in types

    def test_guard_stop_after_compact(
        self,
        gateway: ContextGateway,
        mock_compactor: MagicMock,
        mock_estimator: MagicMock,
        run_state: FakeRunState,
        budget_config: BudgetConfig,
        policy: ContextPolicy,
    ) -> None:
        """compact 后 guard 拒绝继续时返回 stop_reason。"""
        mock_compactor.is_context_length_error.return_value = True
        mock_compactor.compact.return_value = make_compact_result(compacted=True)
        mock_estimator.project_budget.return_value = make_snapshot()
        guard = MagicMock()
        guard.check_pre_model.return_value = "hard_budget_exceeded"

        result = gateway.run_reactive_compact_cycle(
            run_state=run_state,
            budget_config=budget_config,
            context_policy=policy,
            tools=[],
            guard=guard,
            error=RuntimeError("context length exceeded"),
            attempt=1,
        )
        assert result.retry_model_call is False
        assert result.stop_reason == "hard_budget_exceeded"


# =============================================================================
# _add_usage 工具函数
# =============================================================================


class TestAddUsage:
    def test_adds_all_fields(self) -> None:
        """_add_usage 对三个字段分别求和。"""
        a = TokenUsage(10, 20, 30)
        b = TokenUsage(1, 2, 3)
        result = _add_usage(a, b)
        assert result.prompt_tokens == 11
        assert result.completion_tokens == 22
        assert result.total_tokens == 33

    def test_add_zeros(self) -> None:
        """加法零值保持原值不变。"""
        a = TokenUsage(5, 5, 10)
        b = TokenUsage(0, 0, 0)
        assert _add_usage(a, b) == a
