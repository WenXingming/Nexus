"""context 领域统一网关（门面）。

ContextGateway 是 context 模块对外的唯一公开入口，隔离全部内部实现细节：
1. project_budget()                  —— 预算投影：评估本次调用的 token 预算快照；
2. run_pre_model_cycle()             —— pre-model 治理：snip → guard → auto-compact → guard；
3. run_reactive_compact_cycle()      —— 仅处理 context error 的恢复步骤（不发起模型调用）。

外部代码只能通过 src/context/__init__.py 的 ContextGateway 访问本包能力；
内部实现类（BudgetProjector、Snipper、Compactor 等）仅供单元测试通过子模块路径访问。
"""

from __future__ import annotations

from src.core_contracts.model_contracts import Message, TokenUsage
from src.core_contracts.context_contracts import (
    BudgetConfig,
    BudgetProjection,
    CompactionResult,
    ContextPolicy,
    ContextRunState,
    PreModelBudgetGuard,
    PreModelContextOutcome,
    ReactiveCompactOutcome,
    SnipResult,
)
from src.context.token_estimator import TokenEstimator
from src.context.compactor import Compactor
from src.context.snipper import Snipper


_MAX_REACTIVE_RETRIES: int = 2
"""int: reactive compact 允许的最大重试轮次。"""


def _add_usage(a: TokenUsage, b: TokenUsage) -> TokenUsage:
    """合并两个 TokenUsage 统计值。

    Args:
        a (TokenUsage): 被加数。
        b (TokenUsage): 加数。
    Returns:
        TokenUsage: 各字段求和后的新 TokenUsage 实例。
    Raises:
        无。
    """
    return TokenUsage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
    )


class ContextGateway:
    """对外暴露 context 治理能力并严格隔离内部实现细节的网关类。

    所有内部构件（TokenEstimator、Snipper、Compactor）通过构造注入，
    网关本身不实例化任何内部类，只负责编排协调。
    """

    def __init__(
        self,
        *,
        token_estimator: TokenEstimator,
        snipper: Snipper,
        compactor: Compactor | None = None,
    ) -> None:
        """通过依赖注入初始化 context 网关。

        Args:
            token_estimator (TokenEstimator): token 估算器实例，兼预算投影。
            snipper (Snipper): 轻量剪裁器实例，负责 tombstone 化降载。
            compactor (Compactor | None): 摘要压缩器实例；无客户端时传 None。
        Returns:
            None
        Raises:
            无。
        """
        self._token_estimator = token_estimator
        # TokenEstimator: token 估算器，用于预检输入上下文成本。
        self._snipper = snipper
        # Snipper: 轻量剪裁器，用于 soft-over 阶段的 tombstone 化降载。
        self._compactor = compactor
        # Compactor | None: 摘要压缩器；无客户端时保持禁用。

    # =========================================================================
    # 公有接口
    # =========================================================================

    def project_budget(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        budget_config: BudgetConfig | None = None,
    ) -> BudgetProjection:
        """对当前消息和工具集合执行预算投影。

        Args:
            messages (list[Message]): 当前会话消息列表。
            tools (list[dict] | None): 当前工具 schema 列表；None 等同于空列表。
            budget_config (BudgetConfig | None): 预算约束配置；None 使用不设限默认值。
        Returns:
            BudgetProjection: 预算快照，含 projected/hard/soft 投影量与 over 标记。
        Raises:
            无。
        """
        cfg = budget_config or BudgetConfig()
        return self._project_budget(messages, cfg, tools or [])

    def compact_messages(
        self,
        messages: list[Message],
        *,
        preserve_messages: int = 4,
    ) -> tuple[list[Message], CompactionResult]:
        """手动触发 context compact。

        供 /compact 等用户命令调用，将旧对话历史压缩为摘要。

        Args:
            messages (list[Message]): 当前会话消息列表（不修改原列表）。
            preserve_messages (int): 尾部保留不参与压缩的消息条数。
        Returns:
            tuple[list[Message], CompactionResult]: (压缩后的消息列表副本, compact 执行结果)。
        Raises:
            无。
        """
        if self._compactor is None:
            return messages[:], CompactionResult(compacted=False, error="Compactor not available")
        return self._compactor.compact(messages, preserve_messages=preserve_messages)

    def run_pre_model_cycle(
        self,
        *,
        run_state: ContextRunState,
        budget_config: BudgetConfig,
        context_policy: ContextPolicy,
        guard: PreModelBudgetGuard,
        tools: list[dict],
    ) -> PreModelContextOutcome:
        """执行模型调用前的完整上下文治理编排。

        治理顺序：
        预算投影 → soft-over 时 snip → pre-model guard →
        auto-compact（如触发阈值）→ 再次 guard → budget 事件记录。

        Args:
            run_state (ContextRunState): 当前 turn 的运行态（就地更新统计与快照）。
            budget_config (BudgetConfig): 预算约束配置。
            context_policy (ContextPolicy): 上下文治理策略。
            guard (PreModelBudgetGuard): pre-model 预算守卫，决定是否允许继续调用。
            tools (list[dict]): 当前工具 schema 列表，纳入 token 估算。
        Returns:
            PreModelContextOutcome: pre-model 阶段的 stop reason 与有序事件集合。
        Raises:
            RuntimeError: 需要 auto-compact 但未配置模型客户端时抛出。
        """
        events: list[dict] = []

        snapshot = self._project_budget(run_state.session_messages, budget_config, tools)

        if snapshot.is_soft_over:
            run_state.session_messages, snip_result = self._snipper.snip(
                run_state.session_messages,
                preserve_messages=context_policy.compact_preserve_messages,
            )
            if snip_result.snipped_count > 0:
                events.append(_build_snip_event(run_state.turn_index, snip_result))
                snapshot = self._project_budget(run_state.session_messages, budget_config, tools)

        stop = self._check_guard(guard, run_state, snapshot)

        if (
            stop is None
            and self._compactor is not None
            and self._should_auto_compact(snapshot, context_policy)
        ):
            run_state.session_messages, compact_result = self._compactor.compact(
                run_state.session_messages,
                preserve_messages=context_policy.compact_preserve_messages,
            )
            if compact_result.compacted:
                run_state.model_call_count += 1
                run_state.usage_delta = _add_usage(run_state.usage_delta, compact_result.usage)
                events.append(_build_compact_event(run_state.turn_index, "auto", compact_result))
                snapshot = self._project_budget(run_state.session_messages, budget_config, tools)
                stop = self._check_guard(guard, run_state, snapshot)
            elif compact_result.error:
                events.append(
                    _build_compact_failed_event(
                        run_state.turn_index,
                        "auto",
                        compact_result.error,
                        context_policy.compact_preserve_messages,
                    )
                )

        events.append(_build_budget_event(run_state.turn_index, snapshot))
        run_state.token_budget_snapshot = snapshot

        return PreModelContextOutcome(pre_model_stop=stop, events=tuple(events))

    def run_reactive_compact_cycle(
        self,
        *,
        run_state: ContextRunState,
        budget_config: BudgetConfig,
        context_policy: ContextPolicy,
        tools: list[dict],
        guard: PreModelBudgetGuard,
        error: Exception,
        attempt: int,
    ) -> ReactiveCompactOutcome:
        """执行 reactive compact 恢复步骤（不发起模型调用）。

        调用方负责真实的 chat 调用与异常捕获；本方法仅在已捕获异常后
        判断是否可恢复、执行 compact、更新 run_state 并给出是否应重试调用。

        Args:
            run_state (ContextRunState): 当前 turn 的运行态（就地更新统计）。
            budget_config (BudgetConfig): 预算约束配置。
            context_policy (ContextPolicy): 上下文治理策略（compact 保留消息数等）。
            tools (list[dict]): 当前工具 schema 列表。
            guard (PreModelBudgetGuard): 预算守卫，用于重试后的二次检查。
            error (Exception): 调用方捕获到的模型异常。
            attempt (int): 当前恢复尝试序号（从 1 开始）。
        Returns:
            ReactiveCompactOutcome: 是否可重试模型调用、stop_reason 及事件序列。
        Raises:
            无。
        """
        events: list[dict] = []
        compactor = self._compactor
        context_error = str(error)
        if compactor is None or not compactor.is_context_length_error(error):
            events.append(_build_error_event(run_state.turn_index, context_error))
            return ReactiveCompactOutcome(retry_model_call=False, stop_reason=None, events=tuple(events))

        if attempt > _MAX_REACTIVE_RETRIES:
            events.append(_build_error_event(run_state.turn_index, context_error))
            return ReactiveCompactOutcome(retry_model_call=False, stop_reason=None, events=tuple(events))

        preserve_messages = max(1, context_policy.compact_preserve_messages - (attempt - 1))
        run_state.session_messages, compact_result = compactor.compact(
            run_state.session_messages,
            preserve_messages=preserve_messages,
        )

        if not compact_result.compacted:
            events.append(
                _build_reactive_failed_event(
                    run_state.turn_index,
                    attempt,
                    preserve_messages,
                    context_error,
                    compact_result.error or "No progress made",
                )
            )
            events.append(_build_error_event(run_state.turn_index, context_error))
            return ReactiveCompactOutcome(retry_model_call=False, stop_reason=None, events=tuple(events))

        run_state.model_call_count += 1
        run_state.usage_delta = _add_usage(run_state.usage_delta, compact_result.usage)
        events.append(_build_compact_event(run_state.turn_index, "reactive", compact_result, attempt=attempt))
        events.append(
            _build_reactive_ok_event(
                run_state.turn_index,
                attempt,
                preserve_messages,
                context_error,
                compact_result,
            )
        )

        snapshot = self._project_budget(run_state.session_messages, budget_config, tools)
        run_state.token_budget_snapshot = snapshot
        stop = self._check_guard(guard, run_state, snapshot)
        if stop is not None:
            return ReactiveCompactOutcome(retry_model_call=False, stop_reason=stop, events=tuple(events))

        return ReactiveCompactOutcome(retry_model_call=True, stop_reason=None, events=tuple(events))

    # =========================================================================
    # 私有辅助（原子步骤）
    # =========================================================================

    def _project_budget(
        self,
        messages: list[Message],
        budget_config: BudgetConfig,
        tools: list[dict],
    ) -> BudgetProjection:
        """委托 TokenEstimator 执行预算投影。

        Args:
            messages (list[Message]): 当前会话消息列表。
            budget_config (BudgetConfig): 预算约束配置。
            tools (list[dict]): 工具 schema 列表。
        Returns:
            BudgetProjection: 预算快照。
        Raises:
            无。
        """
        return self._token_estimator.project_budget(
            messages,
            tools=tools,
            max_input_tokens=budget_config.max_input_tokens,
            output_reserve_tokens=budget_config.output_reserve_tokens,
            soft_buffer_tokens=budget_config.soft_buffer_tokens,
        )

    def _check_guard(
        self,
        guard: PreModelBudgetGuard,
        run_state: ContextRunState,
        snapshot: BudgetProjection,
    ) -> str | None:
        """委托预算守卫执行 pre-model 检查。

        Args:
            guard (PreModelBudgetGuard): 预算守卫实例。
            run_state (ContextRunState): 当前运行态。
            snapshot (BudgetProjection): 当前预算快照。
        Returns:
            str | None: 停止原因；允许继续时为 None。
        Raises:
            无。
        """
        return guard.check_pre_model(
            turns_offset=run_state.turns_offset,
            turns_this_run=run_state.turns_this_run,
            model_call_count=run_state.model_call_count,
            snapshot=snapshot,
            usage_delta=run_state.usage_delta,
        )

    def _should_auto_compact(
        self,
        snapshot: BudgetProjection,
        context_policy: ContextPolicy,
    ) -> bool:
        """判断当前投影 token 是否达到 auto compact 阈值。

        Args:
            snapshot (BudgetProjection): 当前预算快照。
            context_policy (ContextPolicy): 上下文治理策略。
        Returns:
            bool: True 表示需要触发 auto compact。
        Raises:
            无。
        """
        if context_policy.auto_compact_threshold_tokens is None:
            return False
        return snapshot.projected_input_tokens >= max(0, context_policy.auto_compact_threshold_tokens)

    # 事件构造委托给模块级函数 _build_*_event()


# =========================================================================
# 事件构造辅助（模块级纯函数，无业务逻辑）
# =========================================================================


def _build_budget_event(turn_index: int, snapshot: BudgetProjection) -> dict:
    return {
        "type": "token_budget",
        "turn": turn_index,
        "projected": snapshot.projected_input_tokens,
        "hard_input_limit": snapshot.hard_input_limit,
        "soft_input_limit": snapshot.soft_input_limit,
        "is_hard_over": snapshot.is_hard_over,
        "is_soft_over": snapshot.is_soft_over,
    }


def _build_snip_event(turn_index: int, result: SnipResult) -> dict:
    return {
        "type": "snip_boundary",
        "turn": turn_index,
        "snipped_count": result.snipped_count,
        "tokens_removed": result.tokens_removed,
    }


def _build_compact_event(
    turn_index: int,
    trigger: str,
    result: CompactionResult,
    *,
    attempt: int | None = None,
) -> dict:
    event: dict = {
        "type": "compact_boundary",
        "turn": turn_index,
        "trigger": trigger,
        "messages_replaced": result.messages_replaced,
        "tokens_removed": result.tokens_removed,
        "pre_tokens": result.pre_tokens,
        "post_tokens": result.post_tokens,
        "preserve_messages": result.preserve_messages_used,
    }
    if attempt is not None:
        event["attempt"] = attempt
    return event


def _build_compact_failed_event(
    turn_index: int,
    trigger: str,
    error: str,
    preserve_messages: int,
) -> dict:
    return {
        "type": "compact_failed",
        "turn": turn_index,
        "trigger": trigger,
        "error": error,
        "preserve_messages": preserve_messages,
    }


def _build_reactive_ok_event(
    turn_index: int,
    attempt: int,
    preserve_messages: int,
    context_error: str,
    result: CompactionResult,
) -> dict:
    return {
        "type": "reactive_compact_retry",
        "turn": turn_index,
        "attempt": attempt,
        "preserve_messages": preserve_messages,
        "context_error": context_error,
        "ok": True,
        "tokens_removed": result.tokens_removed,
        "messages_replaced": result.messages_replaced,
    }


def _build_reactive_failed_event(
    turn_index: int,
    attempt: int,
    preserve_messages: int,
    context_error: str,
    error: str,
) -> dict:
    return {
        "type": "reactive_compact_retry",
        "turn": turn_index,
        "attempt": attempt,
        "preserve_messages": preserve_messages,
        "context_error": context_error,
        "ok": False,
        "error": error,
    }


def _build_error_event(turn_index: int, error: str) -> dict:
    return {
        "type": "backend_error",
        "turn": turn_index,
        "error": error,
    }
