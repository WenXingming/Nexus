"""Token 预算投影器。

提供 BudgetProjector，根据消息列表与工具 schema 的启发式 token 估算，
生成本次调用的预算快照（BudgetProjection），供后续 snip/compact/guard 链路使用。
"""

from __future__ import annotations

from src.core_contracts.model_contracts import Message
from src.core_contracts.context_contracts import BudgetProjection
from src.context.token_estimator import TokenEstimator


class BudgetProjector:
    """基于 TokenEstimator 生成上下文预算投影快照。

    核心职责：
    1. 估算当前消息与工具 schema 的 token 开销；
    2. 与 max_input_tokens 硬限对比，生成含 over 标记的 BudgetProjection；
    3. 快照由调用方（ContextGateway）决策是否触发 snip/compact 或中止。
    """

    def __init__(
        self,
        token_estimator: TokenEstimator,
        default_output_reserve_tokens: int = 4_096,
        default_soft_buffer_tokens: int = 13_000,
    ) -> None:
        """通过依赖注入初始化投影器。

        Args:
            token_estimator (TokenEstimator): 共享的启发式 token 估算器。
            default_output_reserve_tokens (int): 输出预留 token 默认值，默认 4096。
            default_soft_buffer_tokens (int): 软缓冲 token 默认值，默认 13000。
        Returns:
            None
        Raises:
            None
        """
        self._estimator = token_estimator
        # TokenEstimator: 共享的 token 估算器，由工厂函数注入。
        self._default_output_reserve = default_output_reserve_tokens
        # int: 未显式传入 output_reserve_tokens 时使用的默认值。
        self._default_soft_buffer = default_soft_buffer_tokens
        # int: 未显式传入 soft_buffer_tokens 时使用的默认值。

    # =========================================================================
    # 公有接口
    # =========================================================================

    def project(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        max_input_tokens: int | None = None,
        output_reserve_tokens: int | None = None,
        soft_buffer_tokens: int | None = None,
    ) -> BudgetProjection:
        """预检本次模型调用的 token 预算并返回快照。

        Args:
            messages (list[Message]): 当前会话消息列表。
            tools (list[dict] | None): 当前工具 schema 列表；None 等同于空列表。
            max_input_tokens (int | None): 输入 token 硬上限；None 表示不设限。
            output_reserve_tokens (int | None): 输出预留 token 覆盖值；None 使用默认值。
            soft_buffer_tokens (int | None): 软缓冲 token 覆盖值；None 使用默认值。
        Returns:
            BudgetProjection: 本次调用的预算快照，含 projected/hard/soft 及 over 标记。
        Raises:
            无。
        """
        output_reserve = output_reserve_tokens if output_reserve_tokens is not None else self._default_output_reserve
        soft_buffer = soft_buffer_tokens if soft_buffer_tokens is not None else self._default_soft_buffer
        projected = (
            self._estimator.estimate_messages(messages)
            + self._estimator.estimate_tools(tools or [])
        )

        if max_input_tokens is None:
            return self._build_unlimited_projection(projected, output_reserve)
        return self._build_limited_projection(projected, max_input_tokens, output_reserve, soft_buffer)

    # =========================================================================
    # 私有辅助（原子步骤）
    # =========================================================================

    def _build_unlimited_projection(self, projected: int, output_reserve: int) -> BudgetProjection:
        """构造不设上限的预算快照（所有 over 标记为 False）。

        Args:
            projected (int): 估算的输入 token 总量。
            output_reserve (int): 输出预留 token 数。
        Returns:
            BudgetProjection: hard/soft limit 均为 None 的快照。
        Raises:
            无。
        """
        return BudgetProjection(
            projected_input_tokens=projected,
            output_reserve_tokens=output_reserve,
            hard_input_limit=None,
            soft_input_limit=None,
            is_hard_over=False,
            is_soft_over=False,
        )

    def _build_limited_projection(
        self,
        projected: int,
        max_input_tokens: int,
        output_reserve: int,
        soft_buffer: int,
    ) -> BudgetProjection:
        """构造带硬限的预算快照，并计算 soft_limit 与 over 标记。

        Args:
            projected (int): 估算的输入 token 总量。
            max_input_tokens (int): 输入 token 硬上限。
            output_reserve (int): 输出预留 token 数。
            soft_buffer (int): 软缓冲 token 数（在 usable 空间内再额外预留）。
        Returns:
            BudgetProjection: 含硬限、软限与 over 标记的快照。
        Raises:
            无。
        """
        usable = max_input_tokens - output_reserve
        soft_limit = max(0, usable - soft_buffer)
        return BudgetProjection(
            projected_input_tokens=projected,
            output_reserve_tokens=output_reserve,
            hard_input_limit=max_input_tokens,
            soft_input_limit=soft_limit,
            is_hard_over=projected > usable,
            is_soft_over=projected > soft_limit,
        )
