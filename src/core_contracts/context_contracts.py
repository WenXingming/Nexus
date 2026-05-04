"""Context 模块跨边界契约定义。

定义 token 预算预检、上下文治理（snip/compact）与模型调用流程中
跨边界共享的配置 DTO、结果 DTO 以及 Protocol 接口。
外部调用方与 context 模块的交互只能基于本文件定义的类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol  # noqa: F401 – retained for ContextModelClient

from src.core_contracts.model_contracts import Message, TokenUsage
from src.core_contracts.client_contracts import LlmRequest, LlmResult


# =============================================================================
# 配置 DTO
# =============================================================================


@dataclass(frozen=True)
class BudgetConfig:
    """Token 预算硬性约束配置。

    由外部运行时持有，通过 ContextGateway 接口传入，
    约束本次模型调用的输入 token 上限与输出预留量。
    """

    max_input_tokens: int | None = None
    """int | None: 输入 token 硬上限；None 表示不设限制。"""

    output_reserve_tokens: int = 4_096
    """int: 从输入上限中预留给模型输出的 token 数，默认 4096。"""

    soft_buffer_tokens: int = 13_000
    """int: 软缓冲区大小；投影超出（硬限 - 输出预留 - 软缓冲）时触发 soft_over，默认 13000。"""


@dataclass(frozen=True)
class ContextPolicy:
    """上下文治理策略配置。

    控制 snip/compact 的触发时机与保留策略，
    由外部运行时持有，通过 ContextGateway 接口传入。
    """

    compact_preserve_messages: int = 4
    """int: snip 与 compact 操作尾部保留的消息条数，默认 4。"""

    auto_compact_threshold_tokens: int | None = 24_000
    """int | None: 触发主动 compact 的 projected token 阈值；None 表示禁用主动 compact。默认 24000。"""


# =============================================================================
# 结果 DTO
# =============================================================================


@dataclass(frozen=True)
class BudgetProjection:
    """一次 token 预算预检的结果快照。"""

    projected_input_tokens: int
    """int: 本次调用预估消耗的输入 token 总量。"""

    output_reserve_tokens: int
    """int: 从输入上限中预留给模型输出的 token 数。"""

    hard_input_limit: int | None
    """int | None: 输入 token 硬上限；None 表示不设限。"""

    soft_input_limit: int | None
    """int | None: 输入 token 软上限；None 表示不设限。"""

    is_hard_over: bool
    """bool: 投影是否已超出硬上限可用空间。"""

    is_soft_over: bool
    """bool: 投影是否已超出软上限可用空间。"""


@dataclass(frozen=True)
class SnipResult:
    """一次 snip 操作的统计结果。"""

    snipped_count: int
    """int: 本次被 tombstone 化的消息条数。"""

    tokens_removed: int
    """int: 本次操作估算节省的 token 数量。"""


@dataclass(frozen=True)
class CompactionResult:
    """一次 compact 操作的完整结果。"""

    compacted: bool
    """bool: 本次 compact 是否实际执行并写回。"""

    summary_text: str = ""
    """str: 模型生成的摘要文本；未成功时为空字符串。"""

    messages_replaced: int = 0
    """int: 被摘要替换掉的原始消息条数。"""

    tokens_removed: int = 0
    """int: compact 后估算节省的 token 数量。"""

    pre_tokens: int = 0
    """int: compact 前消息列表的估算 token 总量。"""

    post_tokens: int = 0
    """int: compact 后消息列表的估算 token 总量。"""

    preserve_messages_used: int = 0
    """int: 本次实际保留的尾部消息条数。"""

    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0, 0))
    """TokenUsage: compact 调用消耗的模型 token 统计。"""

    error: str | None = None
    """str | None: 失败原因描述；成功时为 None。"""


@dataclass(frozen=True)
class PreModelContextOutcome:
    """pre-model 上下文治理阶段的结果快照。"""

    pre_model_stop: str | None
    """str | None: 预算守卫给出的停止原因；允许继续时为 None。"""

    events: tuple[dict, ...]
    """tuple[dict, ...]: 本轮 pre-model 阶段产生的有序事件序列。"""


@dataclass(frozen=True)
class ReactiveCompactOutcome:
    """reactive compact 恢复步骤的执行结果。"""

    retry_model_call: bool
    """bool: True 表示可由调用方重试模型调用；False 表示不可重试。"""

    stop_reason: str | None
    """str | None: 预算守卫给出的停止原因；未触发时为 None。"""

    events: tuple[dict, ...]
    """tuple[dict, ...]: reactive compact 阶段的有序事件序列。"""


# =============================================================================
# Protocol 接口
# =============================================================================


class ContextModelClient(Protocol):
    """context 模块所需的最小化模型客户端协议。

    任何满足此接口的对象均可作为 Compactor 与 ContextGateway 的模型客户端，
    无需依赖具体的 ClientGateway 实现类。
    """

    def chat(self, request: LlmRequest) -> LlmResult:
        """发起一次同步模型调用并返回完整结果。

        Args:
            request (LlmRequest): 标准化的模型调用请求。
        Returns:
            LlmResult: 包含生成文本与 token 统计的标准化结果。
        Raises:
            RuntimeError: 模型调用失败时抛出。
        """
        ...


@dataclass
class ContextRunState:
    """context 网关所需的 turn 级运行态，由主循环持有并传入 ContextGateway。

    context 治理产生的统计与快照通过属性就地更新，避免与主循环强耦合。
    """

    session_messages: list[Message]
    """list[Message]: 当前会话可变消息列表，供 snip/compact 就地修改。"""

    turn_index: int = 0
    """int: 当前 turn 在整个 run 中的序号（0-based）。"""

    usage_delta: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0, 0))
    """TokenUsage: 本次 run 累计消耗的模型 token 量（可被 context 模块写回）。"""

    model_call_count: int = 0
    """int: 本次 run 已发起的模型调用次数（可被 context 模块写回）。"""

    turns_offset: int = 0
    """int: 本次 run 相对于会话起始的 turn 偏移量，供 guard 使用。"""

    turns_this_run: int = 0
    """int: 本次 run 已完成的 turn 总数，供 guard 使用。"""

    token_budget_snapshot: BudgetProjection | None = None
    """BudgetProjection | None: 最近一次预算快照（可被 context 模块写回）."""


class PreModelBudgetGuard:
    """pre-model 阶段预算守卫默认实现。

    该实现只在硬预算超限时阻止继续调用。
    """

    def check_pre_model(
        self,
        *,
        turns_offset: int,
        turns_this_run: int,
        model_call_count: int,
        snapshot: BudgetProjection,
        usage_delta: TokenUsage,
    ) -> str | None:
        """执行 pre-model 预算检查（硬限守卫）。

        Args:
            turns_offset (int): 本次 run 相对于会话起始的 turn 偏移量。
            turns_this_run (int): 本次 run 已完成的 turn 总数。
            model_call_count (int): 本次 run 已发起的模型调用次数。
            snapshot (BudgetProjection): 当前预算快照。
            usage_delta (TokenUsage): 本次 run 累计消耗的 token 量。
        Returns:
            str | None: 停止原因；允许继续时返回 None。
        Raises:
            无。
        """
        del turns_offset, turns_this_run, model_call_count, usage_delta
        if snapshot.is_hard_over:
            return "hard_input_budget_exceeded"
        return None
