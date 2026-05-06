"""Agent 应用层编排网关。

本模块提供 Agent 模块的唯一对外入口，负责依赖注入和委托执行。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.agent_executor import AgentLoopExecutor
from src.client import ClientGateway
from src.context import ContextGateway
from src.core_contracts.context_contracts import (
    BudgetConfig,
    ContextPolicy,
    PreModelBudgetGuard,
)
from src.core_contracts.session_contracts import SessionState
from src.interaction import InteractionGateway
from src.session import SessionGateway
from src.tools import ToolsGateway


@dataclass
class AgentGateway:
    """Agent 主流程编排网关。

    作为 Agent 模块的唯一对外门面，负责接收外部依赖并委托给内部执行器。
    """

    client: ClientGateway
    """ClientGateway: 模型调用客户端。"""

    context_gateway: ContextGateway
    """ContextGateway: 上下文治理网关。"""

    session_gateway: SessionGateway
    """SessionGateway: 会话状态管理网关。"""

    tools_gateway: ToolsGateway
    """ToolsGateway: 工具执行网关。"""

    interaction_gateway: InteractionGateway
    """InteractionGateway: 交互观察网关。"""

    budget_config: BudgetConfig
    """BudgetConfig: Token 预算配置。"""

    context_policy: ContextPolicy
    """ContextPolicy: 上下文治理策略。"""

    budget_guard: PreModelBudgetGuard
    """PreModelBudgetGuard: 预算守卫。"""

    executor: AgentLoopExecutor
    """AgentLoopExecutor: 内部执行器实例。"""

    def run(self, state: SessionState) -> SessionState | None:
        """执行一次 Agent 迭代循环：pre-model → LLM → 工具执行。

        Args:
            state (SessionState): 当前会话状态
        Returns:
            SessionState | None: 更新后的会话状态，失败时返回 None
        Raises:
            None
        """
        return self.executor.execute(state)
