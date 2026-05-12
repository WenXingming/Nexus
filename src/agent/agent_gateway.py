"""Agent 应用层编排网关。

本模块提供 Agent 模块的唯一对外入口，负责依赖注入和委托执行。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.agent_executor import AgentLoopExecutor
from src.core_contracts.session_contracts import SessionState


@dataclass
class AgentGateway:
    """Agent 主流程编排网关。

    作为 Agent 模块的唯一对外门面，接收已装配好的内部执行器并委托执行。
    """

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
