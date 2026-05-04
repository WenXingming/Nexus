"""
Tools 门面网关。

该文件是 tools 子系统对外唯一入口，暴露两个核心接口：
1. list_tools  —— 发现所有可调用工具
2. execute_tool —— 执行指定工具
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core_contracts.tools_contracts import (
    ToolDescriptor,
    ToolExecutionRequest,
    ToolExecutionResult,
)
from src.tools.tool_executor import ToolExecutor
from src.tools.tool_registry import ToolRegistry


@dataclass
class ToolsGateway:
    """tools 领域唯一对外 Facade。

    Attributes:
        local_executor: 本地工具执行器。
        tool_registry: 已注册工具的注册表。
    """

    local_executor: ToolExecutor
    tool_registry: ToolRegistry

    # -------------------------------------------------------------------------
    # 公有接口
    # -------------------------------------------------------------------------

    def list_tools(self) -> list[ToolDescriptor]:
        """返回当前已注册的全部工具描述符列表。

        Returns:
            list[ToolDescriptor]: 工具描述符列表，可用于转换为 LLM function-calling 格式。
        """
        return list(self.tool_registry.values())

    def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """执行一次工具调用并返回结果。

        Args:
            request (ToolExecutionRequest): 包含工具名与参数的请求契约。

        Returns:
            ToolExecutionResult: 工具执行结果（含成功/失败状态）。
        """
        return self.local_executor.execute(self.tool_registry, request)
