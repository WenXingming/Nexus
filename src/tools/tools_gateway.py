"""
Tools 门面网关。

该文件是 tools 子系统对外唯一入口，暴露两个核心接口：
1. list_tools  —— 发现所有可调用工具
2. execute_tool —— 执行指定工具
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core_contracts.tools_contracts import (
    JsonDict,
    ToolDescriptor,
    ToolExecutionRequest,
    ToolExecutionResult,
)
from src.tools.tool_executor import ToolExecutor
from src.tools.tool_registry import ToolRegistry


@dataclass
class ToolsGateway:
    """tools 领域唯一对外 Facade。

    支持两种工作模式：
    1. 同步模式：所有工具在创建时立即加载
    2. 异步模式：MCP 工具在后台异步加载，本地工具立即可用

    Attributes:
        local_executor: 本地工具执行器。
        tool_registry: 已注册工具的注册表。
        _async_mcp_provider: 异步 MCP 工具提供者（可选）。
    """

    local_executor: ToolExecutor
    tool_registry: ToolRegistry
    _async_mcp_provider: object | None = field(default=None, repr=False)
    _openai_tools_cache: list[JsonDict] = field(default_factory=list, init=False, repr=False)
    _tools_version: int = field(default=0, init=False)

    # -------------------------------------------------------------------------
    # 公有接口
    # -------------------------------------------------------------------------

    def list_tools(self) -> list[ToolDescriptor]:
        """返回当前已注册的全部工具描述符列表。

        如果使用异步模式且 MCP 工具已加载完成，会自动合并到结果中。

        Returns:
            list[ToolDescriptor]: 工具描述符列表，可用于转换为 LLM function-calling 格式。
        """
        self._try_merge_async_tools()
        return list(self.tool_registry.values())

    def list_openai_tools(self) -> list[JsonDict]:
        """返回 OpenAI function calling 格式的工具列表。

        自动处理异步工具合并和格式转换，使用缓存避免重复转换。

        Returns:
            list[JsonDict]: OpenAI function calling 格式的工具列表。
        """
        self._try_merge_async_tools()

        # 检查是否需要重新转换
        current_version = len(self.tool_registry)
        if current_version != self._tools_version:
            self._openai_tools_cache = [
                tool.to_openai_tool() for tool in self.tool_registry.values()
            ]
            self._tools_version = current_version

        return self._openai_tools_cache

    def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """执行一次工具调用并返回结果。

        Args:
            request (ToolExecutionRequest): 包含工具名与参数的请求契约。

        Returns:
            ToolExecutionResult: 工具执行结果（含成功/失败状态）。
        """
        return self.local_executor.execute(self.tool_registry, request)

    def is_mcp_loading(self) -> bool:
        """检查 MCP 工具是否正在异步加载中。

        Returns:
            bool: True 表示正在加载；False 表示已完成或未使用异步模式。
        """
        if self._async_mcp_provider is None:
            return False
        return self._async_mcp_provider.is_loading()

    def wait_for_mcp_tools(self, timeout: float | None = 60) -> None:
        """等待 MCP 工具加载完成并注册到 registry。

        Args:
            timeout: 等待超时时间（秒）；None 表示无限等待。

        Raises:
            TimeoutError: 等待超时。
        """
        if self._async_mcp_provider is None:
            return

        tools = self._async_mcp_provider.get_tools(timeout=timeout)
        self._register_mcp_tools(tools)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _try_merge_async_tools(self) -> None:
        """尝试将异步加载的 MCP 工具合并到 registry。"""
        if self._async_mcp_provider is None:
            return

        if not self._async_mcp_provider.is_loading() and self._async_mcp_provider._load_event.is_set():
            tools = self._async_mcp_provider.get_tools(timeout=0)
            self._register_mcp_tools(tools)

    def _register_mcp_tools(self, tools: tuple[ToolDescriptor, ...]) -> None:
        """注册 MCP 工具到 registry。"""
        for tool in tools:
            if tool.name not in self.tool_registry:
                self.tool_registry[tool.name] = tool
