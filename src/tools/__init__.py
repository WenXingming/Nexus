"""Tools 模块对外装配入口。"""

from __future__ import annotations

from src.core_contracts.tools_contracts import McpToolConfig, ToolDescriptor
from src.tools.tool_executor import ToolExecutor as _ToolExecutor
from src.tools.local.filesystem_tools import FileSystemToolProvider as _FileSystemToolProvider
from src.tools.local.shell_tools import ShellToolProvider as _ShellToolProvider
from src.tools.mcp import McpToolProvider as _McpToolProvider
from src.tools.mcp.async_mcp_tools import AsyncMcpToolProvider as _AsyncMcpToolProvider
from src.tools.tool_registry import ToolRegistry as _ToolRegistry
from src.tools.tools_gateway import ToolsGateway


def create_gateway(
    mcp_provider: _McpToolProvider | None = None,
    *,
    mcp_config: McpToolConfig | None = None,
    async_mode: bool = False,
) -> ToolsGateway:
    """函数式门面工厂。

    Args:
        mcp_provider: 可选的 MCP 工具提供者，用于从 MCP 服务器获取工具。
        mcp_config: MCP 工具装配契约；None 表示仅装配本地工具。
        async_mode: 是否使用异步并行加载模式；True 时 MCP 工具在后台加载。

    Returns:
        ToolsGateway: 工具网关实例。
    """
    file_tools = _FileSystemToolProvider().build_tools()
    shell_tool = _ShellToolProvider().build_tool()
    local_tools = (*file_tools, shell_tool)

    if async_mode and mcp_config is not None:
        return _create_async_gateway(local_tools, mcp_config)

    return _create_sync_gateway(local_tools, mcp_provider, mcp_config)


def _create_sync_gateway(
    local_tools: tuple[ToolDescriptor, ...],
    mcp_provider: _McpToolProvider | None,
    mcp_config: McpToolConfig | None,
) -> ToolsGateway:
    """创建同步模式的 ToolsGateway。"""
    if mcp_provider is None and mcp_config is not None:
        mcp_provider = _McpToolProvider(config_path=mcp_config.config_path)
    mcp_tools = mcp_provider.build_tools() if mcp_provider else ()

    local_names = {t.name for t in local_tools}
    resolved_mcp_tools = _resolve_name_conflicts(mcp_tools, local_names)

    tools_registry = _ToolRegistry.from_tools(*local_tools, *resolved_mcp_tools)
    tools_executor = _ToolExecutor()
    return ToolsGateway(
        local_executor=tools_executor,
        tool_registry=tools_registry,
    )


def _create_async_gateway(
    local_tools: tuple[ToolDescriptor, ...],
    mcp_config: McpToolConfig,
) -> ToolsGateway:
    """创建异步模式的 ToolsGateway。

    本地工具立即注册，MCP 工具在后台异步加载。
    """
    tools_registry = _ToolRegistry.from_tools(*local_tools)
    tools_executor = _ToolExecutor()

    async_provider = _AsyncMcpToolProvider(config_path=mcp_config.config_path)
    async_provider.start_async_loading()

    return ToolsGateway(
        local_executor=tools_executor,
        tool_registry=tools_registry,
        _async_mcp_provider=async_provider,
    )


def _resolve_name_conflicts(
    mcp_tools: tuple[ToolDescriptor, ...],
    local_names: set[str],
) -> tuple[ToolDescriptor, ...]:
    """解决 MCP 工具与本地工具的名称冲突。

    重复的工具名会加上 mcp_ 前缀。
    """
    resolved: list[ToolDescriptor] = []
    for tool in mcp_tools:
        if tool.name in local_names:
            # 加前缀：mcp_{original_name}
            new_name = f"mcp_{tool.name}"
            tool = ToolDescriptor(
                name=new_name,
                description=tool.description,
                parameters=tool.parameters,
                handler=tool.handler,
            )
        resolved.append(tool)
    return tuple(resolved)


__all__ = ["ToolsGateway", "create_gateway"]
