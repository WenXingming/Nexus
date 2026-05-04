"""Tools 模块对外装配入口。"""

from __future__ import annotations

from src.core_contracts.tools_contracts import McpToolConfig, ToolDescriptor
from src.tools.tool_executor import ToolExecutor
from src.tools.local.filesystem_tools import FileSystemToolProvider
from src.tools.local.shell_tools import ShellToolProvider
from src.tools.mcp import McpToolProvider
from src.tools.tool_registry import ToolRegistry
from src.tools.tools_gateway import ToolsGateway


def create_gateway(
    mcp_provider: McpToolProvider | None = None,
    *,
    mcp_config: McpToolConfig | None = None,
) -> ToolsGateway:
    """函数式门面工厂。

    Args:
        mcp_provider: 可选的 MCP 工具提供者，用于从 MCP 服务器获取工具。
        mcp_config: MCP 工具装配契约；None 表示仅装配本地工具。

    Returns:
        ToolsGateway: 工具网关实例。
    """
    file_tools = FileSystemToolProvider().build_tools()
    shell_tool = ShellToolProvider().build_tool()
    local_tools = (*file_tools, shell_tool)

    if mcp_provider is None and mcp_config is not None:
        mcp_provider = McpToolProvider(config_path=mcp_config.config_path)
    mcp_tools = mcp_provider.build_tools() if mcp_provider else ()

    # 解决名称冲突：MCP 工具名与本地工具名重复时加前缀
    local_names = {t.name for t in local_tools}
    resolved_mcp_tools = _resolve_name_conflicts(mcp_tools, local_names)

    tools_registry = ToolRegistry.from_tools(*local_tools, *resolved_mcp_tools)
    tools_executor = ToolExecutor()
    return ToolsGateway(
        local_executor=tools_executor,
        tool_registry=tools_registry,
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


__all__ = ["McpToolProvider", "ToolsGateway", "create_gateway"]
