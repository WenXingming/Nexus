"""ToolsGateway 单元测试。"""

from unittest.mock import MagicMock, patch

from src.core_contracts.tools_contracts import (
    McpToolConfig,
    ToolDescriptor,
    ToolExecutionRequest,
    ToolExecutionResult,
)
from src.tools import create_gateway
from src.tools.tool_registry import ToolRegistry
from src.tools.tools_gateway import ToolsGateway


def _make_descriptor(name: str) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        description=f"Tool: {name}",
        parameters={},
        handler=MagicMock(),
    )


def _make_gateway(*tool_names: str) -> ToolsGateway:
    tools = tuple(_make_descriptor(n) for n in tool_names)
    return ToolsGateway(
        local_executor=MagicMock(),
        tool_registry=ToolRegistry.from_tools(*tools),
    )


class TestToolsGateway:
    """验证 tools 门面的核心接口行为。"""

    # -------------------------------------------------------------------------
    # list_tools
    # -------------------------------------------------------------------------

    def test_list_tools_returns_all_registered_descriptors(self) -> None:
        gateway = _make_gateway("list_dir", "bash", "read_file")

        result = gateway.list_tools()

        assert len(result) == 3
        assert [t.name for t in result] == ["list_dir", "bash", "read_file"]

    def test_list_tools_returns_empty_list_when_no_tools(self) -> None:
        gateway = _make_gateway()

        result = gateway.list_tools()

        assert result == []

    def test_list_tools_returns_descriptor_instances(self) -> None:
        gateway = _make_gateway("bash")

        result = gateway.list_tools()

        assert isinstance(result[0], ToolDescriptor)
        assert result[0].name == "bash"

    # -------------------------------------------------------------------------
    # execute_tool
    # -------------------------------------------------------------------------

    def test_execute_tool_delegates_to_executor(self) -> None:
        gateway = _make_gateway("list_dir")
        request = ToolExecutionRequest(tool_name="list_dir", arguments={})
        expected = ToolExecutionResult(name="list_dir", ok=True, content="ok")
        gateway.local_executor.execute.return_value = expected

        result = gateway.execute_tool(request)

        gateway.local_executor.execute.assert_called_once_with(gateway.tool_registry, request)
        assert result is expected

    def test_execute_tool_passes_correct_registry(self) -> None:
        gateway = _make_gateway("bash")
        request = ToolExecutionRequest(tool_name="bash", arguments={"command": "ls"})
        gateway.local_executor.execute.return_value = ToolExecutionResult(
            name="bash", ok=True, content="done",
        )

        gateway.execute_tool(request)

        passed_registry = gateway.local_executor.execute.call_args[0][0]
        assert passed_registry is gateway.tool_registry


class TestToolsFactory:
    def test_create_gateway_uses_mcp_contract_config_path(self) -> None:
        with patch('src.tools._FileSystemToolProvider') as fs_provider, \
            patch('src.tools._ShellToolProvider') as shell_provider, \
            patch('src.tools._McpToolProvider') as mcp_provider_cls:
            fs_provider.return_value.build_tools.return_value = ()
            shell_provider.return_value.build_tool.return_value = _make_descriptor('bash')
            mcp_provider_cls.return_value.build_tools.return_value = ()

            create_gateway(mcp_config=McpToolConfig(config_path='custom/mcp.json'))

        mcp_provider_cls.assert_called_once_with(config_path='custom/mcp.json')
