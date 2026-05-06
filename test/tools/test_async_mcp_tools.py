"""AsyncMcpToolProvider 单元测试。"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core_contracts.tools_contracts import (
    McpToolConfig,
    ToolDescriptor,
)
from src.tools import create_gateway
from src.tools.mcp.async_mcp_tools import AsyncMcpToolProvider
from src.tools.tool_registry import ToolRegistry
from src.tools.tools_gateway import ToolsGateway


def _make_descriptor(name: str) -> ToolDescriptor:
    """创建测试用的 ToolDescriptor。"""
    return ToolDescriptor(
        name=name,
        description=f"Tool: {name}",
        parameters={},
        handler=MagicMock(),
    )


class TestAsyncMcpToolProvider:
    """验证 AsyncMcpToolProvider 的核心行为。"""

    def test_get_tools_raises_if_not_started(self) -> None:
        """未启动时调用 get_tools 应抛出 RuntimeError。"""
        provider = AsyncMcpToolProvider(config_path="nonexistent.json")

        with pytest.raises(RuntimeError, match="Async loading not started"):
            provider.get_tools()

    def test_get_tools_timeout(self) -> None:
        """等待超时时应抛出 TimeoutError。"""
        provider = AsyncMcpToolProvider(config_path="nonexistent.json")

        # 模拟一个不会完成的后台加载
        def slow_load():
            time.sleep(10)
            provider._load_event.set()

        provider._started = True
        provider._loading = True
        thread = threading.Thread(target=slow_load, daemon=True)
        thread.start()

        with pytest.raises(TimeoutError, match="MCP tools loading timeout"):
            provider.get_tools(timeout=0.1)

    def test_get_tools_returns_loaded_tools(self) -> None:
        """加载完成后，get_tools 应返回工具列表。"""
        provider = AsyncMcpToolProvider(config_path="nonexistent.json")
        expected_tools = (_make_descriptor("tool1"), _make_descriptor("tool2"))

        # 模拟加载完成
        provider._tools = expected_tools
        provider._load_event.set()

        result = provider.get_tools()
        assert result == expected_tools

    def test_is_loading_returns_false_when_not_started(self) -> None:
        """未启动时，is_loading 应返回 False。"""
        provider = AsyncMcpToolProvider(config_path="nonexistent.json")

        assert provider.is_loading() is False

    def test_is_loading_returns_true_when_loading(self) -> None:
        """加载中时，is_loading 应返回 True。"""
        provider = AsyncMcpToolProvider(config_path="nonexistent.json")
        provider._started = True
        provider._loading = True

        assert provider.is_loading() is True

    def test_is_loading_returns_false_when_completed(self) -> None:
        """加载完成后，is_loading 应返回 False。"""
        provider = AsyncMcpToolProvider(config_path="nonexistent.json")
        provider._started = True
        provider._loading = False
        provider._load_event.set()

        assert provider.is_loading() is False

    def test_background_load_handles_no_config_file(self) -> None:
        """配置文件不存在时，应返回空工具列表。"""
        provider = AsyncMcpToolProvider(config_path="nonexistent.json")

        provider._background_load()

        assert provider._tools == ()
        assert provider._load_event.is_set()
        assert provider._loading is False

    @patch('src.tools.mcp.mcp_tools.McpToolProvider._from_config')
    def test_background_load_parallel_execution(self, mock_from_config) -> None:
        """后台加载应并行执行所有 server 的加载。"""
        # 模拟 3 个 servers
        server1 = MagicMock(name="server1")
        server2 = MagicMock(name="server2")
        server3 = MagicMock(name="server3")
        mock_from_config.return_value = [server1, server2, server3]

        provider = AsyncMcpToolProvider(config_path="test.json", _max_workers=3)

        # 模拟每个 server 的加载
        def mock_load_server_tools(provider, server):
            time.sleep(0.1)  # 模拟加载时间
            return [_make_descriptor(f"tool_{server.name}")]

        with patch.object(provider, '_load_server_tools', side_effect=mock_load_server_tools):
            provider._background_load()

        assert len(provider._tools) == 3
        assert provider._load_event.is_set()


class TestToolsGatewayAsync:
    """验证 ToolsGateway 的异步加载集成。"""

    def test_is_mcp_loading_returns_false_without_async_provider(self) -> None:
        """没有异步 provider 时，is_mcp_loading 应返回 False。"""
        gateway = ToolsGateway(
            local_executor=MagicMock(),
            tool_registry=ToolRegistry.from_tools(),
        )

        assert gateway.is_mcp_loading() is False

    def test_is_mcp_loading_returns_true_when_loading(self) -> None:
        """异步加载中时，is_mcp_loading 应返回 True。"""
        mock_provider = MagicMock()
        mock_provider.is_loading.return_value = True

        gateway = ToolsGateway(
            local_executor=MagicMock(),
            tool_registry=ToolRegistry.from_tools(),
            _async_mcp_provider=mock_provider,
        )

        assert gateway.is_mcp_loading() is True

    def test_list_tools_merges_async_tools(self) -> None:
        """list_tools 应自动合并异步加载的工具。"""
        local_tool = _make_descriptor("local_tool")
        mcp_tool = _make_descriptor("mcp_tool")

        mock_provider = MagicMock()
        mock_provider.is_loading.return_value = False
        mock_provider._load_event = MagicMock()
        mock_provider._load_event.is_set.return_value = True
        mock_provider.get_tools.return_value = (mcp_tool,)

        registry = ToolRegistry.from_tools(local_tool)
        gateway = ToolsGateway(
            local_executor=MagicMock(),
            tool_registry=registry,
            _async_mcp_provider=mock_provider,
        )

        result = gateway.list_tools()

        assert len(result) == 2
        assert any(t.name == "local_tool" for t in result)
        assert any(t.name == "mcp_tool" for t in result)


class TestCreateGatewayAsyncMode:
    """验证 create_gateway 的异步模式。"""

    def test_create_gateway_async_mode(self) -> None:
        """异步模式应创建 AsyncMcpToolProvider 并启动加载。"""
        with patch('src.tools.AsyncMcpToolProvider') as mock_provider_cls, \
            patch('src.tools.FileSystemToolProvider') as fs_provider, \
            patch('src.tools.ShellToolProvider') as shell_provider:

            fs_provider.return_value.build_tools.return_value = ()
            shell_provider.return_value.build_tool.return_value = _make_descriptor('bash')
            mock_provider = MagicMock()
            mock_provider_cls.return_value = mock_provider

            gateway = create_gateway(
                mcp_config=McpToolConfig(config_path='test.json'),
                async_mode=True,
            )

        mock_provider_cls.assert_called_once_with(config_path='test.json')
        mock_provider.start_async_loading.assert_called_once()
        assert gateway._async_mcp_provider is mock_provider

    def test_create_gateway_sync_mode_default(self) -> None:
        """默认模式应使用同步加载。"""
        with patch('src.tools.McpToolProvider') as mcp_provider_cls, \
            patch('src.tools.FileSystemToolProvider') as fs_provider, \
            patch('src.tools.ShellToolProvider') as shell_provider:

            fs_provider.return_value.build_tools.return_value = ()
            shell_provider.return_value.build_tool.return_value = _make_descriptor('bash')
            mcp_provider_cls.return_value.build_tools.return_value = ()

            gateway = create_gateway(mcp_config=McpToolConfig(config_path='test.json'))

        assert gateway._async_mcp_provider is None
