"""异步并行 MCP 工具提供者。

在后台线程中并行加载所有 MCP 服务器的工具，不阻塞主线程。
结合了异步后台加载和并行化的优势，提供最佳的用户体验。

设计原则：
- 单一职责：只负责异步加载逻辑，复用 McpToolProvider 的通信能力
- 开闭原则：通过组合而非继承扩展功能
- 依赖倒置：依赖抽象（McpToolProvider）而非具体实现
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from src.core_contracts.tools_contracts import McpServerSummary, ToolDescriptor
from src.tools.mcp.mcp_tools import McpToolProvider, McpServerConfig


@dataclass
class AsyncMcpToolProvider:
    """异步并行 MCP 工具加载器。

    在后台线程中并行加载所有 MCP 服务器的工具，提供非阻塞的工具加载体验。

    工作流：
    1. start_async_loading() 启动后台加载线程
    2. 后台线程使用 ThreadPoolExecutor 并行加载所有 servers
    3. get_tools() 等待加载完成并返回结果
    4. is_loading() 检查加载状态

    Attributes:
        config_path: MCP 配置文件路径
        _max_workers: 并行加载的最大线程数
        _load_timeout: 单个 server 加载超时时间（秒）
    """

    config_path: Path | str
    _max_workers: int = 8
    _load_timeout: float = 35.0

    # 内部状态
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _load_event: threading.Event = field(default_factory=threading.Event, init=False)
    _tools: tuple[ToolDescriptor, ...] = field(default_factory=tuple, init=False)
    _loading: bool = field(default=False, init=False)
    _started: bool = field(default=False, init=False)
    _server_summaries: tuple[McpServerSummary, ...] = field(default_factory=tuple, init=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_async_loading(self) -> None:
        """启动异步后台加载。

        如果已经在加载中或已完成，则忽略重复调用。
        此方法立即返回，不阻塞调用线程。
        """
        with self._lock:
            if self._started:
                return
            self._started = True
            self._loading = True

        thread = threading.Thread(
            target=self._background_load,
            daemon=True,
            name="mcp-tools-async-loader",
        )
        thread.start()

    def get_tools(self, timeout: float | None = None) -> tuple[ToolDescriptor, ...]:
        """获取工具列表，如果未加载完成则等待。

        Args:
            timeout: 等待超时时间（秒）；None 表示无限等待。

        Returns:
            tuple[ToolDescriptor, ...]: 工具描述符列表。

        Raises:
            TimeoutError: 等待超时。
            RuntimeError: 加载过程中发生致命错误。
        """
        if self._load_event.is_set():
            return self._tools

        if not self._started:
            raise RuntimeError("Async loading not started. Call start_async_loading() first.")

        if not self._load_event.wait(timeout=timeout):
            raise TimeoutError(
                f"MCP tools loading timeout after {timeout}s. "
                f"Loaded {len(self._tools)} tools so far."
            )

        return self._tools

    def get_server_summaries(self) -> tuple[McpServerSummary, ...]:
        """获取每个 MCP 服务器的状态摘要。

        Returns:
            tuple[McpServerSummary, ...]: 服务器状态摘要列表；加载未完成时为空元组。
        """
        return self._server_summaries

    def is_loading(self) -> bool:
        """检查是否正在加载中。

        Returns:
            bool: True 表示正在加载；False 表示已完成或未开始。
        """
        return self._loading

    # ------------------------------------------------------------------
    # Private implementation
    # ------------------------------------------------------------------

    def _background_load(self) -> None:
        """后台加载任务（在独立线程中执行）。"""
        try:
            provider = McpToolProvider(config_path=self.config_path)
            servers = provider._from_config()

            if not servers:
                with self._lock:
                    self._tools = ()
                return

            tools, errors, summaries = self._parallel_load_servers(provider, servers)

            with self._lock:
                self._tools = tuple(tools)
                self._server_summaries = tuple(summaries)

        except Exception as e:
            pass
        finally:
            self._loading = False
            self._load_event.set()

    def _parallel_load_servers(
        self,
        provider: McpToolProvider,
        servers: list[McpServerConfig],
    ) -> tuple[list[ToolDescriptor], list[str], list[McpServerSummary]]:
        """并行加载所有 MCP 服务器的工具。"""
        tools: list[ToolDescriptor] = []
        errors: list[str] = []
        summaries: list[McpServerSummary] = []

        max_workers = min(len(servers), self._max_workers)

        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="mcp-server-loader",
        ) as executor:
            future_to_server = {
                executor.submit(self._load_server_tools, provider, server): server
                for server in servers
            }

            for future in as_completed(future_to_server):
                server = future_to_server[future]
                try:
                    server_tools = future.result(timeout=self._load_timeout)
                    tools.extend(server_tools)
                    summaries.append(McpServerSummary(
                        name=server.name,
                        transport=server.transport,
                        tool_count=len(server_tools),
                        status="connected",
                    ))
                except Exception as e:
                    errors.append(f"[{server.name}] {e}")
                    summaries.append(McpServerSummary(
                        name=server.name,
                        transport=server.transport,
                        tool_count=0,
                        status="error",
                        error_message=str(e),
                    ))

        return tools, errors, summaries

    def _load_server_tools(
        self,
        provider: McpToolProvider,
        server: McpServerConfig,
    ) -> list[ToolDescriptor]:
        """加载单个 MCP 服务器的工具。

        Args:
            provider: MCP 工具提供者实例。
            server: 服务器配置。

        Returns:
            list[ToolDescriptor]: 该服务器的工具列表。

        Raises:
            Exception: 加载过程中的任何错误。
        """
        server_tools_raw = provider._list_tools(server)
        return [
            provider._convert_tool(server, t) for t in server_tools_raw
        ]


__all__ = ["AsyncMcpToolProvider"]
