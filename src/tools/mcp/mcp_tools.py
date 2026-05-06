"""MCP 工具提供者。

负责从 MCP 配置文件读取服务器列表，通过 stdio / streamable-http 传输协议
与 MCP 服务器通信，获取工具列表并转换为 ToolDescriptor。
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

from src.core_contracts.tools_contracts import (
    JsonDict,
    ToolDescriptor,
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolHandler,
)


@dataclass
class McpServerConfig:
    """单个 MCP 服务器的解析后配置。"""

    name: str
    transport: str
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None
    url: str | None = None
    api_key: str | None = None


@dataclass
class McpToolProvider:
    """从 MCP 配置文件读取所有服务器，获取并构建工具定义。"""

    config_path: Path | str
    _request_id: count = field(default_factory=lambda: count(1), init=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_tools(self) -> tuple[ToolDescriptor, ...]:
        """读取配置文件中的所有 MCP 服务器，获取工具列表并转换为 ToolDescriptor。"""
        servers = self._from_config()
        tools: list[ToolDescriptor] = []
        for server in servers:
            server_tools = self._list_tools(server)
            tools.extend(self._convert_tool(server, t) for t in server_tools)
        return tuple(tools)

    # ------------------------------------------------------------------
    # Private helpers — depth-first from build_tools
    # ------------------------------------------------------------------

    def _from_config(self) -> list[McpServerConfig]:
        """解析 MCP 配置文件，返回服务器配置列表。"""
        path = Path(self.config_path)
        if not path.exists():
            return []
        try:
            with open(path, encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {e}") from e
        if not isinstance(config, dict):
            raise ValueError("配置文件必须是 JSON 对象")
        servers: list[McpServerConfig] = []
        for name, server_cfg in config.get("mcpServers", {}).items():
            servers.append(
                McpServerConfig(
                    name=name,
                    transport=server_cfg.get("transport", "stdio"),
                    command=server_cfg.get("command"),
                    args=server_cfg.get("args"),
                    env=server_cfg.get("env"),
                    cwd=server_cfg.get("cwd"),
                    url=server_cfg.get("url"),
                    api_key=server_cfg.get("api_key"),
                )
            )
        return servers

    def _list_tools(self, server: McpServerConfig) -> list[JsonDict]:
        """调用 MCP 服务器 tools/list 获取工具列表。"""
        result = self._send_mcp_request(server, "tools/list")
        return list(result.get("tools", []))

    def _send_mcp_request(
        self, server: McpServerConfig, method: str, params: JsonDict | None = None
    ) -> JsonDict:
        """根据传输类型分发 JSON-RPC 请求。"""
        if server.transport == "stdio":
            return self._send_stdio_request(server, method, params)
        return self._send_http_request(server, method, params)

    def _send_stdio_request(
        self, server: McpServerConfig, method: str, params: JsonDict | None = None
    ) -> JsonDict:
        """通过 stdio 子进程发送 JSON-RPC 请求。

        启动子进程后依次发送 initialize 握手和实际请求，返回最后一条响应的 result。
        """
        cmd = [server.command] + (server.args or [])
        env = os.environ.copy()
        if server.env:
            env.update(server.env)

        init_payload = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "nexus", "version": "0.1.0"},
            },
        }
        req_payload = {
            "jsonrpc": "2.0",
            "id": next(self._request_id),
            "method": method,
            "params": params or {},
        }
        input_data = (
            json.dumps(init_payload) + "\n" + json.dumps(req_payload) + "\n"
        )

        try:
            proc = subprocess.run(
                cmd,
                input=input_data,
                capture_output=True,
                text=True,
                cwd=server.cwd or ".",
                env=env,
                timeout=30,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"MCP 命令未找到 ({server.name}): {e}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"MCP 请求超时 ({server.name}): {e}") from e

        lines = [l for l in proc.stdout.strip().split("\n") if l]
        if not lines:
            raise RuntimeError(
                f"MCP stdio 服务器无响应 ({server.name}): {proc.stderr.strip()}"
            )

        try:
            response = json.loads(lines[-1])
        except json.JSONDecodeError as e:
            raise RuntimeError(f"MCP 响应解析失败 ({server.name}): {e}") from e
        if "error" in response:
            err = response["error"]
            raise RuntimeError(
                f"MCP 协议错误 ({server.name}, code={err.get('code')}): {err.get('message', '')}"
            )
        return response.get("result", {})

    def _send_http_request(
        self, server: McpServerConfig, method: str, params: JsonDict | None = None
    ) -> JsonDict:
        """通过 HTTP 发送 JSON-RPC 请求。"""
        payload = {
            "jsonrpc": "2.0",
            "id": next(self._request_id),
            "method": method,
            "params": params or {},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if server.api_key:
            headers["Authorization"] = f"Bearer {server.api_key}"

        req = urllib.request.Request(
            server.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw_data = resp.read().decode("utf-8")

                if "text/event-stream" in content_type:
                    body = self._parse_sse_response(raw_data, server.name)
                else:
                    body = json.loads(raw_data)
        except urllib.error.URLError as e:
            raise RuntimeError(f"MCP HTTP 连接失败 ({server.name}): {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"MCP 响应解析失败 ({server.name}): {e}") from e

        if "error" in body:
            err = body["error"]
            raise RuntimeError(
                f"MCP 协议错误 ({server.name}, code={err.get('code')}): {err.get('message', '')}"
            )
        return body.get("result", {})

    def _parse_sse_response(self, raw_data: str, server_name: str) -> JsonDict:
        """解析 SSE (Server-Sent Events) 响应，提取最后一个 JSON-RPC 结果。

        SSE 格式：
        data: {"jsonrpc": "2.0", "id": 1, "result": {...}}

        """
        last_json = None
        for line in raw_data.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                json_str = line[6:]  # 去掉 "data: " 前缀
                try:
                    last_json = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        if last_json is None:
            raise RuntimeError(f"MCP SSE 响应中未找到有效数据 ({server_name})")
        return last_json

    def _convert_tool(
        self, server: McpServerConfig, tool: JsonDict
    ) -> ToolDescriptor:
        """转换单个 MCP 工具为 ToolDescriptor。"""
        return ToolDescriptor(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters=tool.get("inputSchema", tool.get("parameters", {})),
            handler=self._create_handler(server, tool["name"]),
            server_name=server.name,
        )

    def _create_handler(
        self, server: McpServerConfig, tool_name: str
    ) -> ToolHandler:
        """创建 MCP 工具的 handler，闭包捕获 server 和 tool_name。"""

        def handler(request: ToolExecutionRequest) -> ToolExecutionResult:
            result = self._call_tool(server, tool_name, request.arguments)
            return ToolExecutionResult(
                name=request.tool_name,
                ok=result.get("ok", True),
                content=result.get("content", ""),
                metadata=result.get("metadata", {}),
            )

        return handler

    def _call_tool(
        self, server: McpServerConfig, name: str, arguments: JsonDict
    ) -> JsonDict:
        """调用 MCP 工具并返回标准化的结果字典。"""
        mcp_result = self._send_mcp_request(
            server, "tools/call", {"name": name, "arguments": arguments}
        )
        is_error = mcp_result.get("isError", False)
        content_items = mcp_result.get("content", [])
        text = "".join(
            item.get("text", "")
            for item in content_items
            if isinstance(item, dict) and item.get("type") == "text"
        )
        return {"ok": not is_error, "content": text, "metadata": mcp_result}
