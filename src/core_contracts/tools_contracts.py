"""Tools 模块最小跨边界契约。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonDict = dict[str, JsonValue]


@dataclass(frozen=True)
class ToolExecutionRequest:
    """工具执行请求。"""

    tool_name: str
    arguments: JsonDict = field(default_factory=dict)
    runtime: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolExecutionResult:
    """工具执行结果。"""

    name: str
    ok: bool
    content: str
    metadata: JsonDict = field(default_factory=dict)


ToolHandler = Callable[[ToolExecutionRequest], ToolExecutionResult]


@dataclass(frozen=True)
class ToolDescriptor:
    """单个工具定义。"""

    name: str
    description: str
    parameters: JsonDict
    handler: ToolHandler
    server_name: str = ""

    def to_openai_tool(self) -> JsonDict:
        """转换为 OpenAI function-calling 工具声明。
        Args:
            None
        Returns:
            JsonDict: OpenAI 兼容工具 schema。
        Raises:
            None
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass(frozen=True)
class McpToolConfig:
    """MCP 工具装配契约。"""

    config_path: str = ".nexus/mcp.json"


@dataclass(frozen=True)
class McpRequest:
    """MCP 查询契约。

    资源查询和能力查询共享同一组过滤参数，统一使用该契约。
    """

    query: str | None = None
    server_name: str | None = None
    limit: int = 100


@dataclass(frozen=True)
class McpServerSummary:
    """MCP server status summary."""

    name: str
    transport: str
    tool_count: int
    status: str  # "connected" | "error"
    error_message: str = ""
