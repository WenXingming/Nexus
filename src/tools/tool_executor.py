"""Tools 执行器。"""

from __future__ import annotations

from dataclasses import dataclass

from src.core_contracts.tools_contracts import (
    ToolExecutionRequest,
    ToolExecutionResult,
)
from src.tools.tool_registry import ToolRegistry


def _missing_required(parameters: dict, arguments: dict) -> list[str]:
    """校验 required 参数是否全部存在。"""
    required: list[str] = parameters.get("required", [])
    return [key for key in required if key not in arguments]


def _describe_missing(parameters: dict, missing: list[str]) -> str:
    """为缺失的必填参数生成带类型和描述的可读说明。"""
    properties = parameters.get("properties", {})
    lines: list[str] = []
    for name in missing:
        prop = properties.get(name, {})
        ptype = prop.get("type", "string")
        desc = prop.get("description", "")
        detail = f"{name} ({ptype})"
        if desc:
            detail += f": {desc}"
        lines.append(f"  - {detail}")
    return "\n".join(lines)


@dataclass(frozen=True)
class ToolExecutor:
    """工具执行调度器。"""

    def execute(self, tool_registry: ToolRegistry, request: ToolExecutionRequest) -> ToolExecutionResult:
        """执行一次工具调用。"""
        tool = tool_registry.get(request.tool_name)
        if tool is None:
            return self._unknown_tool_result(request.tool_name)

        missing = _missing_required(tool.parameters, request.arguments)
        if missing:
            detail = _describe_missing(tool.parameters, missing)
            return self._failure_result(
                name=request.tool_name,
                exc=ValueError(f"Missing required arguments: {', '.join(missing)}.\n{detail}\nPlease retry with the required arguments."),
                error_kind="missing_arguments",
            )

        try:
            return tool.handler(request)
        except PermissionError as exc:
            return self._failure_result(name=request.tool_name, exc=exc, error_kind="permission_denied")
        except (ValueError, RuntimeError) as exc:
            return self._failure_result(name=request.tool_name, exc=exc, error_kind="tool_execution_error")
        except Exception as exc:  # noqa: BLE001
            return self._failure_result(name=request.tool_name, exc=exc, error_kind="unexpected_error")

    @staticmethod
    def _failure_result(name: str, exc: BaseException, *, error_kind: str) -> ToolExecutionResult:
        """将异常封装为失败结果。
        Args:
            name (str): 工具名。
            exc (BaseException): 捕获异常。
            error_kind (str): 错误分类。
        Returns:
            ToolExecutionResult: 失败结果对象。
        Raises:
            None
        """
        return ToolExecutionResult(
            name=name,
            ok=False,
            content=str(exc),
            metadata={"error_kind": error_kind},
        )

    @staticmethod
    def _unknown_tool_result(name: str) -> ToolExecutionResult:
        """构建未知工具错误结果。
        Args:
            name (str): 工具名。
        Returns:
            ToolExecutionResult: 未知工具结果对象。
        Raises:
            None
        """
        return ToolExecutionResult(
            name=name,
            ok=False,
            content=f"Unknown tool: {name}",
            metadata={"error_kind": "unknown_tool"},
        )
