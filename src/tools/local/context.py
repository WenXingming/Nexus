"""本地工具执行时的内部运行时上下文。

该模块仅供 local/ 子包内部使用，不对外暴露。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.core_contracts.tools_contracts import JsonDict


def truncate_output(text: str, limit: int) -> str:
    """Truncate text to limit, preserving head and tail."""
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return f"{text[:half]}\n...[output truncated, total {len(text)} chars]...\n{text[-half:]}"


def require_string(arguments: JsonDict, key: str) -> str:
    """Extract a required string argument, raising ValueError on missing or non-string."""
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ValueError(f'Argument "{key}" must be a string')
    return value


@dataclass(frozen=True)
class ToolRuntimeContext:
    """工具执行时的运行时上下文，由 ToolExecutionRequest.runtime 反序列化而来。"""

    root: Path
    command_timeout_seconds: float
    max_output_chars: int
    allow_file_write: bool = False
    allow_shell_commands: bool = False
    allow_destructive_shell_commands: bool = False
    safe_env: dict[str, str] = field(default_factory=dict)

    # ---- Schema 常量：单一真相源 ----
    _KEY_ROOT = "root"
    _KEY_TIMEOUT = "command_timeout_seconds"
    _KEY_MAX_OUTPUT = "max_output_chars"
    _KEY_ALLOW_WRITE = "allow_file_write"
    _KEY_ALLOW_SHELL = "allow_shell_commands"
    _KEY_ALLOW_DESTRUCTIVE = "allow_destructive_shell_commands"
    _KEY_SAFE_ENV = "safe_env"

    @staticmethod
    def build_payload(
        workspace_root: Path,
        command_timeout_seconds: float,
        max_output_chars: int,
        *,
        allow_file_write: bool = False,
        allow_shell_commands: bool = False,
        allow_destructive_shell_commands: bool = False,
        safe_env: dict[str, str] | None = None,
    ) -> JsonDict:
        """序列化：供外部调用者生成运行时字典。

        Args:
            workspace_root: 工作区根目录。
            command_timeout_seconds: Shell 命令超时秒数。
            max_output_chars: 工具输出最大字符数。
            allow_file_write: 是否允许写文件。
            allow_shell_commands: 是否允许执行 Shell 命令。
            allow_destructive_shell_commands: 是否允许破坏性 Shell 命令。
            safe_env: 注入的安全环境变量。

        Returns:
            JsonDict: 校验后的运行时参数字典。

        Raises:
            ValueError: 当 timeout 或 max_output 非正数时抛出。
        """
        if command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds 必须大于 0。")
        if max_output_chars <= 0:
            raise ValueError("max_output_chars 必须大于 0。")
        return {
            ToolRuntimeContext._KEY_ROOT: str(workspace_root.resolve()),
            ToolRuntimeContext._KEY_TIMEOUT: float(command_timeout_seconds),
            ToolRuntimeContext._KEY_MAX_OUTPUT: max_output_chars,
            ToolRuntimeContext._KEY_ALLOW_WRITE: allow_file_write,
            ToolRuntimeContext._KEY_ALLOW_SHELL: allow_shell_commands,
            ToolRuntimeContext._KEY_ALLOW_DESTRUCTIVE: allow_destructive_shell_commands,
            ToolRuntimeContext._KEY_SAFE_ENV: dict(safe_env or {}),
        }

    @staticmethod
    def from_payload(payload: JsonDict) -> "ToolRuntimeContext":
        """反序列化：从运行时字典构建强类型上下文。

        Args:
            payload: 请求中携带的运行时参数字典。

        Returns:
            ToolRuntimeContext: 校验并解析后的类型化上下文。

        Raises:
            ValueError: 当参数类型或取值范围不合法时抛出。
        """
        K = ToolRuntimeContext
        root_value = payload.get(K._KEY_ROOT, ".")
        timeout_value = payload.get(K._KEY_TIMEOUT, 30.0)
        max_output_value = payload.get(K._KEY_MAX_OUTPUT, 12000)
        allow_file_write = bool(payload.get(K._KEY_ALLOW_WRITE, False))
        allow_shell_commands = bool(payload.get(K._KEY_ALLOW_SHELL, False))
        allow_destructive = bool(payload.get(K._KEY_ALLOW_DESTRUCTIVE, False))
        safe_env_value = payload.get(K._KEY_SAFE_ENV, {})

        if not isinstance(root_value, str) or not root_value.strip():
            raise ValueError("runtime.root 必须为非空字符串。")
        if not isinstance(timeout_value, (int, float)) or timeout_value <= 0:
            raise ValueError("runtime.command_timeout_seconds 必须大于 0。")
        if not isinstance(max_output_value, int) or max_output_value <= 0:
            raise ValueError("runtime.max_output_chars 必须为正整数。")
        if not isinstance(safe_env_value, dict):
            raise ValueError("runtime.safe_env 必须为字典。")

        safe_env: dict[str, str] = {}
        for key, value in safe_env_value.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("runtime.safe_env 的键和值都必须是字符串。")
            safe_env[key] = value

        return ToolRuntimeContext(
            root=Path(root_value).resolve(),
            command_timeout_seconds=float(timeout_value),
            max_output_chars=max_output_value,
            allow_file_write=allow_file_write,
            allow_shell_commands=allow_shell_commands,
            allow_destructive_shell_commands=allow_destructive,
            safe_env=safe_env,
        )
