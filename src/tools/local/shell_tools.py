"""本地 shell 工具集合。"""

from __future__ import annotations

import os
import subprocess

from src.core_contracts.tools_contracts import (
    JsonDict,
    ToolDescriptor,
    ToolExecutionRequest,
    ToolExecutionResult,
)
from src.tools.local.context import ToolRuntimeContext, truncate_output, require_string


class ShellToolProvider:
    """Shell 工具提供者。"""

    def build_tool(self) -> ToolDescriptor:
        """构建 bash 工具定义。"""
        return ToolDescriptor(
            name="bash",
            description="在当前工作区执行 shell 命令。",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
            handler=self._run,
        )

    def _run(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        runtime = ToolRuntimeContext.from_payload(request.runtime)
        command = require_string(request.arguments, "command")
        self._require_shell_permission(runtime, command)

        environment = dict(os.environ)
        environment.update(runtime.safe_env)
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=runtime.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=runtime.command_timeout_seconds,
                env=environment,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Shell command timed out after {runtime.command_timeout_seconds} seconds") from exc

        rendered = self._render_output(completed.stdout or "", completed.stderr or "", int(completed.returncode))
        output = truncate_output(rendered, runtime.max_output_chars)
        return ToolExecutionResult(
            name=request.tool_name,
            ok=int(completed.returncode) == 0,
            content=output,
            metadata={
                "action": "bash",
                "command": command,
                "exit_code": int(completed.returncode),
            },
        )

    def _render_output(self, stdout: str, stderr: str, exit_code: int) -> str:
        lines = [
            f"exit_code={exit_code}",
            "[stdout]",
            stdout.rstrip(),
            "[stderr]",
            stderr.rstrip(),
        ]
        return "\n".join(lines).strip()

    def _require_shell_permission(self, runtime: ToolRuntimeContext, command: str) -> None:
        if not runtime.allow_shell_commands:
            raise PermissionError("Shell command permission is not enabled.")
        if not runtime.allow_destructive_shell_commands and self._looks_destructive(command):
            raise PermissionError("Destructive shell command permission is not enabled.")

    def _looks_destructive(self, command: str) -> bool:
        normalized = " ".join(command.lower().split())
        if "remove-item" in normalized and "-recurse" in normalized:
            return True
        destructive_markers = (
            "rm -rf",
            "del /s",
            "rmdir /s",
            "rd /s",
            "format ",
        )
        return any(marker in normalized for marker in destructive_markers)
