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
from src.tools.local.context import ToolRuntimeContext


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
        command = self._require_string(request.arguments, "command")

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
        output = self._truncate_output(rendered, runtime.max_output_chars)
        return ToolExecutionResult(
            name=request.tool_name,
            ok=True,
            content=output,
            metadata={
                "action": "bash",
                "command": command,
                "exit_code": int(completed.returncode),
            },
        )

    def _require_string(self, arguments: JsonDict, key: str) -> str:
        value = arguments.get(key)
        if not isinstance(value, str):
            raise ValueError(f'Argument "{key}" must be a string')
        return value

    def _render_output(self, stdout: str, stderr: str, exit_code: int) -> str:
        lines = [
            f"exit_code={exit_code}",
            "[stdout]",
            stdout.rstrip(),
            "[stderr]",
            stderr.rstrip(),
        ]
        return "\n".join(lines).strip()

    def _truncate_output(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        half = max(1, limit // 2)
        return f"{text[:half]}\n...[output truncated, total {len(text)} chars]...\n{text[-half:]}"
