"""本地文件系统工具集合。"""

from __future__ import annotations

from pathlib import Path

from src.core_contracts.tools_contracts import (
    JsonDict,
    ToolDescriptor,
    ToolExecutionRequest,
    ToolExecutionResult,
)
from src.tools.local.context import ToolRuntimeContext, truncate_output, require_string


class FileSystemToolProvider:
    """文件系统工具提供者。"""

    def build_tools(self) -> tuple[ToolDescriptor, ...]:
        """构建文件系统工具定义。"""
        return (
            ToolDescriptor(
                name="list_dir",
                description="列出工作区目录下的文件和子目录。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "要列出的目录路径，相对于工作区根目录。默认为 '.'（当前目录）。",
                        },
                        "max_entries": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                            "description": "最多返回的条目数量，范围 1-500，默认 200。",
                        },
                    },
                },
                handler=self._list_dir,
            ),
            ToolDescriptor(
                name="read_file",
                description="读取工作区内文本文件，可选按行区间截取。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "要读取的文件路径，相对于工作区根目录。",
                        },
                        "start_line": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "起始行号（从 1 开始），不指定则从文件开头读取。",
                        },
                        "end_line": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "结束行号（含），不指定则读到文件末尾。",
                        },
                    },
                    "required": ["path"],
                },
                handler=self._read_file,
            ),
            ToolDescriptor(
                name="write_file",
                description="写入工作区文件，不存在时会自动创建父目录。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "文件路径，相对于工作区根目录。例如: output.md",
                        },
                        "content": {
                            "type": "string",
                            "description": "要写入文件的完整文本内容。",
                        },
                    },
                    "required": ["path", "content"],
                },
                handler=self._write_file,
            ),
            ToolDescriptor(
                name="edit_file",
                description="在工作区文件内替换精确文本，默认只替换首个匹配。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "要编辑的文件路径，相对于工作区根目录。",
                        },
                        "old_text": {
                            "type": "string",
                            "description": "要被替换的原始文本，必须精确匹配文件内容。",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "用于替换的新文本。",
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "是否替换所有匹配项，默认为 false（仅替换首个）。",
                        },
                    },
                    "required": ["path", "old_text", "new_text"],
                },
                handler=self._edit_file,
            ),
        )

    def _list_dir(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        runtime = ToolRuntimeContext.from_payload(request.runtime)
        raw_path = self._get_string(request.arguments, "path", default=".")
        max_entries = self._get_int(request.arguments, "max_entries", default=200, min_value=1, max_value=500)
        target = self._resolve_workspace_path(runtime, raw_path, must_exist=True, expect_dir=True)

        children = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        entries = [f"- {child.name}{'/' if child.is_dir() else ''}" for child in children[:max_entries]]
        rel = self._to_relative_display(target, runtime.root)
        lines = [f"# list_dir: {rel}", ""]
        lines.extend(entries or ["(empty)"])
        if len(children) > max_entries:
            lines.extend(["", f"... omitted {len(children) - max_entries} entries"])
        content = "\n".join(lines)
        output = truncate_output(content, runtime.max_output_chars)
        return ToolExecutionResult(
            name=request.tool_name,
            ok=True,
            content=output,
            metadata={
                "action": "list_dir",
                "path": rel,
                "entry_count": len(children),
                "returned_entries": len(entries),
            },
        )

    def _read_file(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        runtime = ToolRuntimeContext.from_payload(request.runtime)
        raw_path = require_string(request.arguments, "path")
        start_line = self._get_optional_int(request.arguments, "start_line", min_value=1)
        end_line = self._get_optional_int(request.arguments, "end_line", min_value=1)
        if start_line is not None and end_line is not None and end_line < start_line:
            raise ValueError("end_line must be greater than or equal to start_line")

        target = self._resolve_workspace_path(runtime, raw_path, must_exist=True, expect_file=True)
        text = target.read_text(encoding="utf-8")
        if start_line is not None or end_line is not None:
            lines = text.splitlines(keepends=True)
            start = start_line or 1
            end = end_line or len(lines)
            text = "".join(lines[start - 1 : end])

        output = truncate_output(text, runtime.max_output_chars)
        return ToolExecutionResult(
            name=request.tool_name,
            ok=True,
            content=output,
            metadata={"action": "read_file", "path": self._to_relative_display(target, runtime.root)},
        )

    def _write_file(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        runtime = ToolRuntimeContext.from_payload(request.runtime)
        raw_path = require_string(request.arguments, "path")
        content = require_string(request.arguments, "content")
        target = self._resolve_workspace_path(runtime, raw_path, must_exist=False)
        if target.exists() and target.is_dir():
            raise ValueError(f"Path points to a directory, not a file: {raw_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolExecutionResult(
            name=request.tool_name,
            ok=True,
            content=f"Wrote {self._to_relative_display(target, runtime.root)} ({len(content)} chars).",
            metadata={"action": "write_file"},
        )

    def _edit_file(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        runtime = ToolRuntimeContext.from_payload(request.runtime)
        raw_path = require_string(request.arguments, "path")
        old_text = require_string(request.arguments, "old_text")
        new_text = require_string(request.arguments, "new_text")
        replace_all = self._get_bool(request.arguments, "replace_all", default=False)
        if not old_text:
            raise ValueError("old_text cannot be empty")

        target = self._resolve_workspace_path(runtime, raw_path, must_exist=True, expect_file=True)
        original = target.read_text(encoding="utf-8")
        if old_text not in original:
            raise ValueError("old_text not found in target file")
        if replace_all:
            replaced_count = original.count(old_text)
            updated = original.replace(old_text, new_text)
        else:
            replaced_count = 1
            updated = original.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")

        return ToolExecutionResult(
            name=request.tool_name,
            ok=True,
            content=f"Edited {self._to_relative_display(target, runtime.root)}, replaced {replaced_count} occurrence(s).",
            metadata={"action": "edit_file", "replaced_count": replaced_count},
        )

    def _resolve_workspace_path(
        self,
        runtime: ToolRuntimeContext,
        raw_path: str,
        *,
        must_exist: bool,
        expect_file: bool = False,
        expect_dir: bool = False,
    ) -> Path:
        candidate = Path(raw_path)
        resolved = candidate.resolve() if candidate.is_absolute() else (runtime.root / candidate).resolve()
        try:
            resolved.relative_to(runtime.root)
        except ValueError as exc:
            raise ValueError(f"Path escapes workspace root: {raw_path}") from exc
        if must_exist and not resolved.exists():
            raise ValueError(f"Path does not exist: {raw_path}")
        if expect_file and resolved.exists() and not resolved.is_file():
            raise ValueError(f"Path is not a file: {raw_path}")
        if expect_dir and resolved.exists() and not resolved.is_dir():
            raise ValueError(f"Path is not a directory: {raw_path}")
        return resolved

    def _to_relative_display(self, path: Path, root: Path) -> str:
        try:
            relative = path.relative_to(root)
        except ValueError:
            return str(path)
        text = str(relative)
        return text if text else "."

    def _get_string(self, arguments: JsonDict, key: str, *, default: str) -> str:
        value = arguments.get(key, default)
        if not isinstance(value, str):
            raise ValueError(f'Argument "{key}" must be a string')
        return value

    def _get_bool(self, arguments: JsonDict, key: str, *, default: bool) -> bool:
        value = arguments.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f'Argument "{key}" must be a boolean')
        return value

    def _get_int(
        self,
        arguments: JsonDict,
        key: str,
        *,
        default: int,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> int:
        value = arguments.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f'Argument "{key}" must be an integer')
        if min_value is not None and value < min_value:
            raise ValueError(f'Argument "{key}" must be >= {min_value}')
        if max_value is not None and value > max_value:
            raise ValueError(f'Argument "{key}" must be <= {max_value}')
        return value

    def _get_optional_int(self, arguments: JsonDict, key: str, *, min_value: int | None = None) -> int | None:
        value = arguments.get(key)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f'Argument "{key}" must be an integer')
        if min_value is not None and value < min_value:
            raise ValueError(f'Argument "{key}" must be >= {min_value}')
        return value
