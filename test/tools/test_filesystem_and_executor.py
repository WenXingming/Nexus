"""FileSystemToolProvider 与 ToolExecutor 单元测试。"""

from pathlib import Path

import pytest

from src.core_contracts.tools_contracts import ToolExecutionRequest
from src.tools.tool_executor import ToolExecutor
from src.tools.local.filesystem_tools import FileSystemToolProvider
from src.tools.tool_registry import ToolRegistry


@pytest.fixture
def fs_provider() -> FileSystemToolProvider:
    return FileSystemToolProvider()


@pytest.fixture
def runtime(tmp_path: Path) -> dict[str, object]:
    return {
        "root": str(tmp_path),
        "command_timeout_seconds": 5,
        "max_output_chars": 10000,
    }


class TestFileSystemToolProvider:
    """验证文件系统工具行为边界。"""

    def test_write_and_read_roundtrip(self, fs_provider: FileSystemToolProvider, runtime: dict[str, object]) -> None:
        tools = {tool.name: tool for tool in fs_provider.build_tools()}
        write_result = tools["write_file"].handler(
            ToolExecutionRequest(tool_name="write_file", arguments={"path": "a.txt", "content": "line1\nline2\n"}, runtime=runtime)
        )
        read_result = tools["read_file"].handler(
            ToolExecutionRequest(tool_name="read_file", arguments={"path": "a.txt", "start_line": 2, "end_line": 2}, runtime=runtime)
        )

        assert write_result.ok is True
        assert "Wrote" in write_result.content
        assert write_result.metadata["diff_artifact"]["operation"] == "create"
        assert "+++ b/a.txt" in write_result.metadata["diff_artifact"]["diff"]
        assert read_result.ok is True
        assert read_result.content == "line2\n"

    def test_edit_file_returns_diff_artifact(self, fs_provider: FileSystemToolProvider, runtime: dict[str, object]) -> None:
        tools = {tool.name: tool for tool in fs_provider.build_tools()}
        tools["write_file"].handler(
            ToolExecutionRequest(tool_name="write_file", arguments={"path": "a.txt", "content": "old\n"}, runtime=runtime)
        )

        result = tools["edit_file"].handler(
            ToolExecutionRequest(
                tool_name="edit_file",
                arguments={"path": "a.txt", "old_text": "old\n", "new_text": "new\n"},
                runtime=runtime,
            )
        )

        assert result.ok is True
        assert result.metadata["diff_artifact"]["operation"] == "update"
        assert "-old" in result.metadata["diff_artifact"]["diff"]
        assert "+new" in result.metadata["diff_artifact"]["diff"]

    def test_path_escape_is_blocked(self, fs_provider: FileSystemToolProvider, runtime: dict[str, object]) -> None:
        tools = {tool.name: tool for tool in fs_provider.build_tools()}

        with pytest.raises(ValueError, match="escapes workspace root"):
            tools["read_file"].handler(
                ToolExecutionRequest(tool_name="read_file", arguments={"path": "../outside.txt"}, runtime=runtime)
            )


class TestToolExecutor:
    """验证执行器分发与错误封装。"""

    def test_unknown_tool_returns_structured_error(self, runtime: dict[str, object]) -> None:
        executor = ToolExecutor()

        result = executor.execute(
            ToolRegistry.from_tools(),
            ToolExecutionRequest(tool_name="missing_tool", arguments={}, runtime=runtime),
        )

        assert result.ok is False
        assert result.metadata["error_kind"] == "unknown_tool"

    def test_execute_wraps_handler_exception(self, fs_provider: FileSystemToolProvider, runtime: dict[str, object]) -> None:
        executor = ToolExecutor()
        registry = ToolRegistry.from_tools(*fs_provider.build_tools())

        result = executor.execute(
            registry,
            ToolExecutionRequest(tool_name="read_file", arguments={"path": "missing.txt"}, runtime=runtime),
        )

        assert result.ok is False
        assert result.metadata["error_kind"] in {"tool_execution_error", "unexpected_error"}
