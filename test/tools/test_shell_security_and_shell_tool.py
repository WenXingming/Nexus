"""ShellToolProvider 单元测试。"""

from pathlib import Path
from unittest.mock import patch

from src.core_contracts.tools_contracts import ToolExecutionRequest
from src.tools.local.shell_tools import ShellToolProvider


def _runtime(tmp_path: Path) -> dict[str, object]:
    return {
        "root": str(tmp_path),
        "command_timeout_seconds": 3,
        "max_output_chars": 2000,
        "allow_shell_commands": True,
    }


class TestShellToolProvider:
    """验证 shell 工具调度。"""

    def test_run_delegates_to_subprocess(self, tmp_path: Path) -> None:
        provider = ShellToolProvider()
        tool = provider.build_tool()
        request = ToolExecutionRequest(tool_name="bash", arguments={"command": "echo hello"}, runtime=_runtime(tmp_path))

        with patch("src.tools.local.shell_tools.subprocess.run") as run_mock:
            run_mock.return_value.stdout = "hello\n"
            run_mock.return_value.stderr = ""
            run_mock.return_value.returncode = 0

            result = tool.handler(request)

        assert result.ok is True
        assert "exit_code=0" in result.content

    def test_requires_string_command(self, tmp_path: Path) -> None:
        provider = ShellToolProvider()
        tool = provider.build_tool()
        request = ToolExecutionRequest(tool_name="bash", arguments={"command": 123}, runtime=_runtime(tmp_path))

        try:
            tool.handler(request)
            assert False, "should raise"
        except ValueError as exc:
            assert "must be a string" in str(exc)

    def test_requires_shell_permission(self, tmp_path: Path) -> None:
        provider = ShellToolProvider()
        tool = provider.build_tool()
        runtime = {**_runtime(tmp_path), "allow_shell_commands": False}
        request = ToolExecutionRequest(tool_name="bash", arguments={"command": "echo hello"}, runtime=runtime)

        try:
            tool.handler(request)
            assert False, "should raise"
        except PermissionError as exc:
            assert "Shell command permission" in str(exc)

    def test_rejects_destructive_command_without_permission(self, tmp_path: Path) -> None:
        provider = ShellToolProvider()
        tool = provider.build_tool()
        requests = [
            ToolExecutionRequest(tool_name="bash", arguments={"command": "rm -rf build"}, runtime=_runtime(tmp_path)),
            ToolExecutionRequest(
                tool_name="bash",
                arguments={"command": "Remove-Item -LiteralPath build -Recurse"},
                runtime=_runtime(tmp_path),
            ),
        ]

        for request in requests:
            try:
                tool.handler(request)
                assert False, "should raise"
            except PermissionError as exc:
                assert "Destructive shell command permission" in str(exc)

    def test_nonzero_exit_code_returns_failed_result(self, tmp_path: Path) -> None:
        provider = ShellToolProvider()
        tool = provider.build_tool()
        request = ToolExecutionRequest(tool_name="bash", arguments={"command": "exit 7"}, runtime=_runtime(tmp_path))

        with patch("src.tools.local.shell_tools.subprocess.run") as run_mock:
            run_mock.return_value.stdout = ""
            run_mock.return_value.stderr = "bad\n"
            run_mock.return_value.returncode = 7

            result = tool.handler(request)

        assert result.ok is False
        assert result.metadata["exit_code"] == 7
