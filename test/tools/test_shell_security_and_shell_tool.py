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
