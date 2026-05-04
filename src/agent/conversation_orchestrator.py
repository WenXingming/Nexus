"""应用层会话编排器。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable

from src.agent.agent_gateway import AgentGateway
from src.core_contracts.context_contracts import BudgetConfig, ContextPolicy
from src.core_contracts.interaction_contracts import (
    AgentRunResult,
    PermissionPolicy,
    SlashCommandContext,
)
from src.core_contracts.model_config import ModelConfig
from src.core_contracts.session_contracts import SessionState
from src.core_contracts.workspace_config import WorkspaceConfig
from src.interaction import InteractionGateway
from src.session import SessionGateway
from src.tools import ToolsGateway


def _new_session_id() -> str:
    """生成新的会话 ID。"""
    return uuid.uuid4().hex[:32]


@dataclass
class ConversationOrchestrator:
    """统一处理 REPL 输入分流、slash 状态翻译与普通 prompt 编排。"""

    interaction_gateway: InteractionGateway
    agent_gateway: AgentGateway
    session_gateway: SessionGateway
    tools_gateway: ToolsGateway
    budget_config: BudgetConfig
    context_policy: ContextPolicy
    model_config: ModelConfig
    workspace_config: WorkspaceConfig
    session_id_factory: Callable[[], str] = _new_session_id

    def run(self, state: SessionState) -> SessionState:
        """运行交互式会话循环，直到收到退出信号。"""
        current_state = state
        while True:
            user_input = self._read_user_input()
            if not user_input:
                continue

            next_state = self._handle_input(user_input, current_state)
            if next_state is None:
                return current_state
            current_state = next_state

    def _read_user_input(self) -> str:
        """读取并处理用户输入。"""
        try:
            return self.interaction_gateway.read_input("\nUser > ").strip()
        except (KeyboardInterrupt, EOFError):
            return "/quit"

    def _handle_input(self, user_input: str, state: SessionState) -> SessionState | None:
        """分流单条用户输入。"""
        if user_input.startswith("/"):
            return self._handle_slash(user_input, state)
        return self._handle_prompt(user_input, state)

    def _handle_prompt(self, user_input: str, state: SessionState) -> SessionState | None:
        """处理普通 prompt 输入。"""
        self.session_gateway.append_user(state, user_input)
        return self.agent_gateway.run(state)

    def _handle_slash(self, user_input: str, state: SessionState) -> SessionState | None:
        """处理 slash 命令并翻译其结果为应用状态。"""
        parsed = self.interaction_gateway.parse_slash_command(user_input)
        if parsed is None:
            print(f"[未知命令] {user_input}")
            return state

        resolution = self.interaction_gateway.resolve_slash_command(parsed.command_name)
        if resolution.kind == "none":
            print(f"[未知命令] {user_input}")
            return state

        cmd_result = self.interaction_gateway.dispatch_slash_command(
            self._build_slash_context(state),
            user_input,
        )
        return self._apply_slash_result(state, cmd_result)

    def _build_slash_context(self, state: SessionState) -> SlashCommandContext:
        """为本轮 slash 命令构建运行时上下文。"""
        tool_registry = tuple(self.tools_gateway.list_tools())
        tool_names = {tool.name for tool in tool_registry}
        permission_policy = PermissionPolicy(
            allow_file_write=bool({"write_file", "edit_file"} & tool_names),
            allow_shell_commands=bool({"bash", "shell"} & tool_names),
            allow_destructive_shell_commands=False,
        )
        tool_call_count = sum(1 for message in state.messages if message.role == "tool")
        return SlashCommandContext(
            session_state=state,
            session_id=state.session_id,
            turns_offset=max(len(state.transcript_entries) - 1, 0),
            tool_call_count=tool_call_count,
            workspace_path=str(self.workspace_config.root),
            context_policy=self.context_policy,
            permissions=permission_policy,
            budget_config=self.budget_config,
            model_config=self.model_config,
            tool_registry=tool_registry,
        )

    def _apply_slash_result(self, state: SessionState, cmd_result) -> SessionState | None:
        """把 SlashCommandResult 翻译为会话状态或退出信号。"""
        if cmd_result.handled:
            self.interaction_gateway.render_slash_result(
                command_name=cmd_result.command_name,
                output=cmd_result.output,
                metadata=cmd_result.metadata,
            )

        if cmd_result.metadata.get("exit_requested"):
            return None

        new_state = state
        if cmd_result.fork_session:
            new_state = SessionState(session_id=self.session_id_factory())
        elif cmd_result.replacement_session_state is not None:
            new_state = cmd_result.replacement_session_state

        if new_state.session_id != state.session_id:
            self.interaction_gateway.observe_run_result(
                AgentRunResult(session_id=new_state.session_id),
                current_session_id=state.session_id,
            )

        return new_state
