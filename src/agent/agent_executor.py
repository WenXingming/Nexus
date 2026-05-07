"""Agent 循环执行器。

负责执行 Agent 的核心迭代循环：pre-model → LLM → tool execution。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.client import ClientGateway
from src.context import ContextGateway
from src.core_contracts.client_contracts import LlmRequest
from src.core_contracts.context_contracts import (
    BudgetConfig,
    ContextPolicy,
    ContextRunState,
    PreModelBudgetGuard,
)
from src.core_contracts.interaction_contracts import AgentRunResult, TaskProgressEvent, ToolCallRecord
from src.core_contracts.model_contracts import Message, TokenUsage
from src.core_contracts.session_contracts import SessionState
from src.core_contracts.tools_contracts import ToolExecutionRequest
from src.interaction import InteractionGateway
from src.session import SessionGateway
from src.tools import ToolsGateway


@dataclass
class AgentLoopExecutor:
    """Agent 循环执行器。

    执行 pre-model 检查、模型调用、工具执行的完整迭代循环。
    """

    client: ClientGateway
    """ClientGateway: 模型调用客户端。"""

    context_gateway: ContextGateway
    """ContextGateway: 上下文治理网关。"""

    session_gateway: SessionGateway
    """SessionGateway: 会话状态管理网关。"""

    tools_gateway: ToolsGateway
    """ToolsGateway: 工具执行网关。"""

    interaction_gateway: InteractionGateway
    """InteractionGateway: 交互观察网关。"""

    budget_config: BudgetConfig
    """BudgetConfig: Token 预算配置。"""

    context_policy: ContextPolicy
    """ContextPolicy: 上下文治理策略。"""

    budget_guard: PreModelBudgetGuard
    """PreModelBudgetGuard: 预算守卫。"""

    def execute(self, state: SessionState, max_turns: int = 5) -> SessionState | None:
        """执行 Agent 迭代循环。

        Args:
            state (SessionState): 当前会话状态
            max_turns (int): 最大迭代轮次，默认 5
        Returns:
            SessionState | None: 更新后的会话状态，失败时返回 None
        Raises:
            None
        """
        for turn_index in range(max_turns):
            # 每轮 turn 开始时动态获取最新工具列表
            tools = self.tools_gateway.list_openai_tools()
            turn_number = turn_index + 1

            run_state = ContextRunState(
                session_messages=state.messages,
                turn_index=turn_index,
                turns_offset=max(0, len(state.messages) - 1),
                turns_this_run=turn_index,
            )

            pre_model = self.context_gateway.run_pre_model_cycle(
                run_state=run_state,
                budget_config=self.budget_config,
                context_policy=self.context_policy,
                guard=self.budget_guard,
                tools=tools,
            )
            self.interaction_gateway.print_context_events(pre_model.events)
            if pre_model.pre_model_stop is not None:
                self.interaction_gateway.render_task_event(
                    TaskProgressEvent(
                        title="Model request stopped",
                        detail=f"reason={pre_model.pre_model_stop}",
                        status='warning',
                        turn=turn_number,
                    )
                )
                break

            reactive_attempt = 0
            result = None
            self.interaction_gateway.render_task_event(
                TaskProgressEvent(
                    title="Thinking",
                    status='running',
                    turn=turn_number,
                )
            )
            while True:
                try:
                    request = LlmRequest(messages=list(state.messages), tools=tools or None)
                    result = self.client.chat(request)
                    run_state.model_call_count += 1
                    run_state.usage_delta = TokenUsage(
                        prompt_tokens=run_state.usage_delta.prompt_tokens + result.usage.prompt_tokens,
                        completion_tokens=run_state.usage_delta.completion_tokens + result.usage.completion_tokens,
                        total_tokens=run_state.usage_delta.total_tokens + result.usage.total_tokens,
                    )
                    break
                except Exception as exc:
                    reactive_attempt += 1
                    self.interaction_gateway.render_task_event(
                        TaskProgressEvent(
                            title="Model call failed",
                            detail=str(exc),
                            status='warning',
                            turn=turn_number,
                        )
                    )
                    outcome = self.context_gateway.run_reactive_compact_cycle(
                        run_state=run_state,
                        budget_config=self.budget_config,
                        context_policy=self.context_policy,
                        tools=tools,
                        guard=self.budget_guard,
                        error=exc,
                        attempt=reactive_attempt,
                    )
                    self.interaction_gateway.print_context_events(outcome.events)
                    if outcome.stop_reason is not None:
                        self.interaction_gateway.render_task_event(
                            TaskProgressEvent(
                                title="Reactive compact stopped retry",
                                detail=f"reason={outcome.stop_reason}",
                                status='error',
                                turn=turn_number,
                            )
                        )
                        result = None
                        break
                    if not outcome.retry_model_call:
                        self.interaction_gateway.render_task_event(
                            TaskProgressEvent(
                                title="Model request aborted",
                                detail="No more recovery steps are available.",
                                status='error',
                                turn=turn_number,
                            )
                        )
                        result = None
                        break

            if result is None:
                break

            if result.finish_reason == "length":
                self.interaction_gateway.render_task_event(
                    TaskProgressEvent(
                        title="Model output truncated",
                        detail=(
                            "finish_reason=length; current max_tokens may be too low "
                            "for the requested response."
                        ),
                        status='warning',
                        turn=turn_number,
                    )
                )

            if not result.tool_calls:
                self.session_gateway.append_assistant(state, result.content)
                print(f"\nAgent> {result.content}")
                break

            self.session_gateway.append_message(
                state,
                Message(
                    role="assistant",
                    content=result.content or None,
                    tool_calls=result.tool_calls,
                ),
            )

            for tool_call in result.tool_calls:
                tool_name = tool_call["function"]["name"]
                try:
                    arguments = json.loads(tool_call["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    arguments = {}
                self.interaction_gateway.render_task_event(
                    TaskProgressEvent(
                        title=self._build_tool_start_title(tool_name, arguments),
                        detail=self._build_tool_start_detail(tool_name, arguments),
                        status='running',
                        turn=turn_number,
                    )
                )

                exec_result = self.tools_gateway.execute_tool(
                    ToolExecutionRequest(tool_name=tool_name, arguments=arguments)
                )

                self.interaction_gateway.observe_run_result(
                    AgentRunResult(
                        session_id=state.session_id,
                        tool_calls=(ToolCallRecord(name=tool_name, ok=exec_result.ok),),
                    ),
                    current_session_id=state.session_id,
                )

                diff_artifact = self.interaction_gateway.build_diff_artifact(exec_result.metadata)
                self.interaction_gateway.render_task_event(
                    TaskProgressEvent(
                        title=self._build_tool_finish_title(tool_name, exec_result, diff_artifact),
                        detail=self._build_tool_finish_detail(exec_result, diff_artifact),
                        status='success' if exec_result.ok else 'error',
                        turn=turn_number,
                        diff=diff_artifact,
                    )
                )

                self.session_gateway.append_message(
                    state,
                    Message(
                        role="tool",
                        content=exec_result.content,
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                    ),
                )

        return state

    @staticmethod
    def _build_tool_start_title(tool_name: str, arguments: dict[str, object]) -> str:
        path = arguments.get("path")
        if tool_name == "read_file" and isinstance(path, str):
            return f"Reading {path}"
        if tool_name == "write_file" and isinstance(path, str):
            return f"Writing {path}"
        if tool_name == "edit_file" and isinstance(path, str):
            return f"Editing {path}"
        if tool_name == "list_dir" and isinstance(path, str):
            return f"Listing {path}"
        if tool_name == "bash":
            return "Running command"
        return f"Calling {tool_name}"

    @classmethod
    def _build_tool_start_detail(cls, tool_name: str, arguments: dict[str, object]) -> str:
        if tool_name == "bash":
            return cls._shorten_text(arguments.get("command"))
        if tool_name == "read_file":
            start_line = arguments.get("start_line")
            end_line = arguments.get("end_line")
            if isinstance(start_line, int) and isinstance(end_line, int):
                return f"lines {start_line}-{end_line}"
            if isinstance(start_line, int):
                return f"from line {start_line}"
        return ""

    @classmethod
    def _build_tool_finish_title(cls, tool_name: str, exec_result, diff_artifact) -> str:
        if not exec_result.ok:
            return f"{tool_name} failed"
        if diff_artifact is not None:
            verb = "Created" if diff_artifact.operation == "create" else "Updated"
            return f"{verb} {diff_artifact.path}"
        metadata = exec_result.metadata
        action = metadata.get("action")
        path = metadata.get("path")
        if action == "read_file" and isinstance(path, str):
            return f"Read {path}"
        if action == "list_dir" and isinstance(path, str):
            return f"Listed {path}"
        if action == "bash":
            return "Command finished"
        return f"Completed {tool_name}"

    @classmethod
    def _build_tool_finish_detail(cls, exec_result, diff_artifact) -> str:
        if not exec_result.ok:
            return cls._shorten_text(exec_result.content, max_length=180)
        if diff_artifact is not None:
            return ""
        metadata = exec_result.metadata
        action = metadata.get("action")
        if action == "list_dir":
            returned_entries = metadata.get("returned_entries")
            if isinstance(returned_entries, int):
                return f"{returned_entries} entries"
        if action == "bash":
            exit_code = metadata.get("exit_code")
            if isinstance(exit_code, int):
                return f"exit_code={exit_code}"
        return cls._shorten_text(exec_result.content, max_length=100)

    @staticmethod
    def _shorten_text(value: object, *, max_length: int = 80) -> str:
        if not isinstance(value, str):
            return ""
        first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
        if len(first_line) <= max_length:
            return first_line
        return f"{first_line[: max_length - 3].rstrip()}..."


