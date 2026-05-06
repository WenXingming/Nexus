"""Agent 循环执行器。

负责执行 Agent 的核心迭代循环：pre-model → LLM → tool execution。
"""

from __future__ import annotations

import json
import sys
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
from src.core_contracts.interaction_contracts import AgentRunResult
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
                print(f"\n[Context] pre-model stop: {pre_model.pre_model_stop}")
                break

            reactive_attempt = 0
            result = None
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
                        print(f"\n[Context] stop after reactive compact: {outcome.stop_reason}", file=sys.stderr)
                        result = None
                        break
                    if not outcome.retry_model_call:
                        print("\n[Error] 模型调用失败", file=sys.stderr)
                        result = None
                        break

            if result is None:
                break

            if result.finish_reason == "length":
                print(
                    f"\n[警告] LLM 响应因 max_tokens 限制被截断 (finish_reason=length)。"
                    f" 当前 max_tokens 可能不足以容纳工具调用参数。",
                    file=sys.stderr,
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

                print(f"\n[工具调用] {tool_name}({arguments})")

                exec_result = self.tools_gateway.execute_tool(
                    ToolExecutionRequest(tool_name=tool_name, arguments=arguments)
                )

                self.interaction_gateway.observe_run_result(
                    AgentRunResult(
                        session_id=state.session_id,
                        events=({"type": "tool_result", "tool_name": tool_name, "ok": exec_result.ok},),
                    ),
                    current_session_id=state.session_id,
                )

                status = "OK" if exec_result.ok else "FAIL"
                print(f"[工具结果] {tool_name} -> {status}: {exec_result.content[:200]}")

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


