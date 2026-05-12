"""AgentGateway 行为测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.agent.agent_executor import AgentLoopExecutor
from src.agent.agent_gateway import AgentGateway
from src.core_contracts.client_contracts import LlmResult
from src.core_contracts.context_contracts import (
    BudgetConfig,
    ContextPolicy,
    PreModelContextOutcome,
    PreModelBudgetGuard,
    ReactiveCompactOutcome,
)
from src.core_contracts.model_contracts import Message, TokenUsage
from src.core_contracts.session_contracts import SessionState


def _make_gateway() -> AgentGateway:
    client = MagicMock()
    context_gateway = MagicMock()
    session_gateway = MagicMock()
    tools_gateway = MagicMock()
    interaction_gateway = MagicMock()
    executor = AgentLoopExecutor(
        client=client,
        context_gateway=context_gateway,
        session_gateway=session_gateway,
        tools_gateway=tools_gateway,
        interaction_gateway=interaction_gateway,
        budget_config=BudgetConfig(),
        context_policy=ContextPolicy(),
        budget_guard=PreModelBudgetGuard(),
    )
    return AgentGateway(executor=executor)


def _make_state() -> SessionState:
    return SessionState(
        session_id="session-1",
        messages=[Message(role="user", content="hi")],
    )


def _make_tool_call(name: str = "list_dir", arguments: str = '{"path": "."}') -> dict:
    return {
        "id": "tool-1",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class TestAgentGatewayRun:
    def test_run_appends_assistant_when_model_returns_text(self) -> None:
        gateway = _make_gateway()
        executor = gateway.executor
        state = _make_state()
        executor.context_gateway.run_pre_model_cycle.return_value = PreModelContextOutcome(
            pre_model_stop=None,
            events=(),
        )
        executor.client.chat.return_value = LlmResult(
            content="hello",
            model="gpt-4o",
            finish_reason="stop",
            usage=TokenUsage(1, 2, 3),
            tool_calls=None,
        )

        result = gateway.run(state)

        assert result is state
        executor.session_gateway.append_assistant.assert_called_once_with(state, "hello")

    def test_run_executes_tool_calls_and_continues_until_final_answer(self) -> None:
        gateway = _make_gateway()
        executor = gateway.executor
        state = _make_state()
        executor.context_gateway.run_pre_model_cycle.return_value = PreModelContextOutcome(
            pre_model_stop=None,
            events=(),
        )
        executor.client.chat.side_effect = [
            LlmResult(
                content="",
                model="gpt-4o",
                finish_reason="tool_calls",
                usage=TokenUsage(1, 1, 2),
                tool_calls=[
                    _make_tool_call()
                ],
            ),
            LlmResult(
                content="done",
                model="gpt-4o",
                finish_reason="stop",
                usage=TokenUsage(1, 1, 2),
                tool_calls=None,
            ),
        ]
        executor.tools_gateway.execute_tool.return_value = MagicMock(ok=True, content="file.txt")

        result = gateway.run(state)

        assert result is state
        executor.session_gateway.append_message.assert_called()
        executor.tools_gateway.execute_tool.assert_called_once()
        executor.interaction_gateway.observe_run_result.assert_called_once()
        executor.session_gateway.append_assistant.assert_called_once_with(state, "done")

    def test_run_retries_after_reactive_compact(self) -> None:
        gateway = _make_gateway()
        executor = gateway.executor
        state = _make_state()
        executor.context_gateway.run_pre_model_cycle.return_value = PreModelContextOutcome(
            pre_model_stop=None,
            events=(),
        )
        executor.client.chat.side_effect = [
            RuntimeError("context length exceeded"),
            LlmResult(
                content="after retry",
                model="gpt-4o",
                finish_reason="stop",
                usage=TokenUsage(1, 1, 2),
                tool_calls=None,
            ),
        ]
        executor.context_gateway.run_reactive_compact_cycle.return_value = ReactiveCompactOutcome(
            retry_model_call=True,
            stop_reason=None,
            events=(),
        )

        gateway.run(state)

        assert executor.client.chat.call_count == 2
        executor.context_gateway.run_reactive_compact_cycle.assert_called_once()
        executor.session_gateway.append_assistant.assert_called_once_with(state, "after retry")

    def test_run_stops_before_model_call_when_pre_model_requests_stop(self) -> None:
        gateway = _make_gateway()
        executor = gateway.executor
        state = _make_state()
        executor.context_gateway.run_pre_model_cycle.return_value = PreModelContextOutcome(
            pre_model_stop="hard_input_budget_exceeded",
            events=(),
        )

        result = gateway.run(state)

        assert result is state
        executor.client.chat.assert_not_called()

    def test_run_uses_messages_rewritten_by_context_cycle(self) -> None:
        gateway = _make_gateway()
        executor = gateway.executor
        state = SessionState(
            session_id="session-1",
            messages=[
                Message(role="user", content="old"),
                Message(role="assistant", content="long old content"),
            ],
        )
        rewritten = [Message(role="system", content="compact summary"), Message(role="user", content="old")]

        def rewrite_context(run_state, **kwargs):
            del kwargs
            run_state.session_messages = rewritten
            return PreModelContextOutcome(pre_model_stop=None, events=())

        executor.context_gateway.run_pre_model_cycle.side_effect = rewrite_context
        executor.client.chat.return_value = LlmResult(
            content="hello",
            model="gpt-4o",
            finish_reason="stop",
            usage=TokenUsage(1, 2, 3),
            tool_calls=None,
        )

        gateway.run(state)

        assert state.messages is rewritten
        request = executor.client.chat.call_args.args[0]
        assert request.messages == rewritten

    def test_run_passes_explicit_runtime_to_tool_execution(self, tmp_path: Path) -> None:
        gateway = _make_gateway()
        executor = gateway.executor
        executor.workspace_root = tmp_path
        state = _make_state()
        executor.context_gateway.run_pre_model_cycle.return_value = PreModelContextOutcome(
            pre_model_stop=None,
            events=(),
        )
        executor.client.chat.side_effect = [
            LlmResult(
                content="",
                model="gpt-4o",
                finish_reason="tool_calls",
                usage=TokenUsage(1, 1, 2),
                tool_calls=[_make_tool_call()],
            ),
            LlmResult(
                content="done",
                model="gpt-4o",
                finish_reason="stop",
                usage=TokenUsage(1, 1, 2),
                tool_calls=None,
            ),
        ]
        executor.tools_gateway.execute_tool.return_value = MagicMock(ok=True, content="file.txt")

        gateway.run(state)

        request = executor.tools_gateway.execute_tool.call_args.args[0]
        assert request.runtime["root"] == str(tmp_path.resolve())
        assert request.runtime["allow_file_write"] is True
        assert request.runtime["allow_shell_commands"] is True
        assert request.runtime["allow_destructive_shell_commands"] is False
