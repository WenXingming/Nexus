"""AgentGateway 行为测试。"""

from __future__ import annotations

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
        tools=[],
    )
    return AgentGateway(
        client=client,
        context_gateway=context_gateway,
        session_gateway=session_gateway,
        tools_gateway=tools_gateway,
        interaction_gateway=interaction_gateway,
        budget_config=BudgetConfig(),
        context_policy=ContextPolicy(),
        budget_guard=PreModelBudgetGuard(),
        tools=[],
        executor=executor,
    )


def _make_state() -> SessionState:
    return SessionState(
        session_id="session-1",
        messages=[Message(role="user", content="hi")],
    )


class TestAgentGatewayRun:
    def test_run_appends_assistant_when_model_returns_text(self) -> None:
        gateway = _make_gateway()
        state = _make_state()
        gateway.context_gateway.run_pre_model_cycle.return_value = PreModelContextOutcome(
            pre_model_stop=None,
            events=(),
        )
        gateway.client.chat.return_value = LlmResult(
            content="hello",
            model="gpt-4o",
            finish_reason="stop",
            usage=TokenUsage(1, 2, 3),
            tool_calls=None,
        )

        result = gateway.run(state)

        assert result is state
        gateway.session_gateway.append_assistant.assert_called_once_with(state, "hello")

    def test_run_executes_tool_calls_and_continues_until_final_answer(self) -> None:
        gateway = _make_gateway()
        state = _make_state()
        gateway.context_gateway.run_pre_model_cycle.return_value = PreModelContextOutcome(
            pre_model_stop=None,
            events=(),
        )
        gateway.client.chat.side_effect = [
            LlmResult(
                content="",
                model="gpt-4o",
                finish_reason="tool_calls",
                usage=TokenUsage(1, 1, 2),
                tool_calls=[
                    {
                        "id": "tool-1",
                        "type": "function",
                        "function": {"name": "list_dir", "arguments": '{"path": "."}'},
                    }
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
        gateway.tools_gateway.execute_tool.return_value = MagicMock(ok=True, content="file.txt")

        result = gateway.run(state)

        assert result is state
        gateway.session_gateway.append_message.assert_called()
        gateway.tools_gateway.execute_tool.assert_called_once()
        gateway.interaction_gateway.observe_run_result.assert_called_once()
        gateway.session_gateway.append_assistant.assert_called_once_with(state, "done")

    def test_run_retries_after_reactive_compact(self) -> None:
        gateway = _make_gateway()
        state = _make_state()
        gateway.context_gateway.run_pre_model_cycle.return_value = PreModelContextOutcome(
            pre_model_stop=None,
            events=(),
        )
        gateway.client.chat.side_effect = [
            RuntimeError("context length exceeded"),
            LlmResult(
                content="after retry",
                model="gpt-4o",
                finish_reason="stop",
                usage=TokenUsage(1, 1, 2),
                tool_calls=None,
            ),
        ]
        gateway.context_gateway.run_reactive_compact_cycle.return_value = ReactiveCompactOutcome(
            retry_model_call=True,
            stop_reason=None,
            events=(),
        )

        gateway.run(state)

        assert gateway.client.chat.call_count == 2
        gateway.context_gateway.run_reactive_compact_cycle.assert_called_once()
        gateway.session_gateway.append_assistant.assert_called_once_with(state, "after retry")

    def test_run_stops_before_model_call_when_pre_model_requests_stop(self) -> None:
        gateway = _make_gateway()
        state = _make_state()
        gateway.context_gateway.run_pre_model_cycle.return_value = PreModelContextOutcome(
            pre_model_stop="hard_input_budget_exceeded",
            events=(),
        )

        result = gateway.run(state)

        assert result is state
        gateway.client.chat.assert_not_called()