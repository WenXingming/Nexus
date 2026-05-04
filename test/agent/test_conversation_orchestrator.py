"""ConversationOrchestrator 行为测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.agent.conversation_orchestrator import ConversationOrchestrator
from src.core_contracts.context_contracts import BudgetConfig, ContextPolicy
from src.core_contracts.interaction_contracts import SlashCommandContext, SlashCommandResult
from src.core_contracts.model_config import ModelConfig
from src.core_contracts.session_contracts import SessionState
from src.core_contracts.workspace_config import WorkspaceConfig


def _make_orchestrator() -> ConversationOrchestrator:
    interaction_gateway = MagicMock()
    agent_gateway = MagicMock()
    session_gateway = MagicMock()
    tools_gateway = MagicMock()
    tools_gateway.list_tools.return_value = []
    return ConversationOrchestrator(
        interaction_gateway=interaction_gateway,
        agent_gateway=agent_gateway,
        session_gateway=session_gateway,
        tools_gateway=tools_gateway,
        budget_config=BudgetConfig(),
        context_policy=ContextPolicy(),
        model_config=ModelConfig(api_key='sk-test'),
        workspace_config=WorkspaceConfig.from_cwd(),
        session_id_factory=lambda: 'forked-session',
    )


def _make_state() -> SessionState:
    return SessionState(session_id='session-1')


class TestConversationOrchestratorSlashHandling:
    def test_help_renders_and_keeps_state(self) -> None:
        orchestrator = _make_orchestrator()
        state = _make_state()
        orchestrator.interaction_gateway.parse_slash_command.return_value = MagicMock(command_name='help')
        orchestrator.interaction_gateway.resolve_slash_command.return_value = MagicMock(kind='exact')
        orchestrator.interaction_gateway.dispatch_slash_command.return_value = SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name='help',
            output='Slash Commands',
        )

        result = orchestrator._handle_slash('/help', state)

        assert result is state
        orchestrator.interaction_gateway.render_slash_result.assert_called_once()

    def test_exit_returns_none(self) -> None:
        orchestrator = _make_orchestrator()
        state = _make_state()
        orchestrator.interaction_gateway.parse_slash_command.return_value = MagicMock(command_name='exit')
        orchestrator.interaction_gateway.resolve_slash_command.return_value = MagicMock(kind='exact')
        orchestrator.interaction_gateway.dispatch_slash_command.return_value = SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name='exit',
            output='Exiting local session interaction.',
            metadata={'exit_requested': True},
        )

        result = orchestrator._handle_slash('/exit', state)

        assert result is None

    def test_clear_forks_new_session(self) -> None:
        orchestrator = _make_orchestrator()
        state = _make_state()
        orchestrator.interaction_gateway.parse_slash_command.return_value = MagicMock(command_name='clear')
        orchestrator.interaction_gateway.resolve_slash_command.return_value = MagicMock(kind='exact')
        orchestrator.interaction_gateway.dispatch_slash_command.return_value = SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name='clear',
            output='Cleared in-memory session context.',
            fork_session=True,
        )

        result = orchestrator._handle_slash('/clear', state)

        assert isinstance(result, SessionState)
        assert result.session_id == 'forked-session'
        orchestrator.interaction_gateway.observe_run_result.assert_called_once()

    def test_new_session_via_replacement(self) -> None:
        orchestrator = _make_orchestrator()
        state = _make_state()
        new_state = SessionState(session_id='session-2')
        orchestrator.interaction_gateway.parse_slash_command.return_value = MagicMock(command_name='new')
        orchestrator.interaction_gateway.resolve_slash_command.return_value = MagicMock(kind='exact')
        orchestrator.interaction_gateway.dispatch_slash_command.return_value = SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name='new',
            output='[新会话] session_id=session-2',
            replacement_session_state=new_state,
        )

        result = orchestrator._handle_slash('/new', state)

        assert result is new_state

    def test_unknown_command_keeps_state(self) -> None:
        orchestrator = _make_orchestrator()
        state = _make_state()
        orchestrator.interaction_gateway.parse_slash_command.return_value = MagicMock(command_name='unknown')
        orchestrator.interaction_gateway.resolve_slash_command.return_value = MagicMock(kind='none')

        result = orchestrator._handle_slash('/unknown', state)

        assert result is state
        orchestrator.interaction_gateway.dispatch_slash_command.assert_not_called()

    def test_rag_index_dispatches_with_built_context(self) -> None:
        orchestrator = _make_orchestrator()
        state = _make_state()
        orchestrator.interaction_gateway.parse_slash_command.return_value = MagicMock(command_name='rag-index')
        orchestrator.interaction_gateway.resolve_slash_command.return_value = MagicMock(kind='exact')
        orchestrator.interaction_gateway.dispatch_slash_command.return_value = SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name='rag-index',
            output='[RAG 已索引] collection=main-loop, docs=1, chunks=2',
        )

        result = orchestrator._handle_slash('/rag-index docs/', state)

        assert result is state
        call_context = orchestrator.interaction_gateway.dispatch_slash_command.call_args.args[0]
        assert isinstance(call_context, SlashCommandContext)


class TestConversationOrchestratorPromptHandling:
    def test_prompt_appends_user_and_delegates_to_agent(self) -> None:
        orchestrator = _make_orchestrator()
        state = _make_state()
        orchestrator.agent_gateway.run.return_value = state

        result = orchestrator._handle_prompt('hello', state)

        assert result is state
        orchestrator.session_gateway.append_user.assert_called_once_with(state, 'hello')
        orchestrator.agent_gateway.run.assert_called_once_with(state)

    def test_run_loops_until_exit(self) -> None:
        orchestrator = _make_orchestrator()
        state = _make_state()
        orchestrator.interaction_gateway.read_input.side_effect = ['hello', '/quit']
        orchestrator.agent_gateway.run.return_value = state
        orchestrator.interaction_gateway.parse_slash_command.return_value = MagicMock(command_name='quit')
        orchestrator.interaction_gateway.resolve_slash_command.return_value = MagicMock(kind='exact')
        orchestrator.interaction_gateway.dispatch_slash_command.return_value = SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name='quit',
            output='bye',
            metadata={'exit_requested': True},
        )

        result = orchestrator.run(state)

        assert result is state
        orchestrator.agent_gateway.run.assert_called_once_with(state)