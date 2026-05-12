"""应用层 slash 命令装配测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.agent import build_default_slash_command_specs
from src.core_contracts.context_contracts import BudgetConfig, ContextPolicy
from src.core_contracts.interaction_contracts import PermissionPolicy, SlashCommandContext
from src.core_contracts.model_config import ModelConfig
from src.core_contracts.session_contracts import SessionState
from src.interaction.slash_commands import SlashCommandDispatcher


def _make_context(session_state: SessionState | None = None) -> SlashCommandContext:
    return SlashCommandContext(
        session_state=session_state or SessionState(session_id='sess-1'),
        session_id='sess-1',
        turns_offset=0,
        tool_call_count=0,
        workspace_path='D:/WorkSpace/Nexus',
        context_policy=ContextPolicy(),
        permissions=PermissionPolicy(),
        budget_config=BudgetConfig(),
        model_config=ModelConfig(api_key='test-key'),
        tool_registry=(),
    )


def _dispatch(
    input_text: str,
    *,
    context: SlashCommandContext | None = None,
    context_gateway: object | None = None,
    session_gateway: object | None = None,
    rag_gateway: object | None = None,
    client_gateway: object | None = None,
):
    specs = build_default_slash_command_specs(
        context_gateway=context_gateway,
        session_gateway=session_gateway,
        rag_gateway=rag_gateway,
        client_gateway=client_gateway,
    )
    dispatcher = SlashCommandDispatcher(specs=specs)
    return dispatcher.dispatch_slash_command(context or _make_context(), input_text)


class TestSessionCommands:
    def test_new_creates_replacement_state(self) -> None:
        session_gateway = MagicMock()
        session_gateway.create_state.return_value = SessionState(session_id='sess-2')
        session_gateway.create_empty_state.return_value = SessionState(session_id='sess-2')

        result = _dispatch('/new', session_gateway=session_gateway)

        session_gateway.create_empty_state.assert_called_once_with()
        session_gateway.create_state.assert_not_called()
        assert result.replacement_session_state is not None
        assert result.replacement_session_state.session_id == 'sess-2'

    def test_save_uses_session_gateway(self) -> None:
        session_gateway = MagicMock()
        session_gateway.save_state.return_value = ('sess-1', 'sessions/sess-1.json')
        context = _make_context(SessionState(session_id='sess-1'))

        result = _dispatch('/save', context=context, session_gateway=session_gateway)

        session_gateway.save_state.assert_called_once()
        assert 'sessions/sess-1.json' in result.output

    def test_load_replaces_state(self) -> None:
        session_gateway = MagicMock()
        snapshot = MagicMock(messages=('m1', 'm2'), transcript=('t1',))
        new_state = SessionState(session_id='sess-2')
        session_gateway.load.return_value = snapshot
        session_gateway.resume_state.return_value = new_state

        result = _dispatch('/load sess-2', session_gateway=session_gateway)

        session_gateway.load.assert_called_once_with('sess-2')
        assert result.replacement_session_state is new_state


class TestRagCommands:
    def test_rag_index_passes_source_path_to_rag_gateway(self) -> None:
        rag_gateway = MagicMock()
        rag_gateway.index.return_value = MagicMock(
            collection_name='main-loop',
            docs_indexed=1,
            chunks_created=2,
        )

        result = _dispatch('/rag-index docs', rag_gateway=rag_gateway)

        rag_gateway.index.assert_called_once()
        request = rag_gateway.index.call_args.args[0]
        assert request.source_path == 'docs'
        assert request.documents == ()
        assert 'collection=main-loop' in result.output

    def test_rag_ask_updates_session_and_returns_answer(self) -> None:
        session_gateway = MagicMock()
        client_gateway = MagicMock()
        rag_gateway = MagicMock()
        rag_gateway.retrieve_and_build_messages.return_value = [
            MagicMock(role='system', content='sys'),
            MagicMock(role='user', content='query'),
        ]
        client_gateway.chat.return_value = MagicMock(content='检索增强生成。')
        context = _make_context(SessionState(session_id='sess-1'))

        result = _dispatch(
            '/rag-ask 什么是 RAG',
            context=context,
            session_gateway=session_gateway,
            rag_gateway=rag_gateway,
            client_gateway=client_gateway,
        )

        session_gateway.append_user.assert_called_once()
        session_gateway.append_assistant.assert_called_once()
        assert '检索增强生成。' in result.output
