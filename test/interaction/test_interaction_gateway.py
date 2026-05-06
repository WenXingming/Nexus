"""interaction 模块集成测试套件。

覆盖范围：
- InteractionGateway 全部公开方法（通过 mock 注入的七个内部组件）
- SlashCommandDispatcher：parse_slash_command、dispatch_slash_command、
  resolve_slash_command、find_slash_command 与八个内置 handler
- SessionInteractionTracker：start、observe_run_result、observe_tool_result、to_summary
- EnvironmentLoadSummary.render_line()
- SlashAutocompleteCatalog.get_matches()
"""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.agent import build_default_slash_command_specs
from src.core_contracts.context_contracts import BudgetConfig, BudgetProjection, ContextPolicy
from src.core_contracts.interaction_contracts import (
    AgentRunResult,
    EnvironmentLoadSummary,
    JSONDict,
    PermissionPolicy,
    SessionSummary,
    SlashAutocompleteEntry,
    SlashCommandContext,
    SlashCommandSpec,
)
from src.core_contracts.model_config import ModelConfig
from src.core_contracts.session_contracts import SessionState
from src.core_contracts.tools_contracts import ToolDescriptor
from src.interaction import InteractionGateway, create_interaction_gateway
from src.interaction.quit_render import ExitRenderer
from src.interaction.runtime_event_printer import RuntimeEventPrinter
from src.interaction.session_summary import SessionInteractionTracker
from src.interaction.slash_autocomplete import SlashAutocompleteCatalog, SlashAutocompletePrompt
from src.interaction.slash_commands import SlashCommandDispatcher
from src.interaction.slash_render import SlashCommandRenderer
from src.interaction.startup_render import StartupRenderer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gateway(
    dispatcher: SlashCommandDispatcher | None = None,
    event_printer: RuntimeEventPrinter | None = None,
    session_tracker: SessionInteractionTracker | None = None,
    stream: io.StringIO | None = None,
) -> InteractionGateway:
    """构建以最小 mock 装配的 InteractionGateway。"""
    _stream = stream or io.StringIO()
    gw = InteractionGateway(
        dispatcher=dispatcher or SlashCommandDispatcher(),
        startup_renderer=StartupRenderer(),
        exit_renderer=ExitRenderer(),
        slash_renderer=SlashCommandRenderer(),
        event_printer=event_printer or RuntimeEventPrinter(stream=_stream),
        autocomplete_prompt=SlashAutocompletePrompt(entries=(), fallback_reader=lambda _: ''),
        stream=_stream,
    )
    if session_tracker is not None:
        gw._session_tracker = session_tracker
    return gw


def _make_slash_context(
    tool_registry: tuple[ToolDescriptor, ...] = (),
    model_name: str = 'gpt-4o',
    workspace_path: str = '/tmp',
    turns_offset: int = 0,
    tool_call_count: int = 0,
    session_id: str = 'sess-1',
    session_state: SessionState | None = None,
    permissions: PermissionPolicy | None = None,
) -> SlashCommandContext:
    """构建 SlashCommandContext 测试辅助对象。"""
    return SlashCommandContext(
        session_state=session_state or SessionState(),
        session_id=session_id,
        turns_offset=turns_offset,
        tool_call_count=tool_call_count,
        workspace_path=workspace_path,
        context_policy=ContextPolicy(),
        permissions=permissions or PermissionPolicy(),
        budget_config=BudgetConfig(),
        model_config=ModelConfig(api_key='test-key', model_name=model_name),
        tool_registry=tool_registry,
    )


def _make_dispatcher_with_specs() -> SlashCommandDispatcher:
    """构建已装配全部内置规格的分发器。"""
    specs = build_default_slash_command_specs()
    return SlashCommandDispatcher(specs=specs)


# ---------------------------------------------------------------------------
# EnvironmentLoadSummary
# ---------------------------------------------------------------------------


class TestEnvironmentLoadSummary:
    def test_render_line_empty_returns_empty_string(self) -> None:
        summary = EnvironmentLoadSummary()
        assert summary.render_line() == ''

    def test_render_line_single_mcp_server(self) -> None:
        summary = EnvironmentLoadSummary(mcp_servers=1)
        line = summary.render_line()
        assert '1 MCP server' in line

    def test_render_line_multiple_mcp_servers(self) -> None:
        summary = EnvironmentLoadSummary(mcp_servers=3)
        line = summary.render_line()
        assert '3 MCP servers' in line

    def test_render_line_multiple_fields(self) -> None:
        summary = EnvironmentLoadSummary(mcp_servers=2, plugins=1, load_errors=1)
        line = summary.render_line()
        assert '2 MCP servers' in line
        assert '1 plugin' in line
        assert '1 load error' in line

    def test_render_line_starts_with_prefix(self) -> None:
        summary = EnvironmentLoadSummary(plugins=2)
        assert summary.render_line().startswith('Environment loaded:')

    def test_render_line_zero_load_errors_not_included(self) -> None:
        summary = EnvironmentLoadSummary(mcp_servers=1, load_errors=0)
        line = summary.render_line()
        assert 'error' not in line


# ---------------------------------------------------------------------------
# SlashAutocompleteCatalog
# ---------------------------------------------------------------------------


class TestSlashAutocompleteCatalog:
    def _catalog(self) -> SlashAutocompleteCatalog:
        return SlashAutocompleteCatalog(entries=(
            SlashAutocompleteEntry(name='help', description='Show help'),
            SlashAutocompleteEntry(name='status', description='Show status'),
            SlashAutocompleteEntry(name='context', description='Show context'),
        ))

    def test_get_matches_empty_input_returns_empty(self) -> None:
        assert self._catalog().get_matches('') == ()

    def test_get_matches_no_slash_returns_empty(self) -> None:
        assert self._catalog().get_matches('help') == ()

    def test_get_matches_slash_only_returns_all(self) -> None:
        results = self._catalog().get_matches('/')
        assert len(results) == 3

    def test_get_matches_prefix_filters(self) -> None:
        results = self._catalog().get_matches('/s')
        names = {e.name for e in results}
        assert names == {'status'}

    def test_get_matches_exact(self) -> None:
        results = self._catalog().get_matches('/help')
        assert len(results) == 1
        assert results[0].name == 'help'

    def test_get_matches_with_space_returns_empty(self) -> None:
        assert self._catalog().get_matches('/help arg') == ()

    def test_get_matches_leading_space_allowed(self) -> None:
        results = self._catalog().get_matches('  /h')
        assert any(e.name == 'help' for e in results)


# ---------------------------------------------------------------------------
# SlashCommandDispatcher — parse
# ---------------------------------------------------------------------------


class TestParseSlashCommand:
    def setup_method(self) -> None:
        self.d = _make_dispatcher_with_specs()

    def test_plain_text_returns_none(self) -> None:
        assert self.d.parse_slash_command('hello world') is None

    def test_slash_only(self) -> None:
        result = self.d.parse_slash_command('/')
        assert result is not None
        assert result.command_name == ''

    def test_slash_help(self) -> None:
        result = self.d.parse_slash_command('/help')
        assert result is not None
        assert result.command_name == 'help'
        assert result.arguments == ''

    def test_slash_with_arguments(self) -> None:
        result = self.d.parse_slash_command('/status  extra  ')
        assert result is not None
        assert result.command_name == 'status'
        assert result.arguments == 'extra'

    def test_uppercase_normalized(self) -> None:
        result = self.d.parse_slash_command('/HELP')
        assert result is not None
        assert result.command_name == 'help'

    def test_raw_input_preserved(self) -> None:
        raw = '  /help  '
        result = self.d.parse_slash_command(raw)
        assert result is not None
        assert result.raw_input == raw


# ---------------------------------------------------------------------------
# SlashCommandDispatcher — resolve
# ---------------------------------------------------------------------------


class TestResolveSlashCommand:
    def setup_method(self) -> None:
        self.d = _make_dispatcher_with_specs()

    def test_exact_match(self) -> None:
        r = self.d.resolve_slash_command('help')
        assert r.kind == 'exact'
        assert r.spec is not None

    def test_empty_returns_empty_kind(self) -> None:
        r = self.d.resolve_slash_command('')
        assert r.kind == 'empty'
        assert len(r.candidates) > 0

    def test_unknown_returns_none_kind(self) -> None:
        r = self.d.resolve_slash_command('zzz')
        assert r.kind == 'none'

    def test_unique_prefix_returns_prefix_kind(self) -> None:
        r = self.d.resolve_slash_command('he')
        assert r.kind == 'prefix'
        assert r.matched_name == 'help'

    def test_ambiguous_prefix_returns_ambiguous(self) -> None:
        # 'c' matches 'context' and 'clear'
        r = self.d.resolve_slash_command('c')
        assert r.kind == 'ambiguous'
        assert len(r.candidates) >= 2

    def test_exit_alias_quit(self) -> None:
        r = self.d.resolve_slash_command('quit')
        assert r.kind == 'exact'


# ---------------------------------------------------------------------------
# SlashCommandDispatcher — find
# ---------------------------------------------------------------------------


class TestFindSlashCommand:
    def setup_method(self) -> None:
        self.d = _make_dispatcher_with_specs()

    def test_find_existing(self) -> None:
        spec = self.d.find_slash_command('status')
        assert spec is not None
        assert 'status' in spec.names

    def test_find_missing_returns_none(self) -> None:
        assert self.d.find_slash_command('nonexistent') is None

    def test_find_case_insensitive(self) -> None:
        assert self.d.find_slash_command('STATUS') is not None


# ---------------------------------------------------------------------------
# SlashCommandDispatcher — dispatch (non-slash input)
# ---------------------------------------------------------------------------


class TestDispatchNonSlash:
    def setup_method(self) -> None:
        self.d = _make_dispatcher_with_specs()
        self.ctx = _make_slash_context()

    def test_plain_prompt_not_handled(self) -> None:
        result = self.d.dispatch_slash_command(self.ctx, 'hello')
        assert result.handled is False
        assert result.continue_query is True
        assert result.prompt == 'hello'

    def test_slash_only_shows_help(self) -> None:
        result = self.d.dispatch_slash_command(self.ctx, '/')
        assert result.handled is True
        assert result.command_name == 'help'

    def test_unknown_command_returns_error_metadata(self) -> None:
        result = self.d.dispatch_slash_command(self.ctx, '/xyzzy')
        assert result.handled is True
        assert result.metadata.get('error') == 'unknown_command'

    def test_ambiguous_command_returns_error_metadata(self) -> None:
        result = self.d.dispatch_slash_command(self.ctx, '/c')
        assert result.handled is True
        assert result.metadata.get('error') == 'ambiguous_command'

    def test_prefix_match_adds_metadata(self) -> None:
        result = self.d.dispatch_slash_command(self.ctx, '/he')
        assert result.metadata.get('match_mode') == 'prefix'
        assert result.metadata.get('matched_name') == 'help'


# ---------------------------------------------------------------------------
# SlashCommandDispatcher — built-in handlers
# ---------------------------------------------------------------------------


class TestHandleHelp:
    def setup_method(self) -> None:
        self.d = _make_dispatcher_with_specs()
        self.ctx = _make_slash_context()

    def test_handled_and_not_continue_query(self) -> None:
        result = self.d.dispatch_slash_command(self.ctx, '/help')
        assert result.handled is True
        assert result.continue_query is False

    def test_output_contains_all_commands(self) -> None:
        result = self.d.dispatch_slash_command(self.ctx, '/help')
        for name in ('help', 'status', 'context', 'permissions', 'tools', 'clear', 'exit', 'quit'):
            assert name in result.output or name in ('exit', 'quit')


class TestHandleStatus:
    def setup_method(self) -> None:
        self.d = _make_dispatcher_with_specs()

    def test_output_contains_model_name(self) -> None:
        ctx = _make_slash_context(model_name='claude-3')
        result = self.d.dispatch_slash_command(ctx, '/status')
        assert 'claude-3' in result.output

    def test_output_contains_workspace_path(self) -> None:
        ctx = _make_slash_context(workspace_path='/my/workspace')
        result = self.d.dispatch_slash_command(ctx, '/status')
        assert '/my/workspace' in result.output

    def test_output_contains_session_id(self) -> None:
        ctx = _make_slash_context(session_id='test-session')
        result = self.d.dispatch_slash_command(ctx, '/status')
        assert 'test-session' in result.output


class TestHandlePermissions:
    def setup_method(self) -> None:
        self.d = _make_dispatcher_with_specs()

    def test_file_write_disabled_by_default(self) -> None:
        ctx = _make_slash_context(permissions=PermissionPolicy(allow_file_write=False))
        result = self.d.dispatch_slash_command(ctx, '/permissions')
        assert 'no' in result.output.lower()

    def test_file_write_enabled(self) -> None:
        ctx = _make_slash_context(permissions=PermissionPolicy(allow_file_write=True))
        result = self.d.dispatch_slash_command(ctx, '/permissions')
        assert 'File write: yes' in result.output


class TestHandleMcp:
    def test_mcp_without_tools_gateway(self) -> None:
        d = _make_dispatcher_with_specs()
        ctx = _make_slash_context()
        result = d.dispatch_slash_command(ctx, '/mcp')
        assert 'MCP gateway not available' in result.output

    def test_mcp_no_servers(self) -> None:
        mock_gw = MagicMock()
        mock_gw.get_mcp_server_summaries.return_value = ()
        specs = build_default_slash_command_specs(tools_gateway=mock_gw)
        d = SlashCommandDispatcher(specs=specs)
        ctx = _make_slash_context()
        result = d.dispatch_slash_command(ctx, '/mcp')
        assert 'No MCP servers configured' in result.output

    def test_mcp_server_list(self) -> None:
        from src.core_contracts.tools_contracts import McpServerSummary
        mock_gw = MagicMock()
        mock_gw.get_mcp_server_summaries.return_value = (
            McpServerSummary(name='test-server', transport='stdio', tool_count=5, status='connected'),
            McpServerSummary(name='bad-server', transport='streamable-http', tool_count=0, status='error', error_message='timeout'),
        )
        specs = build_default_slash_command_specs(tools_gateway=mock_gw)
        d = SlashCommandDispatcher(specs=specs)
        ctx = _make_slash_context()
        result = d.dispatch_slash_command(ctx, '/mcp')
        assert 'test-server' in result.output
        assert 'stdio' in result.output
        assert '5 tools' in result.output
        assert 'connected' in result.output
        assert 'bad-server' in result.output
        assert 'timeout' in result.output

    def test_mcp_server_detail(self) -> None:
        from src.core_contracts.tools_contracts import McpServerSummary
        tool = ToolDescriptor(
            name='my_mcp_tool',
            description='A test MCP tool',
            parameters={},
            handler=lambda _req: None,  # type: ignore[arg-type, return-value]
            server_name='test-server',
        )
        mock_gw = MagicMock()
        mock_gw.get_mcp_server_summaries.return_value = (
            McpServerSummary(name='test-server', transport='stdio', tool_count=1, status='connected'),
        )
        mock_gw.list_tools.return_value = [tool]
        specs = build_default_slash_command_specs(tools_gateway=mock_gw)
        d = SlashCommandDispatcher(specs=specs)
        ctx = _make_slash_context()
        result = d.dispatch_slash_command(ctx, '/mcp test-server')
        assert 'test-server' in result.output
        assert 'my_mcp_tool' in result.output

    def test_mcp_server_not_found(self) -> None:
        mock_gw = MagicMock()
        mock_gw.get_mcp_server_summaries.return_value = ()
        specs = build_default_slash_command_specs(tools_gateway=mock_gw)
        d = SlashCommandDispatcher(specs=specs)
        ctx = _make_slash_context()
        result = d.dispatch_slash_command(ctx, '/mcp unknown')
        assert 'Server not found' in result.output


class TestHandleClear:
    def setup_method(self) -> None:
        self.d = _make_dispatcher_with_specs()

    def test_fork_session_is_true(self) -> None:
        ctx = _make_slash_context()
        result = self.d.dispatch_slash_command(ctx, '/clear')
        assert result.fork_session is True

    def test_replacement_session_state_is_fresh(self) -> None:
        ctx = _make_slash_context()
        result = self.d.dispatch_slash_command(ctx, '/clear')
        assert result.replacement_session_state is not None
        assert result.replacement_session_state.messages == []

    def test_had_history_false_when_empty(self) -> None:
        ctx = _make_slash_context()
        result = self.d.dispatch_slash_command(ctx, '/clear')
        assert result.metadata.get('had_history') is False

    def test_had_history_true_when_has_turns(self) -> None:
        ctx = _make_slash_context(turns_offset=3)
        result = self.d.dispatch_slash_command(ctx, '/clear')
        assert result.metadata.get('had_history') is True


class TestHandleExit:
    def setup_method(self) -> None:
        self.d = _make_dispatcher_with_specs()
        self.ctx = _make_slash_context()

    def test_exit_sets_exit_requested(self) -> None:
        result = self.d.dispatch_slash_command(self.ctx, '/exit')
        assert result.metadata.get('exit_requested') is True

    def test_quit_alias_works(self) -> None:
        result = self.d.dispatch_slash_command(self.ctx, '/quit')
        assert result.metadata.get('exit_requested') is True

    def test_exit_not_continue_query(self) -> None:
        result = self.d.dispatch_slash_command(self.ctx, '/exit')
        assert result.continue_query is False


class TestHandleContext:
    def setup_method(self) -> None:
        self.dispatcher_no_ctx = SlashCommandDispatcher(
            specs=build_default_slash_command_specs(context_gateway=None)
        )

    def test_no_context_manager_returns_degraded_message(self) -> None:
        ctx = _make_slash_context()
        result = self.dispatcher_no_ctx.dispatch_slash_command(ctx, '/context')
        assert result.handled is True
        assert 'not available' in result.output.lower()

    def test_with_context_manager_returns_budget_info(self) -> None:
        from unittest.mock import MagicMock
        mock_ctx_manager = MagicMock()
        mock_ctx_manager.project_budget.return_value = BudgetProjection(
            projected_input_tokens=1000,
            output_reserve_tokens=256,
            hard_input_limit=None,
            soft_input_limit=None,
            is_hard_over=False,
            is_soft_over=False,
        )
        dispatcher = SlashCommandDispatcher(
            specs=build_default_slash_command_specs(context_gateway=mock_ctx_manager)
        )
        ctx = _make_slash_context()
        result = dispatcher.dispatch_slash_command(ctx, '/context')
        assert result.handled is True
        assert '1000' in result.output


# ---------------------------------------------------------------------------
# SessionInteractionTracker
# ---------------------------------------------------------------------------


class TestSessionInteractionTracker:
    def test_start_sets_session_id(self) -> None:
        tracker = SessionInteractionTracker.start('initial-id')
        assert tracker.session_id == 'initial-id'

    def test_start_sets_started_time(self) -> None:
        tracker = SessionInteractionTracker.start()
        assert tracker.started_time > 0

    def test_observe_tool_result_ok_increments_successes(self) -> None:
        tracker = SessionInteractionTracker.start()
        tracker.observe_tool_result(ok=True)
        assert tracker.tool_calls == 1
        assert tracker.tool_successes == 1
        assert tracker.tool_failures == 0

    def test_observe_tool_result_fail_increments_failures(self) -> None:
        tracker = SessionInteractionTracker.start()
        tracker.observe_tool_result(ok=False)
        assert tracker.tool_failures == 1
        assert tracker.tool_successes == 0

    def test_observe_run_result_updates_session_id(self) -> None:
        tracker = SessionInteractionTracker.start()
        result = AgentRunResult(session_id='new-id', events=())
        tracker.observe_run_result(result, current_session_id='fallback')
        assert tracker.session_id == 'new-id'

    def test_observe_run_result_uses_fallback_if_no_session_id(self) -> None:
        tracker = SessionInteractionTracker.start()
        result = AgentRunResult(session_id=None, events=())
        tracker.observe_run_result(result, current_session_id='fallback')
        assert tracker.session_id == 'fallback'

    def test_observe_run_result_counts_tool_events(self) -> None:
        tracker = SessionInteractionTracker.start()
        events: tuple[JSONDict, ...] = (
            {'type': 'tool_result', 'ok': True},
            {'type': 'tool_result', 'ok': False},
            {'type': 'model_turn'},
        )
        result = AgentRunResult(session_id='s1', events=events)
        tracker.observe_run_result(result, current_session_id=None)
        assert tracker.tool_calls == 2
        assert tracker.tool_successes == 1
        assert tracker.tool_failures == 1

    def test_to_summary_returns_session_summary(self) -> None:
        tracker = SessionInteractionTracker.start('abc')
        tracker.observe_tool_result(ok=True)
        summary = tracker.to_summary()
        assert isinstance(summary, SessionSummary)
        assert summary.session_id == 'abc'
        assert summary.tool_calls == 1

    def test_to_summary_wall_time_positive(self) -> None:
        tracker = SessionInteractionTracker.start()
        summary = tracker.to_summary()
        assert summary.wall_time_seconds >= 0.0

    def test_success_rate_all_success(self) -> None:
        tracker = SessionInteractionTracker.start()
        tracker.observe_tool_result(ok=True)
        tracker.observe_tool_result(ok=True)
        summary = tracker.to_summary()
        assert summary.success_rate == 1.0

    def test_success_rate_no_calls(self) -> None:
        tracker = SessionInteractionTracker.start()
        summary = tracker.to_summary()
        assert summary.success_rate == 0.0

    def test_update_session_id_ignores_none(self) -> None:
        tracker = SessionInteractionTracker.start('original')
        tracker.update_session_id(None)
        assert tracker.session_id == 'original'


# ---------------------------------------------------------------------------
# InteractionGateway — delegation
# ---------------------------------------------------------------------------


class TestInteractionGatewayDelegation:
    def test_render_startup_delegates_to_startup_renderer(self) -> None:
        stream = io.StringIO()
        gw = _make_gateway(stream=stream)
        env = EnvironmentLoadSummary(mcp_servers=1)
        gw.render_startup(stream=stream, environment_summary=env)
        # 只要不抛出即为通过

    def test_render_exit_delegates_to_exit_renderer(self) -> None:
        stream = io.StringIO()
        gw = _make_gateway(stream=stream)
        summary = SessionSummary(session_id='s1', tool_calls=2, tool_successes=2,
                                 tool_failures=0, wall_time_seconds=1.5)
        gw.render_exit(summary, stream=stream)

    def test_render_slash_result_delegates(self) -> None:
        stream = io.StringIO()
        gw = _make_gateway(stream=stream)
        gw.render_slash_result(command_name='help', output='some help', stream=stream)

    def test_dispatch_slash_command_delegates(self) -> None:
        stream = io.StringIO()
        d = _make_dispatcher_with_specs()
        gw = _make_gateway(dispatcher=d, stream=stream)
        ctx = _make_slash_context()
        result = gw.dispatch_slash_command(ctx, '/help')
        assert result.handled is True

    def test_parse_slash_command_delegates(self) -> None:
        gw = _make_gateway()
        assert gw.parse_slash_command('/help') is not None
        assert gw.parse_slash_command('hello') is None

    def test_resolve_slash_command_delegates(self) -> None:
        d = _make_dispatcher_with_specs()
        gw = _make_gateway(dispatcher=d)
        r = gw.resolve_slash_command('help')
        assert r.kind == 'exact'

    def test_find_slash_command_delegates(self) -> None:
        d = _make_dispatcher_with_specs()
        gw = _make_gateway(dispatcher=d)
        assert gw.find_slash_command('status') is not None

    def test_get_slash_command_specs_delegates(self) -> None:
        d = _make_dispatcher_with_specs()
        gw = _make_gateway(dispatcher=d)
        specs = gw.get_slash_command_specs()
        assert len(specs) > 0

    def test_get_autocomplete_entries_returns_entries(self) -> None:
        d = _make_dispatcher_with_specs()
        gw = _make_gateway(dispatcher=d)
        entries = gw.get_autocomplete_entries()
        names = {e.name for e in entries}
        assert 'help' in names
        assert 'exit' in names
        assert 'quit' in names


class TestInteractionGatewaySessionTracking:
    def test_get_session_summary_before_start_returns_default(self) -> None:
        gw = _make_gateway()
        summary = gw.get_session_summary()
        assert summary.session_id is None
        assert summary.tool_calls == 0

    def test_start_session_tracker_initializes_tracker(self) -> None:
        gw = _make_gateway()
        gw.start_session_tracker('s1')
        assert gw._session_tracker is not None
        assert gw._session_tracker.session_id == 's1'

    def test_observe_run_result_no_op_without_tracker(self) -> None:
        gw = _make_gateway()
        result = AgentRunResult(session_id='x', events=())
        # Should not raise
        gw.observe_run_result(result, current_session_id='x')

    def test_observe_run_result_updates_tracker(self) -> None:
        gw = _make_gateway()
        gw.start_session_tracker()
        events: tuple[JSONDict, ...] = ({'type': 'tool_result', 'ok': True},)
        result = AgentRunResult(session_id='new-s', events=events)
        gw.observe_run_result(result, current_session_id='fallback')
        summary = gw.get_session_summary()
        assert summary.session_id == 'new-s'
        assert summary.tool_calls == 1

    def test_get_session_summary_after_start(self) -> None:
        gw = _make_gateway()
        gw.start_session_tracker('sess-abc')
        summary = gw.get_session_summary()
        assert summary.session_id == 'sess-abc'

    def test_build_progress_reporter_returns_callable(self) -> None:
        stream = io.StringIO()
        gw = _make_gateway(stream=stream)
        reporter = gw.build_progress_reporter()
        assert callable(reporter)

    def test_flush_runtime_events_does_not_raise(self) -> None:
        gw = _make_gateway()
        gw.flush_runtime_events()

    def test_read_input_uses_fallback_reader(self) -> None:
        captured: list[str] = []

        def reader(prompt: str) -> str:
            captured.append(prompt)
            return 'user input'

        stream = io.StringIO()
        d = SlashCommandDispatcher()
        prompt = SlashAutocompletePrompt(entries=(), fallback_reader=reader)
        gw = InteractionGateway(
            dispatcher=d,
            startup_renderer=StartupRenderer(),
            exit_renderer=ExitRenderer(),
            slash_renderer=SlashCommandRenderer(),
            event_printer=RuntimeEventPrinter(stream=stream),
            autocomplete_prompt=prompt,
            stream=stream,
        )
        result = gw.read_input('Enter> ')
        assert result == 'user input'
        assert 'Enter> ' in captured


# ---------------------------------------------------------------------------
# create_interaction_gateway factory
# ---------------------------------------------------------------------------


class TestCreateInteractionGatewayFactory:
    def test_returns_interaction_gateway(self) -> None:
        gw = create_interaction_gateway(
            slash_specs=build_default_slash_command_specs(),
            stream=io.StringIO(),
        )
        assert isinstance(gw, InteractionGateway)

    def test_specs_are_loaded(self) -> None:
        gw = create_interaction_gateway(
            slash_specs=build_default_slash_command_specs(),
            stream=io.StringIO(),
        )
        specs = gw.get_slash_command_specs()
        assert len(specs) > 0

    def test_autocomplete_entries_cover_all_commands(self) -> None:
        gw = create_interaction_gateway(
            slash_specs=build_default_slash_command_specs(),
            stream=io.StringIO(),
        )
        names = {e.name for e in gw.get_autocomplete_entries()}
        for name in ('help', 'context', 'status', 'permissions', 'mcp', 'clear', 'exit', 'quit'):
            assert name in names

    def test_custom_exit_title_used_in_render(self) -> None:
        stream = io.StringIO()
        gw = create_interaction_gateway(
            slash_specs=build_default_slash_command_specs(),
            stream=stream,
            exit_title='Goodbye custom!',
        )
        summary = SessionSummary(session_id=None, tool_calls=0, tool_successes=0,
                                 tool_failures=0, wall_time_seconds=0.1)
        gw.render_exit(summary, stream=stream)
        output = stream.getvalue()
        assert 'Goodbye custom!' in output
