"""应用层 slash 命令装配。

本模块位于 interaction 之外，负责把具体业务命令处理器装配为
SlashCommandSpec，并由组合根注入 InteractionGateway。
"""

from __future__ import annotations

from src.context.context_gateway import ContextGateway
from src.core_contracts.interaction_contracts import (
    ParsedSlashCommand,
    SlashCommandContext,
    SlashCommandResult,
    SlashCommandSpec,
)
from src.core_contracts.client_contracts import LlmRequest
from src.core_contracts.rag_contracts import (
    RagError,
    RagIndexRequest,
)
from src.core_contracts.session_contracts import SessionState
from src.rag import RagGateway
from src.session import SessionGateway

_MAIN_LOOP_COLLECTION = 'main-loop'


def build_default_slash_command_specs(
    *,
    context_gateway: ContextGateway | None = None,
    session_gateway: SessionGateway | None = None,
    rag_gateway: RagGateway | None = None,
    client_gateway = None,
) -> tuple[SlashCommandSpec, ...]:
    """构造应用默认的 slash 命令规格列表。"""
    registry = SlashCommandRegistry(
        context_gateway=context_gateway,
        session_gateway=session_gateway,
        rag_gateway=rag_gateway,
        client_gateway=client_gateway,
    )
    return registry.get_specs()


class SlashCommandRegistry:
    """Slash 命令注册表。

    职责：
    1. 接收依赖注入（Gateway）
    2. 注册所有命令处理器
    3. 提供命令规格列表
    """

    def __init__(
        self,
        *,
        context_gateway: ContextGateway | None = None,
        session_gateway: SessionGateway | None = None,
        rag_gateway: RagGateway | None = None,
        client_gateway = None,
    ):
        self._context_gateway = context_gateway
        self._session_gateway = session_gateway
        self._rag_gateway = rag_gateway
        self._client_gateway = client_gateway
        self._specs = self._register_commands()

    def get_specs(self) -> tuple[SlashCommandSpec, ...]:
        """获取所有命令规格。"""
        return self._specs

    def _register_commands(self) -> tuple[SlashCommandSpec, ...]:
        """注册所有命令处理器。"""
        return (
            SlashCommandSpec(
                names=('help',),
                description='Show supported local slash commands.',
                handler=self._handle_help,
            ),
            SlashCommandSpec(
                names=('context',),
                description='Show local context status.',
                handler=self._handle_context,
            ),
            SlashCommandSpec(
                names=('status',),
                description='Show current session status.',
                handler=self._handle_status,
            ),
            SlashCommandSpec(
                names=('permissions',),
                description='Show current tool permissions.',
                handler=self._handle_permissions,
            ),
            SlashCommandSpec(
                names=('tools',),
                description='List registered local tools.',
                handler=self._handle_tools,
            ),
            SlashCommandSpec(
                names=('clear',),
                description='Fork a new cleared session snapshot.',
                handler=self._handle_clear,
            ),
            SlashCommandSpec(
                names=('exit', 'quit'),
                description='Stop local interaction and return to caller.',
                handler=self._handle_exit,
            ),
            SlashCommandSpec(
                names=('new',),
                description='Create a new session.',
                handler=self._handle_new,
            ),
            SlashCommandSpec(
                names=('save',),
                description='Save current session.',
                handler=self._handle_save,
            ),
            SlashCommandSpec(
                names=('load',),
                description='Load a session by ID.',
                handler=self._handle_load,
            ),
            SlashCommandSpec(
                names=('rag-index',),
                description='Index text documents from file or directory.',
                handler=self._handle_rag_index,
            ),
            SlashCommandSpec(
                names=('rag-ask',),
                description='Query RAG collection with retrieval-augmented generation.',
                handler=self._handle_rag_ask,
            ),
        )

    def _handle_help(
        self,
        context: SlashCommandContext,
        parsed: ParsedSlashCommand,
    ) -> SlashCommandResult:
        del context, parsed
        lines = ['Slash Commands', '==============', '']
        for spec in self._specs:
            lines.append(f'/{spec.names[0]} - {spec.description}')
        lines.extend(['', 'Tip: input / to list commands, or use a unique prefix such as /st.'])
        return SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name='help',
            output='\n'.join(lines),
        )

    def _handle_context(
        self,
        context: SlashCommandContext,
        parsed: ParsedSlashCommand,
    ) -> SlashCommandResult:
        del parsed
        if self._context_gateway is None:
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='context',
                output=(
                    'Context Status\n'
                    '==============\n'
                    'Context gateway not available.\n'
                    'Provide a ContextGateway when assembling slash command handlers to enable this command.'
                ),
            )
        openai_tools = [tool.to_openai_tool() for tool in context.tool_registry]
        snapshot = self._context_gateway.project_budget(
            context.session_state.messages,
            tools=openai_tools,
            budget_config=context.budget_config,
        )
        lines = [
            'Context Status',
            '==============',
            f'Messages: {len(context.session_state.messages)}',
            f'Transcript entries: {len(context.session_state.transcript_entries)}',
            f'Tool calls: {context.tool_call_count}',
            f'Projected input tokens: {snapshot.projected_input_tokens}',
            f'Hard input limit: {self._render_optional_int(snapshot.hard_input_limit)}',
            f'Soft input limit: {self._render_optional_int(snapshot.soft_input_limit)}',
            f'Is soft over: {self._render_bool(snapshot.is_soft_over)}',
            f'Is hard over: {self._render_bool(snapshot.is_hard_over)}',
            f'Compact preserve messages: {context.context_policy.compact_preserve_messages}',
        ]
        return SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name='context',
            output='\n'.join(lines),
        )

    def _handle_status(
        self,
        context: SlashCommandContext,
        parsed: ParsedSlashCommand,
    ) -> SlashCommandResult:
        del parsed
        lines = [
            'Session Status',
            '==============',
            f'Session id: {context.session_id}',
            f'Model: {context.model_config.model_name}',
            f'Working directory: {context.workspace_path}',
            f'Completed turns: {context.turns_offset}',
            f'Tool calls: {context.tool_call_count}',
        ]
        return SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name='status',
            output='\n'.join(lines),
        )

    def _handle_permissions(
        self,
        context: SlashCommandContext,
        parsed: ParsedSlashCommand,
    ) -> SlashCommandResult:
        del parsed
        permissions = context.permissions
        lines = [
            'Permissions',
            '===========',
            f'File write: {self._render_bool(permissions.allow_file_write)}',
            f'Shell commands: {self._render_bool(permissions.allow_shell_commands)}',
            f'Destructive shell: {self._render_bool(permissions.allow_destructive_shell_commands)}',
        ]
        return SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name='permissions',
            output='\n'.join(lines),
        )

    def _handle_tools(
        self,
        context: SlashCommandContext,
        parsed: ParsedSlashCommand,
    ) -> SlashCommandResult:
        del parsed
        permissions = context.permissions
        lines = [
            'Registered Tools',
            '================',
            f'File write enabled: {self._render_bool(permissions.allow_file_write)}',
            f'Shell enabled: {self._render_bool(permissions.allow_shell_commands)}',
            '',
        ]
        for tool in context.tool_registry:
            lines.append(f'{tool.name} - {tool.description}')
        if context.plugin_summary.strip():
            lines.extend(['', context.plugin_summary.strip()])
        return SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name='tools',
            output='\n'.join(lines),
        )

    def _handle_clear(
        self,
        context: SlashCommandContext,
        parsed: ParsedSlashCommand,
    ) -> SlashCommandResult:
        del parsed
        had_history = bool(
            context.session_state.messages
            or context.session_state.transcript_entries
            or context.tool_call_count
            or context.turns_offset
        )
        return SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name='clear',
            output='Cleared in-memory session context.',
            replacement_session_state=SessionState(),
            fork_session=True,
            metadata={'had_history': had_history},
        )

    def _handle_exit(
        self,
        context: SlashCommandContext,
        parsed: ParsedSlashCommand,
    ) -> SlashCommandResult:
        del context
        return SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name=parsed.command_name,
            output='Exiting local session interaction.',
            metadata={'exit_requested': True},
        )

    def _handle_new(
        self,
        context: SlashCommandContext,
        parsed: ParsedSlashCommand,
    ) -> SlashCommandResult:
        del context, parsed
        if self._session_gateway is None:
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='new',
                output='Session gateway not available.',
                metadata={'error': 'session_gateway_missing'},
            )
        new_state = self._session_gateway.create_state('新会话已创建')
        return SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name='new',
            output=f'[新会话] session_id={new_state.session_id}',
            replacement_session_state=new_state,
        )

    def _handle_save(
        self,
        context: SlashCommandContext,
        parsed: ParsedSlashCommand,
    ) -> SlashCommandResult:
        del parsed
        if self._session_gateway is None:
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='save',
                output='Session gateway not available.',
                metadata={'error': 'session_gateway_missing'},
            )
        session_id, path = self._session_gateway.save_state(
            state=context.session_state,
            model_config=context.model_config,
        )
        return SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name='save',
            output=f'[已保存] session_id={session_id}, path={path}',
        )

    def _handle_load(
        self,
        context: SlashCommandContext,
        parsed: ParsedSlashCommand,
    ) -> SlashCommandResult:
        del context
        if self._session_gateway is None:
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='load',
                output='Session gateway not available.',
                metadata={'error': 'session_gateway_missing'},
            )
        session_id = parsed.arguments.strip()
        if not session_id:
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='load',
                output='[错误] 用法: /load <session_id>',
                metadata={'error': 'missing_argument'},
            )
        try:
            snapshot = self._session_gateway.load(session_id)
            new_state = self._session_gateway.resume_state(
                session_id=session_id,
                messages=snapshot.messages,
                transcript=snapshot.transcript,
            )
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='load',
                output=f'[已加载] session_id={session_id}, 共 {len(snapshot.messages)} 条消息',
                replacement_session_state=new_state,
            )
        except FileNotFoundError:
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='load',
                output=f'[错误] 会话不存在: {session_id}',
                metadata={'error': 'session_not_found'},
            )
        except Exception as exc:
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='load',
                output=f'[错误] 加载失败: {exc}',
                metadata={'error': 'load_failed'},
            )

    def _handle_rag_index(
        self,
        context: SlashCommandContext,
        parsed: ParsedSlashCommand,
    ) -> SlashCommandResult:
        del context
        if self._rag_gateway is None:
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='rag-index',
                output='RAG gateway not available.',
                metadata={'error': 'rag_gateway_missing'},
            )
        source = parsed.arguments.strip()
        if not source:
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='rag-index',
                output='[错误] 用法: /rag-index <file-or-dir>',
                metadata={'error': 'missing_argument'},
            )
        try:
            result = self._rag_gateway.index(
                RagIndexRequest(
                    source_path=source,
                    collection_name=_MAIN_LOOP_COLLECTION,
                )
            )
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='rag-index',
                output=(
                    '[RAG 已索引] '
                    f'collection={result.collection_name}, docs={result.docs_indexed}, '
                    f'chunks={result.chunks_created}'
                ),
            )
        except Exception as exc:
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='rag-index',
                output=f'[错误] RAG 索引失败: {exc}',
                metadata={'error': 'index_failed'},
            )

    def _handle_rag_ask(
        self,
        context: SlashCommandContext,
        parsed: ParsedSlashCommand,
    ) -> SlashCommandResult:
        if self._rag_gateway is None:
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='rag-ask',
                output='RAG gateway not available.',
                metadata={'error': 'rag_gateway_missing'},
            )
        question = parsed.arguments.strip()
        if not question:
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='rag-ask',
                output='[错误] 用法: /rag-ask <question>',
                metadata={'error': 'missing_argument'},
            )
        try:
            if self._session_gateway is not None:
                self._session_gateway.append_user(context.session_state, question)
            messages = self._rag_gateway.retrieve_and_build_messages(
                query=question,
                collection_name=_MAIN_LOOP_COLLECTION,
            )
            llm_result = self._client_gateway.chat(
                LlmRequest(messages=messages, max_tokens=1024)
            )
            answer = llm_result.content.strip()
            if self._session_gateway is not None:
                self._session_gateway.append_assistant(context.session_state, answer)
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='rag-ask',
                output=f'\nRAG> {answer}',
            )
        except RagError as exc:
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='rag-ask',
                output=f'[错误] RAG 查询失败: {exc}',
                metadata={'error': 'rag_error'},
            )
        except Exception as exc:
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='rag-ask',
                output=f'[错误] RAG 问答失败: {exc}',
                metadata={'error': 'query_failed'},
            )

    def _render_optional_int(self, value: int | None) -> str:
        if value is None:
            return 'unlimited'
        return str(value)

    def _render_bool(self, value: bool) -> str:
        return 'yes' if value else 'no'
