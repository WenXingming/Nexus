"""interaction 模块跨边界契约定义。

定义 interaction 模块对外暴露的全部 DTO、运行态协议与辅助类型别名。
外部调用方与 interaction 模块的交互**只能**基于本文件定义的类型；
禁止直接依赖 src/interaction/ 内的任何具体实现类。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol

from src.core_contracts.context_contracts import BudgetConfig, ContextPolicy
from src.core_contracts.model_config import ModelConfig
from src.core_contracts.session_contracts import SessionState
from src.core_contracts.tools_contracts import ToolDescriptor

# ---------------------------------------------------------------------------
# 通用类型别名
# ---------------------------------------------------------------------------

type JSONDict = dict[str, object]
"""运行期事件与元数据的字典类型。"""

type WorkspaceScope = str
"""工作区路径（当前工作目录字符串）。"""


# ---------------------------------------------------------------------------
# 配置与权限 DTO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermissionPolicy:
    """工具执行权限策略配置。

    由外部运行时持有并通过 SlashCommandContext 传入；
    interaction 内部只读，不做任何运行时修改。
    """

    allow_file_write: bool = False
    """bool: 是否允许文件写入操作。"""

    allow_shell_commands: bool = False
    """bool: 是否允许执行 shell 命令。"""

    allow_destructive_shell_commands: bool = False
    """bool: 是否允许执行破坏性 shell 命令（如 rm -rf）。"""


# ---------------------------------------------------------------------------
# 运行结果 DTO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentRunResult:
    """单轮 agent 执行结果的最小跨模块契约。

    interaction 模块通过本 DTO 从外部接收每轮运行产物，
    用于更新会话统计追踪器。
    """

    session_id: str | None = None
    """str | None: 本轮执行产生或确认的活动会话 ID。"""

    events: tuple[JSONDict, ...] = field(default_factory=tuple)
    """tuple[JSONDict, ...]: 本轮产生的结构化事件序列（如 tool_result）。"""


# ---------------------------------------------------------------------------
# 环境加载摘要
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvironmentLoadSummary:
    """描述交互式启动阶段已发现的环境加载结果。"""

    mcp_servers: int = 0
    """int：发现的 MCP 服务器数量。"""

    plugins: int = 0
    """int：发现的插件数量。"""

    hook_policies: int = 0
    """int：发现的 hook 策略数量。"""

    search_providers: int = 0
    """int：发现的搜索提供商数量。"""

    load_errors: int = 0
    """int：加载阶段的错误数量。"""

    def render_line(self) -> str:
        """渲染单行环境摘要文本。

        Returns:
            str: 可直接输出的一行摘要；无正向统计时返回空字符串。
        """
        parts = self._build_summary_parts()
        if not parts:
            return ''
        return f'Environment loaded: {", ".join(parts)}'

    def _build_summary_parts(self) -> tuple[str, ...]:
        """按固定顺序构建环境摘要片段。

        Returns:
            tuple[str, ...]: 已格式化摘要片段元组。
        """
        parts: list[str] = []
        self._append_count_part(parts, self.mcp_servers, singular='MCP server', plural='MCP servers')
        self._append_count_part(parts, self.plugins, singular='plugin', plural='plugins')
        self._append_count_part(parts, self.hook_policies, singular='hook policy', plural='hook policies')
        self._append_count_part(parts, self.search_providers, singular='search provider', plural='search providers')
        self._append_count_part(parts, self.load_errors, singular='load error', plural='load errors')
        return tuple(parts)

    @staticmethod
    def _append_count_part(parts: list[str], count: int, *, singular: str, plural: str) -> None:
        """把单个计数片段追加到摘要列表中。

        Args:
            parts (list[str]): 摘要片段累积列表。
            count (int): 当前统计值。
            singular (str): 单数名词。
            plural (str): 复数名词。
        Returns:
            None
        """
        if count <= 0:
            return
        noun = singular if count == 1 else plural
        parts.append(f'{count} {noun}')


# ---------------------------------------------------------------------------
# 会话摘要
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionSummary:
    """表示一次 CLI 交互结束时的只读汇总快照。"""

    session_id: str | None = None
    """str | None：会话 ID。"""

    tool_calls: int = 0
    """int：累计工具调用次数。"""

    tool_successes: int = 0
    """int：成功工具调用次数。"""

    tool_failures: int = 0
    """int：失败工具调用次数。"""

    wall_time_seconds: float = 0.0
    """float：挂钟耗时，单位秒。"""

    @property
    def success_rate(self) -> float:
        """返回工具调用成功率。

        Returns:
            float: 成功率，tool_calls 为 0 时返回 0.0。
        """
        if self.tool_calls <= 0:
            return 0.0
        return self.tool_successes / self.tool_calls


# ---------------------------------------------------------------------------
# slash 自动补全条目
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlashAutocompleteEntry:
    """描述一个可补全的 slash 命令项。"""

    name: str
    """str：命令名称（不含前导斜杠）。"""

    description: str
    """str：命令说明，面向自动补全菜单展示。"""


# ---------------------------------------------------------------------------
# slash 命令上下文与结果 DTO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlashCommandContext:
    """封装 slash 命令执行期间所需的只读运行时上下文。

    由外部 ChatLoop（或等价协调器）在调用 dispatch_slash_command 前装配，
    interaction 内部只读。
    """

    session_state: SessionState
    """SessionState：当前会话状态（消息历史与转录条目）。"""

    session_id: str
    """str：当前活动会话 ID。"""

    turns_offset: int
    """int：已完成的 turn 轮次偏移量。"""

    tool_call_count: int
    """int：累计工具调用次数。"""

    workspace_path: WorkspaceScope
    """WorkspaceScope：当前工作目录路径字符串。"""

    context_policy: ContextPolicy
    """ContextPolicy：上下文治理策略配置。"""

    permissions: PermissionPolicy
    """PermissionPolicy：工具执行权限策略。"""

    budget_config: BudgetConfig
    """BudgetConfig：token 预算硬性约束配置。"""

    model_config: ModelConfig
    """ModelConfig：当前模型连接与行为配置。"""

    tool_registry: tuple[ToolDescriptor, ...]
    """tuple[ToolDescriptor, ...]: 当前注册的全部工具描述符列表。"""

    plugin_summary: str = ''
    """str：插件加载摘要文本，供 /tools 命令展示。"""


@dataclass(frozen=True)
class SlashCommandResult:
    """描述一次 slash 分流后的处理结果。"""

    handled: bool
    """bool：命令是否已被处理。"""

    continue_query: bool
    """bool：是否继续发起模型查询。"""

    command_name: str = ''
    """str：匹配到的命令名称。"""

    output: str = ''
    """str：命令执行输出文本。"""

    prompt: str | None = None
    """str | None：注入的用户 prompt（非 slash 输入透传时使用）。"""

    replacement_session_state: SessionState | None = None
    """SessionState | None：需要替换的会话状态（/clear 时产生）。"""

    fork_session: bool = False
    """bool：是否开启新会话分支。"""

    metadata: JSONDict = field(default_factory=dict)
    """JSONDict：额外元数据，如错误码、匹配模式等。"""


# ---------------------------------------------------------------------------
# slash 命令解析与规格 DTO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedSlashCommand:
    """表示一次成功解析的 slash 输入。"""

    command_name: str
    """str: 规范化后的命令名，不包含前导斜杠，已转为小写。"""

    arguments: str
    """str: 命令后的原始参数文本，保留空格折叠后的用户输入。"""

    raw_input: str
    """str: 用户提交的原始输入，供日志与回显复用。"""


SlashHandler = Callable[[SlashCommandContext, ParsedSlashCommand], SlashCommandResult]
"""slash 命令处理器函数签名。"""


@dataclass(frozen=True)
class SlashCommandSpec:
    """定义单个 slash 命令的名称、描述与处理器。"""

    names: tuple[str, ...]
    """tuple[str, ...]: 当前命令支持的全部名称与别名（不含前导斜杠）。"""

    description: str
    """str: 面向 /help 输出的人类可读描述。"""

    handler: SlashHandler
    """SlashHandler: 真正执行业务逻辑的命令处理函数。"""


@dataclass(frozen=True)
class SlashCommandResolution:
    """表示一次 slash 命令匹配的解析结果。"""

    kind: Literal['exact', 'prefix', 'ambiguous', 'none', 'empty']
    """Literal: 解析结果类型标签。"""

    spec: SlashCommandSpec | None = None
    """SlashCommandSpec | None: 匹配到的命令规格；歧义或未找到时为 None。"""

    matched_name: str = ''
    """str: 实际匹配到的规范命令名。"""

    candidates: tuple[str, ...] = ()
    """tuple[str, ...]: 歧义或补全时的候选命令名列表。"""


# ---------------------------------------------------------------------------
# 协议接口
# ---------------------------------------------------------------------------


class SlashDispatcher(Protocol):
    """slash 命令分发器的最小跨域协议。

    实现该协议的类可直接插入 ChatLoop 的 slash_dispatcher 依赖槽，
    无需了解具体实现细节。
    """

    def dispatch_slash_command(
        self,
        context: SlashCommandContext,
        input_text: str,
    ) -> SlashCommandResult:
        """分发一条 slash 输入并返回处理结果。

        Args:
            context (SlashCommandContext): slash 命令执行所需的只读上下文。
            input_text (str): 用户提交的原始输入文本。
        Returns:
            SlashCommandResult: 分流结果。
        """
        ...


__all__ = [
    'JSONDict',
    'WorkspaceScope',
    'PermissionPolicy',
    'AgentRunResult',
    'EnvironmentLoadSummary',
    'SessionSummary',
    'SlashAutocompleteEntry',
    'SlashCommandContext',
    'SlashCommandResult',
    'ParsedSlashCommand',
    'SlashHandler',
    'SlashCommandSpec',
    'SlashCommandResolution',
    'SlashDispatcher',
]
