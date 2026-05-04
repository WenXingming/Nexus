"""interaction 包公开入口。

架构约定：
- 对外仅暴露 InteractionGateway 与 create_interaction_gateway 工厂函数。
- 所有外部消费者必须通过本入口访问交互能力；禁止跨包直接依赖内部子模块。
- 白盒测试如需访问内部实现，应直接从对应内部模块导入。
"""

from __future__ import annotations

import sys
from typing import TextIO

from src.core_contracts.interaction_contracts import SlashAutocompleteEntry, SlashCommandSpec

from src.interaction.interaction_gateway import InteractionGateway
from src.interaction.quit_render import ExitRenderer as _ExitRenderer
from src.interaction.runtime_event_printer import RuntimeEventPrinter as _RuntimeEventPrinter
from src.interaction.slash_autocomplete import SlashAutocompletePrompt as _SlashAutocompletePrompt
from src.interaction.slash_commands import SlashCommandDispatcher as _SlashCommandDispatcher
from src.interaction.slash_render import SlashCommandRenderer as _SlashCommandRenderer
from src.interaction.startup_render import StartupRenderer as _StartupRenderer


def create_interaction_gateway(
    *,
    slash_specs: tuple[SlashCommandSpec, ...],
    stream: TextIO | None = None,
    stdin: TextIO | None = None,
    startup_lines: tuple[str, ...] | None = None,
    startup_subtitle: str | None = None,
    exit_title: str = 'Agent powering down. Goodbye!',
) -> InteractionGateway:
    """工厂函数：构造全部内部组件并通过依赖注入装配 InteractionGateway。

    调用方负责传入已装配的 slash 命令规格；interaction 只消费规格，
    不再持有任何业务命令实现。

    Args:
        slash_specs (tuple[SlashCommandSpec, ...]): 外部装配完成的 slash 命令规格列表。
        stream (TextIO | None): 统一输出流；None 时默认 sys.stdout。
        stdin (TextIO | None): 统一输入流；None 时默认 sys.stdin。
        startup_lines (tuple[str, ...] | None): 自定义 ASCII-art 标题行。
        startup_subtitle (str | None): 自定义副标题文本。
        exit_title (str): 退出提示框标题文本。
    Returns:
        InteractionGateway: 完整初始化的交互网关实例。
    """
    _stream = stream or sys.stdout
    _stdin = stdin or sys.stdin

    dispatcher = _SlashCommandDispatcher(specs=slash_specs)

    startup_renderer = _StartupRenderer(lines=startup_lines, subtitle=startup_subtitle)
    exit_renderer = _ExitRenderer(title=exit_title)
    slash_renderer = _SlashCommandRenderer()
    event_printer = _RuntimeEventPrinter(stream=_stream)

    autocomplete_entries = _build_autocomplete_entries(dispatcher)
    autocomplete_prompt = _SlashAutocompletePrompt(
        entries=autocomplete_entries,
        stdin=_stdin,
        stdout=_stream,
    )

    return InteractionGateway(
        dispatcher=dispatcher,
        startup_renderer=startup_renderer,
        exit_renderer=exit_renderer,
        slash_renderer=slash_renderer,
        event_printer=event_printer,
        autocomplete_prompt=autocomplete_prompt,
        stream=_stream,
        stdin=_stdin,
    )
def _build_autocomplete_entries(
    dispatcher: _SlashCommandDispatcher,
) -> tuple[SlashAutocompleteEntry, ...]:
    """从分发器命令规格构建自动补全条目（展开所有别名）。

    Args:
        dispatcher (SlashCommandDispatcher): 已装配规格的 slash 分发器。
    Returns:
        tuple[SlashAutocompleteEntry, ...]: 每个命令名展开为独立补全条目。
    """
    return tuple(
        SlashAutocompleteEntry(name=name, description=spec.description)
        for spec in dispatcher.get_slash_command_specs()
        for name in spec.names
    )


__all__ = [
    'InteractionGateway',
    'create_interaction_gateway',
]
