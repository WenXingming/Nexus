"""Slash 命令内部分发器实现（interaction 包内部实现）。

本模块属于 interaction 包的内部实现，外部代码禁止直接导入。
所有功能均通过 interaction.interaction_gateway.InteractionGateway 对外暴露。

职责边界：
1. 解析用户原始输入中的本地 slash 命令（parse_slash_command）。
2. 按精确名或唯一前缀解析命令规格，含歧义检测（resolve_slash_command）。
3. 调用外部注入的命令处理器并返回结构化结果（dispatch_slash_command）。
"""

from __future__ import annotations

from dataclasses import replace

from src.core_contracts.interaction_contracts import (
    ParsedSlashCommand,
    SlashCommandContext,
    SlashCommandResolution,
    SlashCommandResult,
    SlashCommandSpec,
)


class SlashCommandDispatcher:
    """本地 slash 命令解析、索引与分发的内部实现。"""

    def __init__(
        self,
        *,
        specs: tuple[SlashCommandSpec, ...] = (),
    ) -> None:
        """初始化 slash 命令分发器。"""
        self._specs: tuple[SlashCommandSpec, ...] = specs
        self._spec_index = self._build_spec_index(self._specs)

    def load_specs(self, specs: tuple[SlashCommandSpec, ...]) -> None:
        """从外部加载已装配的 slash 命令规格并重建索引。"""
        self._specs = specs
        self._spec_index = self._build_spec_index(specs)

    def dispatch_slash_command(
        self,
        context: SlashCommandContext,
        input_text: str,
    ) -> SlashCommandResult:
        """分发单条输入；非 slash 输入透传给常规 query 路径。

        Args:
            context (SlashCommandContext): 命令执行所需的只读上下文。
            input_text (str): 用户输入文本。
        Returns:
            SlashCommandResult: 分流结果，描述是否已处理以及后续是否继续 query。
        """
        parsed = self.parse_slash_command(input_text)
        if parsed is None:
            return SlashCommandResult(
                handled=False,
                continue_query=True,
                prompt=input_text,
            )

        resolution = self.resolve_slash_command(parsed.command_name)
        if resolution.kind == 'empty':
            return SlashCommandResult(
                handled=True,
                continue_query=False,
                command_name='help',
                output='\n'.join(self._build_help_lines()),
                metadata={'match_mode': 'list_all'},
            )

        if resolution.kind == 'ambiguous':
            return self._build_ambiguous_command_result(parsed.command_name, resolution.candidates)

        if resolution.kind == 'none' or resolution.spec is None:
            return self._build_unknown_command_result(parsed.command_name)

        effective_parsed = ParsedSlashCommand(
            command_name=resolution.matched_name,
            arguments=parsed.arguments,
            raw_input=parsed.raw_input,
        )
        result = resolution.spec.handler(context, effective_parsed)
        if resolution.kind != 'prefix':
            return result

        return replace(
            result,
            metadata={
                **result.metadata,
                'match_mode': 'prefix',
                'typed_name': parsed.command_name,
                'matched_name': resolution.matched_name,
            },
        )

    def parse_slash_command(self, input_text: str) -> ParsedSlashCommand | None:
        """从原始输入中提取 slash 命令。

        Args:
            input_text (str): 用户提交的原始输入文本。
        Returns:
            ParsedSlashCommand | None: 成功时返回解析结果；普通 prompt 返回 None。
        """
        stripped = input_text.strip()
        if not stripped.startswith('/'):
            return None

        body = stripped[1:]
        command_name, _, arguments = body.partition(' ')
        return ParsedSlashCommand(
            command_name=command_name.strip().lower(),
            arguments=arguments.strip(),
            raw_input=input_text,
        )

    def get_slash_command_specs(self) -> tuple[SlashCommandSpec, ...]:
        """返回当前分发器支持的 slash 命令规格列表。

        Returns:
            tuple[SlashCommandSpec, ...]: 按帮助展示顺序排列的命令规格。
        """
        return self._specs

    def get_command_completions(self, prefix: str) -> tuple[str, ...]:
        """返回给定前缀可匹配的全部命令名与别名。

        Args:
            prefix (str): 用户已输入的命令名前缀，为空时返回全部命令。
        Returns:
            tuple[str, ...]: 按命令定义顺序排列的匹配命令名列表。
        """
        normalized_prefix = prefix.strip().lower()
        completions: list[str] = []
        for spec in self._specs:
            for name in spec.names:
                if not normalized_prefix or name.startswith(normalized_prefix):
                    completions.append(name)
        return tuple(completions)

    def find_slash_command(self, command_name: str) -> SlashCommandSpec | None:
        """按名称查找 slash 命令规格。

        Args:
            command_name (str): 待查找的命令名，可包含大小写与首尾空白。
        Returns:
            SlashCommandSpec | None: 找到时返回规格对象，否则返回 None。
        """
        return self._spec_index.get(command_name.strip().lower())

    def resolve_slash_command(self, command_name: str) -> SlashCommandResolution:
        """按精确名或唯一前缀解析 slash 命令。

        Args:
            command_name (str): 用户输入的命令名，可包含大小写与首尾空白。
        Returns:
            SlashCommandResolution: 解析结果，含 kind / spec / candidates 等字段。
        """
        normalized_name = command_name.strip().lower()
        if not normalized_name:
            return SlashCommandResolution(
                kind='empty',
                candidates=self.get_command_completions(''),
            )

        spec = self.find_slash_command(normalized_name)
        if spec is not None:
            return SlashCommandResolution(kind='exact', spec=spec, matched_name=normalized_name)

        candidates = self.get_command_completions(normalized_name)
        if not candidates:
            return SlashCommandResolution(kind='none')
        if len(candidates) == 1:
            matched_name = candidates[0]
            return SlashCommandResolution(
                kind='prefix',
                spec=self._spec_index[matched_name],
                matched_name=matched_name,
                candidates=candidates,
            )
        return SlashCommandResolution(kind='ambiguous', candidates=candidates)

    def _build_unknown_command_result(self, command_name: str) -> SlashCommandResult:
        """为未知 slash 命令构造统一错误结果。

        Args:
            command_name (str): 用户输入的命令名，可能为空字符串。
        Returns:
            SlashCommandResult: 包含 unknown_command 错误码的本地处理结果。
        """
        command_label = command_name or '(empty)'
        return SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name=command_label,
            output=(
                f'Unknown slash command: /{command_label}\n'
                'Type / to list supported local commands.'
            ),
            metadata={'error': 'unknown_command'},
        )

    def _build_ambiguous_command_result(
        self,
        command_name: str,
        candidates: tuple[str, ...],
    ) -> SlashCommandResult:
        """为歧义前缀构造统一提示结果。

        Args:
            command_name (str): 用户输入的歧义命令名。
            candidates (tuple[str, ...]): 匹配该前缀的候选命令名列表。
        Returns:
            SlashCommandResult: 包含歧义命令提示的本地处理结果。
        """
        lines = [f'Ambiguous slash command: /{command_name}', 'Matches:']
        lines.extend(self._build_candidate_lines(candidates))
        lines.append('Type a longer prefix or input / to list all commands.')
        return SlashCommandResult(
            handled=True,
            continue_query=False,
            command_name=command_name,
            output='\n'.join(lines),
            metadata={
                'error': 'ambiguous_command',
                'candidates': list(candidates),
            },
        )

    def _build_spec_index(
        self,
        specs: tuple[SlashCommandSpec, ...],
    ) -> dict[str, SlashCommandSpec]:
        """根据命令规格构建名称索引。"""
        index: dict[str, SlashCommandSpec] = {}
        for spec in specs:
            for name in spec.names:
                index[name] = spec
        return index

    def _build_help_lines(self) -> list[str]:
        """构建 slash 命令总览文本。

        Returns:
            list[str]: 格式化的帮助文本行列表。
        """
        lines = ['Slash Commands', '==============', '']
        for spec in self.get_slash_command_specs():
            lines.append(f'/{spec.names[0]} - {spec.description}')
        lines.extend(['', 'Tip: input / to list commands, or use a unique prefix such as /st.'])
        return lines

    def _build_candidate_lines(self, candidates: tuple[str, ...]) -> list[str]:
        """把候选命令名格式化为带描述的文本行。

        Args:
            candidates (tuple[str, ...]): 候选命令名列表。
        Returns:
            list[str]: 每条命令格式化为 /name - description 的文本行列表。
        """
        lines: list[str] = []
        for candidate in candidates:
            spec = self._spec_index[candidate]
            lines.append(f'/{candidate} - {spec.description}')
        return lines


__all__ = ['SlashCommandDispatcher']
