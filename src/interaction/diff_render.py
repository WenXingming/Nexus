"""统一 diff 渲染器。"""

from __future__ import annotations

import sys
from typing import TextIO

from src.core_contracts.interaction_contracts import CodeDiffArtifact
from src.interaction.terminal_render import TerminalRenderer


class DiffRenderer(TerminalRenderer):
    """把 unified diff 产物渲染成高亮差异块。"""

    _TITLE_RGB = (0x8A, 0xF0, 0xFF)
    _ADDITION_RGB = (0x73, 0xE2, 0x9A)
    _DELETION_RGB = (0xF7, 0x76, 0x8E)
    _HUNK_RGB = (0x79, 0xD9, 0xEA)
    _META_RGB = (0xB9, 0xC5, 0xD1)

    def __init__(self, *, top_padding: int = 1, bottom_padding: int = 0) -> None:
        super().__init__(
            frame_horizontal_padding=1,
            frame_vertical_padding=0,
            top_padding=top_padding,
            bottom_padding=bottom_padding,
        )

    def render(self, artifact: CodeDiffArtifact, stream: TextIO | None = None) -> None:
        target = stream or sys.stdout
        title = f'Diff · {artifact.path} ({artifact.operation})'
        diff_lines = tuple(artifact.diff.splitlines()) or ('(no diff)',)
        self._render_block((title, '', *diff_lines), stream=target, active_title=title)

    def _render_content_text(
        self,
        text: str,
        content_width: int,
        use_ansi: bool,
        *,
        active_title: str = '',
    ) -> str:
        padded = self._pad_to_display_width(text, content_width)
        if not use_ansi:
            return padded
        if active_title and text == active_title:
            return self._colorize_line(padded, self._TITLE_RGB)
        if text.startswith('+++') or text.startswith('---'):
            return self._colorize_line(padded, self._META_RGB)
        if text.startswith('@@'):
            return self._colorize_line(padded, self._HUNK_RGB)
        if text.startswith('+'):
            return self._colorize_line(padded, self._ADDITION_RGB)
        if text.startswith('-'):
            return self._colorize_line(padded, self._DELETION_RGB)
        return padded

    @staticmethod
    def _colorize_line(text: str, rgb: tuple[int, int, int]) -> str:
        red, green, blue = rgb
        return f'\x1b[38;2;{red};{green};{blue}m{text}\x1b[0m'
