"""轻量任务过程流渲染器。"""

from __future__ import annotations

import sys
from typing import TextIO

from src.core_contracts.interaction_contracts import TaskProgressEvent
from src.interaction.terminal_render import TerminalRenderer


class TaskProgressRenderer(TerminalRenderer):
    """把结构化任务事件渲染为轻量流式过程行。"""

    _STATUS_ICON = {
        'running': '…',
        'info': '•',
        'success': '✓',
        'warning': '!',
        'error': '×',
    }

    _STATUS_RGB = {
        'running': (0x79, 0xD9, 0xEA),
        'info': (0xB9, 0xC5, 0xD1),
        'success': (0x73, 0xE2, 0x9A),
        'warning': (0xF6, 0xC4, 0x53),
        'error': (0xF7, 0x76, 0x8E),
    }

    def __init__(self, *, top_padding: int = 0, bottom_padding: int = 0) -> None:
        super().__init__(
            frame_horizontal_padding=0,
            frame_vertical_padding=0,
            top_padding=top_padding,
            bottom_padding=bottom_padding,
        )
        self._active_status = 'info'

    def render(self, event: TaskProgressEvent, stream: TextIO | None = None) -> None:
        target = stream or sys.stdout
        use_ansi = self._stream_supports_ansi(target)
        self._active_status = event.status
        self._write_blank_lines(target, self._top_padding)
        for line in self._build_lines(event, use_ansi=use_ansi):
            self._write_line(target, line)
        self._write_blank_lines(target, self._bottom_padding)

    def _build_lines(self, event: TaskProgressEvent, *, use_ansi: bool) -> tuple[str, ...]:
        turn_prefix = f'Turn {event.turn} · ' if event.turn is not None else ''
        icon = self._STATUS_ICON[event.status]
        header = f'{icon} {turn_prefix}{event.title}'.rstrip()
        rendered_header = self._colorize_header(header) if use_ansi else header
        detail_lines = self._build_detail_lines(event.detail)
        if not detail_lines:
            return (rendered_header,)
        return (rendered_header, *detail_lines)

    def _build_detail_lines(self, detail: str) -> tuple[str, ...]:
        if not detail:
            return ()
        cleaned = [line.rstrip() for line in detail.splitlines() if line.strip()]
        if not cleaned:
            return ()
        limited = cleaned[:3]
        if len(cleaned) > 3:
            limited.append('...')
        return tuple(f'  {line}' for line in limited)

    def _colorize_header(self, text: str) -> str:
        red, green, blue = self._STATUS_RGB[self._active_status]
        return f'\x1b[38;2;{red};{green};{blue}m{text}\x1b[0m'
