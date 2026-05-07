"""交互式会话展示协调器。"""

from __future__ import annotations

from typing import TextIO

from src.core_contracts.interaction_contracts import TaskProgressEvent
from src.interaction.diff_render import DiffRenderer
from src.interaction.task_render import TaskProgressRenderer


class ConversationRenderer:
    """协调过程块和 diff 块的渲染。"""

    def __init__(
        self,
        *,
        task_renderer: TaskProgressRenderer,
        diff_renderer: DiffRenderer,
    ) -> None:
        self._task_renderer = task_renderer
        self._diff_renderer = diff_renderer

    def render_task_event(self, event: TaskProgressEvent, stream: TextIO | None = None) -> None:
        self._task_renderer.render(event, stream=stream)
        if event.diff is not None and event.diff.diff.strip():
            self._diff_renderer.render(event.diff, stream=stream)

    def flush(self) -> None:
        """当前渲染器不维护流式缓存，保留空实现。"""
        return None
