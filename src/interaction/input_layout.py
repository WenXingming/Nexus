"""输入提示布局生成器。"""

from __future__ import annotations

import shutil


class InputPromptLayout:
    """生成输入阶段使用的双横线 prompt 布局。"""

    def __init__(self, *, min_width: int = 36) -> None:
        self._min_width = min_width

    def build_plain_prompt(self) -> str:
        rule = self.build_rule_text()
        return f'{rule}\n›  '

    def build_formatted_prompt_fragments(self) -> list[tuple[str, str]]:
        rule = self.build_rule_text()
        return [
            ('class:prompt.rule', rule),
            ('', '\n'),
            ('class:prompt.chevron', '›'),
            ('', '  '),
        ]

    def build_bottom_toolbar_fragments(self) -> list[tuple[str, str]]:
        return [('class:bottom-toolbar class:prompt.rule', self.build_rule_text())]

    def build_rule_text(self) -> str:
        return '─' * self._resolve_rule_width()

    def _resolve_rule_width(self) -> int:
        terminal_columns = shutil.get_terminal_size(fallback=(120, 24)).columns
        return max(terminal_columns - 1, self._min_width, 8)
