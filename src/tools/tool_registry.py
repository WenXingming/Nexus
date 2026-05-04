"""tools 模块内的工具注册表。"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field

from src.core_contracts.tools_contracts import ToolDescriptor


@dataclass
class ToolRegistry(MutableMapping[str, ToolDescriptor]):
    """像字典一样管理工具描述符。"""

    _items: dict[str, ToolDescriptor] = field(default_factory=dict)

    @classmethod
    def from_tools(cls, *tools: ToolDescriptor) -> "ToolRegistry":
        """通过工具列表创建注册表。"""
        items: dict[str, ToolDescriptor] = {}
        for tool in tools:
            if tool.name in items:
                raise ValueError(f"重复工具名: {tool.name}")
            items[tool.name] = tool
        return cls(items)

    def __getitem__(self, key: str) -> ToolDescriptor:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __setitem__(self, key: str, value: ToolDescriptor) -> None:
        self._items[key] = value

    def __delitem__(self, key: str) -> None:
        del self._items[key]
