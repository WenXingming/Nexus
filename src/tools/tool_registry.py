"""tools 模块内的工具注册表。"""

from __future__ import annotations

from src.core_contracts.tools_contracts import ToolDescriptor


class ToolRegistry(dict[str, ToolDescriptor]):
    """基于 dict 的工具注册表，支持重复名称检测。"""

    @classmethod
    def from_tools(cls, *tools: ToolDescriptor) -> "ToolRegistry":
        """通过工具列表创建注册表，检测重复名称。"""
        registry = cls()
        for tool in tools:
            if tool.name in registry:
                raise ValueError(f"重复工具名: {tool.name}")
            registry[tool.name] = tool
        return registry
