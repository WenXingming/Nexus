"""
LLM 领域公共数据模型。

本文件包含与具体客户端实现无关的、可跨模块复用的
基础数据传输对象（DTO）：对话消息、Token 用量。
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Message:
    """单条对话消息，支持纯文本与工具调用两种形态。

    Attributes:
        role: 消息角色，包含 system / user / assistant / tool 四种。
        content: 消息文本内容。assistant 角色携带 tool_calls 时可为 None，
            其余角色必须为非空字符串。
        tool_calls: 工具调用列表，仅 assistant 角色使用。每项为包含 id / type / function
            的字典，与 OpenAI tool_calls 格式一致。
        tool_call_id: 工具调用 ID，仅 tool 角色使用，对应 assistant 消息中 tool_calls 的 id。
        name: 工具名称，仅 tool 角色使用。
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class TokenUsage:
    """单次 LLM 调用的 Token 用量统计。

    Attributes:
        prompt_tokens: 提示词消耗的 Token 数。
        completion_tokens: 模型生成消耗的 Token 数。
        total_tokens: 总计消耗的 Token 数。
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
