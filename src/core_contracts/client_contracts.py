"""
LLM 客户端操作契约定义。

本文件包含 ClientGateway 与外部调用者之间交互所需的
请求/响应 DTO、流式块 DTO。
所有类型均为纯数据类，不包含任何业务逻辑或 I/O 依赖。

领域公共模型 Message / TokenUsage 定义于 src.core_contracts.model_contracts；
ModelConfig 定义于 src.core_contracts.model_config，本文件按需引用。
"""

from dataclasses import dataclass

from src.core_contracts.model_contracts import Message, TokenUsage  # noqa: F401  – re-exported for convenience


# =============================================================================
# 请求 DTO
# =============================================================================


@dataclass(frozen=True)
class LlmRequest:
    """LLM 聊天补全请求的标准化契约。

    model / temperature / max_tokens 为 None 时将回退至 ModelConfig 的默认值。

    Attributes:
        messages: 对话消息列表，至少包含一条消息。
        tools: 可用的工具定义列表，格式与 OpenAI function-calling tools 一致。
            为 None 或不传时不启用工具调用。
        model: 模型名称，为 None 时使用 ModelConfig.model_name。
        temperature: 采样温度，范围 [0, 2]，为 None 时使用 ModelConfig.temperature。
        max_tokens: 最大生成 Token 数，为 None 时使用 ModelConfig.max_tokens。
        stream: 是否启用流式输出。
        stop: 可选的停止词列表。
        top_p: 核采样参数，范围 [0, 1]。
        frequency_penalty: 频率惩罚，范围 [-2, 2]。
        presence_penalty: 存在惩罚，范围 [-2, 2]。
    """

    messages: list[Message]
    tools: list[dict] | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None


# =============================================================================
# 结果 DTO
# =============================================================================


@dataclass(frozen=True)
class LlmResult:
    """LLM 聊天补全的标准化结果。

    Attributes:
        content: 模型生成的文本内容。当 finish_reason 为 "tool_calls" 时可能为空字符串。
        model: 实际使用的模型名称。
        finish_reason: 完成原因（如 "stop"、"length"、"tool_calls"）。
        usage: Token 用量统计。
        tool_calls: 模型请求的工具调用列表，无工具调用时为 None。
            每项包含 id / type / function（name + arguments）字段。
    """

    content: str
    model: str
    finish_reason: str
    usage: TokenUsage
    tool_calls: list[dict] | None = None


@dataclass(frozen=True)
class LlmStreamChunk:
    """LLM 流式输出的单个增量块。

    Attributes:
        content: 本次增量的文本内容。
        finish_reason: 完成原因，仅最后一个块中有值。
        index: 选择序号（通常为 0）。
    """

    content: str
    finish_reason: str | None
    index: int
