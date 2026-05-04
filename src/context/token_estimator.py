"""上下文 token 启发式估算器。

提供 TokenEstimator，使用统一的字符长度启发式规则估算消息与工具定义的
token 开销，供 BudgetProjector、Snipper、Compactor 共享同一口径。
不依赖真实 tokenizer，以最低开销完成 pre-model 预算预检。
"""

from __future__ import annotations

import json

from src.core_contracts.model_contracts import Message
from src.core_contracts.context_contracts import BudgetProjection


_CHARS_PER_TOKEN: int = 4
"""int: 默认按每 4 个字符近似 1 个 token。"""

_MSG_OVERHEAD: int = 4
"""int: 每条消息额外附带的固定协议开销（role 分隔符等）。"""

_CHAT_BASE: int = 3
"""int: 整个聊天请求的基础固定开销（请求结构固定字段）。"""


class TokenEstimator:
    """按统一启发式规则估算消息与工具定义的输入 token 数。

    所有调用方共享同一实例，确保整个上下文治理链路的统计口径严格一致。
    核心工作流：
    1. estimate_messages() 估算整段会话上下文的总 token 量；
    2. estimate_message()  供 snip/compact 比较单条消息改写前后的成本差异；
    3. estimate_tools()    估算工具 schema 带来的附加 token 开销。
    """

    def __init__(
        self,
        chars_per_token: int = _CHARS_PER_TOKEN,
        message_overhead_tokens: int = _MSG_OVERHEAD,
        chat_base_tokens: int = _CHAT_BASE,
    ) -> None:
        """初始化估算器，接受可覆盖的启发式常量。

        Args:
            chars_per_token (int): 每多少字符近似等于 1 个 token，默认 4。
            message_overhead_tokens (int): 每条消息的固定开销 token 数，默认 4。
            chat_base_tokens (int): 整个消息列表的基础固定 token 数，默认 3。
        Returns:
            None
        Raises:
            None
        """
        self._chars_per_token = chars_per_token
        # int: 启发式字符到 token 的比率，默认按每 4 字符 1 token 计算。
        self._message_overhead = message_overhead_tokens
        # int: 每条消息因协议头（role 分隔符等）产生的固定 token 开销。
        self._chat_base = chat_base_tokens
        # int: 整个聊天请求结构的基础固定 token 开销。

    # =========================================================================
    # 公有接口
    # =========================================================================

    def estimate_messages(self, messages: list[Message]) -> int:
        """估算消息列表的总输入 token 数。

        对列表中每条消息调用 estimate_message()，累加后加上聊天级基础开销。

        Args:
            messages (list[Message]): 待估算的消息列表。
        Returns:
            int: 估算的总 token 数，最小值为 chat_base_tokens。
        Raises:
            无。
        """
        return self._chat_base + sum(self.estimate_message(msg) for msg in messages)

    def estimate_message(self, message: Message) -> int:
        """估算单条消息的 token 数。

        支持 content 为字符串或 None，并额外统计 tool_calls 的序列化开销。

        Args:
            message (Message): 单条消息对象。
        Returns:
            int: 该消息的估算 token 数（至少为 message_overhead_tokens）。
        Raises:
            无。
        """
        role_tokens = self._count_chars(message.role)
        content_tokens = self._count_content(message.content)
        tool_calls_tokens = self._count_tool_calls(message.tool_calls)
        return role_tokens + content_tokens + tool_calls_tokens + self._message_overhead

    def estimate_tools(self, tools: list[dict]) -> int:
        """估算工具 schema 列表的 token 数。

        将工具列表序列化为 JSON 后按字符长度启发式估算。

        Args:
            tools (list[dict]): 工具 schema 列表（OpenAI function-calling 格式）。
        Returns:
            int: 估算的 token 数；工具列表为空则返回 0。
        Raises:
            无。
        """
        if not tools:
            return 0
        return self._count_chars(json.dumps(tools, ensure_ascii=False))

    def project_budget(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        max_input_tokens: int | None = None,
        output_reserve_tokens: int = 4_096,
        soft_buffer_tokens: int = 13_000,
    ) -> BudgetProjection:
        """预检本次模型调用的 token 预算并返回快照。

        Args:
            messages (list[Message]): 当前会话消息列表。
            tools (list[dict] | None): 当前工具 schema 列表；None 等同于空列表。
            max_input_tokens (int | None): 输入 token 硬上限；None 表示不设限。
            output_reserve_tokens (int): 输出预留 token 数，默认 4096。
            soft_buffer_tokens (int): 软缓冲 token 数，默认 13000。
        Returns:
            BudgetProjection: 本次调用的预算快照，含 projected/hard/soft 及 over 标记。
        Raises:
            无。
        """
        projected = self.estimate_messages(messages) + self.estimate_tools(tools or [])

        if max_input_tokens is None:
            return BudgetProjection(
                projected_input_tokens=projected,
                output_reserve_tokens=output_reserve_tokens,
                hard_input_limit=None,
                soft_input_limit=None,
                is_hard_over=False,
                is_soft_over=False,
            )

        usable = max_input_tokens - output_reserve_tokens
        soft_limit = max(0, usable - soft_buffer_tokens)
        return BudgetProjection(
            projected_input_tokens=projected,
            output_reserve_tokens=output_reserve_tokens,
            hard_input_limit=max_input_tokens,
            soft_input_limit=soft_limit,
            is_hard_over=projected > usable,
            is_soft_over=projected > soft_limit,
        )

    # =========================================================================
    # 私有辅助（原子步骤）
    # =========================================================================

    def _count_content(self, content: str | None) -> int:
        """估算 content 字段的 token 数。

        Args:
            content (str | None): 消息 content 字段。
        Returns:
            int: 估算的 token 数；None 或空字符串时返回 0。
        Raises:
            无。
        """
        if not content:
            return 0
        return self._count_chars(content)

    def _count_tool_calls(self, tool_calls: list[dict] | None) -> int:
        """估算 tool_calls 字段的额外 token 数。

        Args:
            tool_calls (list[dict] | None): 消息的工具调用列表。
        Returns:
            int: 估算的 token 数；None 或空列表时返回 0。
        Raises:
            无。
        """
        if not tool_calls:
            return 0
        return self._count_chars(json.dumps(tool_calls, ensure_ascii=False))

    def _count_chars(self, text: str) -> int:
        """按字符长度启发式估算文本 token 数（向上取整，最小值为 1）。

        Args:
            text (str): 待估算的文本。
        Returns:
            int: 估算的 token 数，最小值为 1。
        Raises:
            无。
        """
        return max(1, (len(text) + self._chars_per_token - 1) // self._chars_per_token)
