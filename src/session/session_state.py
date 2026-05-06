"""
Session 运行态状态构建组件。

SessionStateRuntime 负责：
1. 校验首条 prompt 并创建全新会话状态
2. 校验持久化消息序列并恢复运行态状态
3. 向运行态状态追加消息
4. 将运行态状态转换为持久化快照

该组件不承担任何 I/O，也不感知存储格式。
"""

from __future__ import annotations

import uuid

from src.core_contracts.model_config import ModelConfig


def generate_session_id() -> str:
    """Generate a new unique session identifier."""
    return uuid.uuid4().hex[:32]
from src.core_contracts.model_contracts import Message, TokenUsage
from src.core_contracts.session_contracts import (
    JsonDict,
    SessionSnapshot,
    SessionState,
)


class SessionStateRuntime:
    """会话运行态状态构建器。"""

    def __init__(self) -> None:
        """初始化运行态构建器。

        Args:
            None
        Returns:
            None
        Raises:
            None
        """
        self._allowed_roles: set[str] = {"system", "user", "assistant", "tool"}
        # 允许恢复的消息角色集合，用于防御非法持久化数据。

    # =========================================================================
    # 公有接口
    # =========================================================================

    def build_new(self, prompt: str) -> SessionState:
        """根据首条用户输入创建全新会话状态。

        Args:
            prompt (str): 用户首条输入。
        Returns:
            SessionState: 已初始化的会话状态。
        Raises:
            ValueError: prompt 非法时抛出。
        """
        normalized_prompt = self._validate_prompt(prompt)
        session_id = generate_session_id()
        state = SessionState(session_id=session_id)
        self.append_user(state, normalized_prompt)
        return state

    def build_from_persisted(
        self,
        session_id: str,
        messages: tuple[Message, ...],
        transcript: tuple[Message, ...],
    ) -> SessionState:
        """从持久化消息与转录数据恢复会话状态。

        Args:
            session_id (str): 会话唯一标识。
            messages (tuple[Message, ...]): 持久化消息序列。
            transcript (tuple[Message, ...]): 持久化转录序列。
        Returns:
            SessionState: 恢复后的会话状态对象。
        Raises:
            ValueError: 输入序列格式非法时抛出。
        """
        message_list = self._normalize_messages(messages, "messages")
        transcript_list = self._normalize_messages(transcript, "transcript") if transcript else list(message_list)
        return SessionState(session_id=session_id, messages=message_list, transcript_entries=transcript_list)

    def append_user(self, state: SessionState, prompt: str) -> None:
        """向状态追加一条用户消息。

        Args:
            state (SessionState): 目标会话状态。
            prompt (str): 用户输入内容。
        Returns:
            None
        Raises:
            None
        """
        self.append_message(state, Message(role="user", content=prompt))

    def append_assistant(self, state: SessionState, content: str) -> None:
        """向状态追加一条助手消息。

        Args:
            state (SessionState): 目标会话状态。
            content (str): 助手输出内容。
        Returns:
            None
        Raises:
            None
        """
        self.append_message(state, Message(role="assistant", content=content))

    def append_message(self, state: SessionState, message: Message) -> None:
        """向状态追加一条任意角色的消息。

        Args:
            state (SessionState): 目标会话状态。
            message (Message): 待追加的消息对象，role 须在 _allowed_roles 内。
        Returns:
            None
        Raises:
            ValueError: message 非法时抛出（由 _validate_message 判定）。
        """
        self._validate_message(message, "message", 0)
        state.messages.append(message)
        state.transcript_entries.append(message)

    def to_snapshot(
        self,
        state: SessionState,
        session_id: str,
        model_config: ModelConfig,
        usage: TokenUsage | None = None,
        last_response: str = "",
        metadata: JsonDict | None = None,
    ) -> SessionSnapshot:
        """将运行态状态封装为持久化快照。

        Args:
            state (SessionState): 当前运行态状态。
            session_id (str): 会话唯一标识。
            model_config (ModelConfig): 当前模型配置。
            usage (TokenUsage | None): Token 统计；None 时使用零值。
            last_response (str): 最后一轮助手输出。
            metadata (JsonDict | None): 扩展元数据。
        Returns:
            SessionSnapshot: 可落盘的快照对象。
        Raises:
            None
        """
        effective_usage = usage if usage is not None else TokenUsage(0, 0, 0)
        effective_metadata = metadata if metadata is not None else {}
        return SessionSnapshot(
            session_id=session_id,
            model_config=model_config,
            messages=tuple(state.messages),
            transcript=tuple(state.transcript_entries),
            last_response=last_response,
            turns=self.get_turn_count(state),
            usage=effective_usage,
            metadata=effective_metadata,
        )

    def get_turn_count(self, state: SessionState) -> int:
        """返回当前用户轮次数。

        Args:
            state (SessionState): 目标会话状态。
        Returns:
            int: user 角色消息数量。
        Raises:
            None
        """
        return sum(1 for message in state.messages if message.role == "user")

    # =========================================================================
    # 私有辅助函数（深度优先展开）
    # =========================================================================

    def _validate_prompt(self, prompt: str) -> str:
        """校验 prompt 的类型与内容。

        Args:
            prompt (str): 原始提示词。
        Returns:
            str: 去除首尾空白后的提示词。
        Raises:
            ValueError: prompt 不是非空字符串时抛出。
        """
        if not isinstance(prompt, str):
            raise ValueError(f"prompt 必须为字符串，当前类型: {type(prompt).__name__}")
        normalized = prompt.strip()
        if not normalized:
            raise ValueError("prompt 不能为空白字符串。")
        return normalized

    def _normalize_messages(self, messages: tuple[Message, ...], field_name: str) -> list[Message]:
        """校验并复制 Message 序列。

        Args:
            messages (tuple[Message, ...]): 原始消息序列。
            field_name (str): 字段名。
        Returns:
            list[Message]: 规范化后的消息列表。
        Raises:
            ValueError: 序列中存在非法消息时抛出。
        """
        if not isinstance(messages, tuple):
            raise ValueError(f"{field_name} 必须为 Message 元组。")
        normalized: list[Message] = []
        for index, message in enumerate(messages):
            self._validate_message(message, field_name, index)
            normalized.append(
                Message(
                    role=message.role,
                    content=message.content,
                    tool_calls=message.tool_calls,
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                )
            )
        return normalized

    def _validate_message(self, message: Message, field_name: str, index: int) -> None:
        """校验单条消息对象。

        tool 角色要求 content 为非空字符串且 tool_call_id 不为 None；
        assistant 角色携带 tool_calls 时 content 可为 None；
        其余角色要求 content 为非空字符串。

        Args:
            message (Message): 待校验的消息对象。
            field_name (str): 上层字段名。
            index (int): 当前消息索引。
        Returns:
            None
        Raises:
            ValueError: 消息对象非法时抛出。
        """
        if not isinstance(message, Message):
            raise ValueError(f"{field_name}[{index}] 必须为 Message 对象。")
        if message.role not in self._allowed_roles:
            raise ValueError(f"{field_name}[{index}].role 非法: {message.role}")
        if message.role == "tool":
            if not isinstance(message.tool_call_id, str) or not message.tool_call_id.strip():
                raise ValueError(
                    f"{field_name}[{index}] 为 tool 角色时必须提供非空 tool_call_id。"
                )
            if not isinstance(message.content, str) or not message.content.strip():
                raise ValueError(
                    f"{field_name}[{index}] 为 tool 角色时 content 必须为非空字符串。"
                )
        elif message.tool_calls is not None:
            return
        elif not isinstance(message.content, str) or not message.content.strip():
            raise ValueError(
                f"{field_name}[{index}].content 必须为非空字符串。"
            )
