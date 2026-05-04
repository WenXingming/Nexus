"""
Session 模块 Facade。

SessionGateway 是 session 域对外暴露的唯一入口。
它只负责接收参数、分发到内部组件。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.core_contracts.model_config import ModelConfig
from src.core_contracts.model_contracts import Message, TokenUsage
from src.core_contracts.session_contracts import (
    JsonDict,
    SessionSnapshot,
    SessionState,
)

if TYPE_CHECKING:
    from src.session.session_state import SessionStateRuntime
    from src.session.session_store import SessionStore


class SessionGateway:
    """Session 域门面网关。"""

    def __init__(self, session_store: "SessionStore", session_state: "SessionStateRuntime") -> None:
        """初始化 SessionGateway。

        Args:
            session_store (SessionStore): 持久化组件。
            session_state (SessionStateRuntime): 运行态状态组件。
        Returns:
            None
        Raises:
            None
        """
        self._session_store = session_store
        self._session_state = session_state

    @property
    def directory(self) -> Path:
        """返回当前会话快照根目录。

        Args:
            None
        Returns:
            Path: 会话快照目录。
        Raises:
            None
        """
        return self._session_store.directory

    def save(self, snapshot: SessionSnapshot) -> tuple[str, str]:
        """保存会话快照。

        Args:
            snapshot (SessionSnapshot): 待保存的会话快照。
        Returns:
            tuple[str, str]: (session_id, session_path)。
        Raises:
            ValueError: session_id 非法时抛出。
            RuntimeError: 序列化或写文件失败时抛出。
        """
        session_path = self._session_store.save(snapshot)
        return snapshot.session_id, str(session_path)

    def load(self, session_id: str) -> SessionSnapshot:
        """加载会话快照。

        Args:
            session_id (str): 会话唯一标识。
        Returns:
            SessionSnapshot: 加载的会话快照。
        Raises:
            ValueError: session_id 非法时抛出。
            FileNotFoundError: 文件不存在时抛出。
            RuntimeError: 文件损坏或载荷异常时抛出。
        """
        return self._session_store.load(session_id)

    def save_state(
        self,
        state: SessionState,
        model_config: ModelConfig,
        usage: TokenUsage | None = None,
        last_response: str = "",
        metadata: JsonDict | None = None,
    ) -> tuple[str, str]:
        """将运行态状态转换为快照并保存。

        Args:
            state (SessionState): 当前运行态状态。
            model_config (ModelConfig): 当前模型配置。
            usage (TokenUsage | None): Token 统计；None 时使用零值。
            last_response (str): 最后一轮助手输出。
            metadata (JsonDict | None): 扩展元数据。
        Returns:
            tuple[str, str]: (session_id, session_path)。
        Raises:
            ValueError: session_id 非法时抛出。
            RuntimeError: 转换或保存失败时抛出。
        """
        snapshot = self._session_state.to_snapshot(
            state=state,
            session_id=state.session_id,
            model_config=model_config,
            usage=usage,
            last_response=last_response,
            metadata=metadata,
        )
        session_path = self._session_store.save(snapshot)
        return state.session_id, str(session_path)
    
    def create_state(self, prompt: str) -> SessionState:
        """创建全新运行态会话状态。

        Args:
            prompt (str): 用户首条输入。
        Returns:
            SessionState: 新建的运行态状态。
        Raises:
            ValueError: prompt 非法时抛出。
        """
        return self._session_state.build_new(prompt)

    def resume_state(
        self,
        session_id: str,
        messages: tuple[Message, ...],
        transcript: tuple[Message, ...] = (),
    ) -> SessionState:
        """从持久化数据恢复运行态会话状态。

        Args:
            session_id (str): 会话唯一标识。
            messages (tuple[Message, ...]): 持久化消息序列。
            transcript (tuple[Message, ...]): 持久化转录序列。
        Returns:
            SessionState: 恢复后的运行态状态。
        Raises:
            ValueError: 输入序列格式非法时抛出。
        """
        return self._session_state.build_from_persisted(session_id, messages, transcript)

    def append_user(self, state: SessionState, prompt: str) -> None:
        """向运行态状态追加一条用户消息。

        Args:
            state (SessionState): 目标会话状态。
            prompt (str): 用户输入内容。
        Returns:
            None
        Raises:
            None
        """
        self._session_state.append_user(state, prompt)

    def append_assistant(self, state: SessionState, content: str) -> None:
        """向运行态状态追加一条助手消息。

        Args:
            state (SessionState): 目标会话状态。
            content (str): 助手输出内容。
        Returns:
            None
        Raises:
            None
        """
        self._session_state.append_assistant(state, content)

    def append_system(self, state: SessionState, content: str) -> None:
        """向运行态状态追加一条系统消息。

        Args:
            state (SessionState): 目标会话状态。
            content (str): 系统提示内容。
        Returns:
            None
        Raises:
            None
        """
        self._session_state.append_message(state, Message(role="system", content=content))

    def append_message(self, state: SessionState, message: Message) -> None:
        """向运行态状态追加一条任意角色的消息。

        Args:
            state (SessionState): 目标会话状态。
            message (Message): 待追加的消息对象。
        Returns:
            None
        Raises:
            ValueError: message 不合法时抛出。
        """
        self._session_state.append_message(state, message)
