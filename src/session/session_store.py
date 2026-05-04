"""
Session 快照持久化组件。

SessionStore 负责 session_id 校验、文件路径计算、JSON 编解码与磁盘 I/O。
该类是 session 模块内部唯一接触文件系统的组件。
"""

from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path

from src.core_contracts.session_contracts import (
    SessionSnapshot,
)

DEFAULT_SESSION_DIR = (Path(".nexus") / "sessions").resolve()


class SessionStore:
    """会话快照存储仓库。"""

    def __init__(self, directory: Path | None = None) -> None:
        """初始化存储仓库。

        Args:
            directory (Path | None): 快照目录；None 时使用默认目录。
        Returns:
            None
        Raises:
            None
        """
        self.directory: Path = (directory or DEFAULT_SESSION_DIR).resolve()
        # 会话快照文件根目录，统一使用绝对路径规避相对路径漂移。

    def save(self, snapshot: SessionSnapshot) -> Path:
        """将快照序列化并写入磁盘。

        Args:
            snapshot (SessionSnapshot): 待持久化的会话快照。
        Returns:
            Path: 落盘后的文件路径。
        Raises:
            ValueError: session_id 非法时抛出。
            RuntimeError: 序列化或写文件失败时抛出。
        """
        session_id = self._validate_id(snapshot.session_id)
        payload = self._encode(snapshot)
        return self._write_file(session_id, payload)

    def load(self, session_id: str) -> SessionSnapshot:
        """从磁盘读取并恢复快照对象。

        Args:
            session_id (str): 会话唯一标识。
        Returns:
            SessionSnapshot: 反序列化后的会话快照。
        Raises:
            ValueError: session_id 非法时抛出。
            FileNotFoundError: 文件不存在时抛出。
            RuntimeError: 文件损坏或载荷异常时抛出。
        """
        normalized_id = self._validate_id(session_id)
        payload = self._read_file(normalized_id)
        snapshot = self._decode(payload)
        if snapshot.session_id != normalized_id:
            raise RuntimeError(
                f"会话 ID 不一致: 文件内为 {snapshot.session_id}，请求为 {normalized_id}"
            )
        return snapshot

    # =========================================================================
    # 私有辅助函数（深度优先展开）
    # =========================================================================

    def _validate_id(self, session_id: str) -> str:
        """校验 session_id 的合法性。

        Args:
            session_id (str): 原始会话 ID。
        Returns:
            str: 规范化后的会话 ID。
        Raises:
            ValueError: ID 非法时抛出。
        """
        if not isinstance(session_id, str):
            raise ValueError("session_id 必须为字符串。")
        normalized = session_id.strip()
        if not normalized:
            raise ValueError("session_id 不能为空白字符串。")
        if normalized in {".", ".."}:
            raise ValueError(f"非法 session_id: {session_id!r}")
        if any(separator in normalized for separator in ("/", "\\")):
            raise ValueError(f"session_id 不得包含路径分隔符: {session_id!r}")
        if Path(normalized).name != normalized:
            raise ValueError(f"非法 session_id: {session_id!r}")
        return normalized

    def _encode(self, snapshot: SessionSnapshot) -> str:
        """将快照编码为 JSON 字符串。

        Args:
            snapshot (SessionSnapshot): 待编码快照对象。
        Returns:
            str: JSON 字符串。
        Raises:
            RuntimeError: 编码失败时抛出。
        """
        try:
            return json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False)
        except Exception as exc:
            raise RuntimeError(f"快照序列化失败: {exc}") from exc

    def _write_file(self, session_id: str, payload: str) -> Path:
        """将 JSON 文本写入目标文件。

        Args:
            session_id (str): 已校验的会话 ID。
            payload (str): 待写入的 JSON 文本。
        Returns:
            Path: 写入成功后的文件路径。
        Raises:
            RuntimeError: 目录创建或写入失败时抛出。
        """
        file_path = self._build_file_path(session_id)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(payload, encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"写入会话快照失败: {file_path}") from exc
        return file_path

    def _build_file_path(self, session_id: str) -> Path:
        """计算会话文件路径。

        Args:
            session_id (str): 已校验的会话 ID。
        Returns:
            Path: 会话 JSON 文件绝对路径。
        Raises:
            None
        """
        return self.directory / f"{session_id}.json"

    def _read_file(self, session_id: str) -> str:
        """读取会话文件内容。

        Args:
            session_id (str): 已校验的会话 ID。
        Returns:
            str: 文件中的 JSON 文本。
        Raises:
            FileNotFoundError: 文件不存在时抛出。
            RuntimeError: 读取失败时抛出。
        """
        file_path = self._build_file_path(session_id)
        try:
            return file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise RuntimeError(f"读取会话快照失败: {file_path}") from exc

    def _decode(self, payload: str) -> SessionSnapshot:
        """将 JSON 文本解码为快照对象。

        Args:
            payload (str): 原始 JSON 文本。
        Returns:
            SessionSnapshot: 恢复后的快照对象。
        Raises:
            RuntimeError: JSON 或字段内容损坏时抛出。
        """
        raw = self._parse_json(payload)
        try:
            return SessionSnapshot.from_dict(raw)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"会话快照载荷不合法: {exc}") from exc

    def _parse_json(self, payload: str) -> dict[str, object]:
        """解析 JSON 文本并校验顶层结构。

        Args:
            payload (str): 原始 JSON 文本。
        Returns:
            dict[str, object]: 解析后的顶层对象。
        Raises:
            RuntimeError: JSON 文本损坏时抛出。
        """
        try:
            parsed = json.loads(payload)
        except JSONDecodeError as exc:
            raise RuntimeError("会话快照 JSON 内容损坏。") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("会话快照顶层结构必须为 JSON 对象。")
        return parsed
