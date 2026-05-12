from pathlib import Path

from src.session.session_gateway import SessionGateway
from src.session.session_state import SessionStateRuntime as _SessionStateRuntime, generate_session_id
from src.session.session_store import SessionStore as _SessionStore


def create_gateway(session_store_directory: Path | None = None) -> SessionGateway:
    """构造默认注入依赖并返回 SessionGateway。

    Args:
        session_store_directory (Path | None): 快照存储目录；None 时使用默认目录。
    Returns:
        SessionGateway: 已装配完成的 session 门面实例。
    Raises:
        None
    """
    session_store = _SessionStore(directory=session_store_directory)
    session_state = _SessionStateRuntime()
    return SessionGateway(session_store=session_store, session_state=session_state)


__all__ = ["SessionGateway", "create_gateway", "generate_session_id"]