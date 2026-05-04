"""
Workspace 模块跨边界契约定义。

本文件定义工作空间相关的配置契约，供工具、会话、Agent 等模块共享。
外部调用方只能依赖这些纯数据结构，不得感知任何内部实现细节。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceConfig:
    """工作空间配置契约。

    记录当前工作空间的根目录和相关配置，供各模块共享使用。
    """

    root: Path
    """Path: 工作空间根目录的绝对路径。"""

    @classmethod
    def from_cwd(cls) -> WorkspaceConfig:
        """从当前工作目录创建配置。

        Args:
            None
        Returns:
            WorkspaceConfig: 基于当前工作目录的配置实例。
        Raises:
            None
        """
        return cls(root=Path.cwd().resolve())

    @classmethod
    def from_path(cls, path: str | Path) -> WorkspaceConfig:
        """从指定路径创建配置。

        Args:
            path (str | Path): 工作空间根目录路径。
        Returns:
            WorkspaceConfig: 基于指定路径的配置实例。
        Raises:
            ValueError: 路径不存在或不是目录时抛出。
        """
        root = Path(path).resolve()
        if not root.exists():
            raise ValueError(f"工作空间路径不存在: {root}")
        if not root.is_dir():
            raise ValueError(f"工作空间路径不是目录: {root}")
        return cls(root=root)

    def resolve_path(self, path: str | Path) -> Path:
        """解析相对于工作空间的路径。

        如果 path 是绝对路径且在工作空间内，直接返回；
        否则视为相对于工作空间根目录的路径。

        Args:
            path (str | Path): 待解析的路径。
        Returns:
            Path: 解析后的绝对路径。
        Raises:
            ValueError: 解析后的路径逃逸出工作空间时抛出。
        """
        target = Path(path)
        if target.is_absolute():
            resolved = target.resolve()
        else:
            resolved = (self.root / target).resolve()

        # 安全检查：防止路径逃逸
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise ValueError(
                f"路径逃逸出工作空间: {resolved} 不在 {self.root} 内"
            )
        return resolved

    def relative_to_root(self, path: Path) -> str:
        """获取路径相对于工作空间根目录的显示文本。

        Args:
            path (Path): 绝对路径。
        Returns:
            str: 相对路径文本；如果不在工作空间内则返回绝对路径。
        Raises:
            None
        """
        try:
            relative = path.resolve().relative_to(self.root)
            return str(relative) if str(relative) else "."
        except ValueError:
            return str(path)
