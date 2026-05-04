"""pytest 测试基座。

确保项目根目录进入 Python 模块搜索路径，使测试可稳定导入 src 包。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))