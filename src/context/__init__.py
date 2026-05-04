"""context 包公开入口。

架构约定：
- 对外仅暴露 ContextGateway 与 create_gateway 工厂函数。
- 所有外部消费者必须通过本入口访问 context 治理能力；
  禁止跨包直接依赖内部子模块。
- 内部实现类（TokenEstimator、Snipper、Compactor 等）仅允许通过
  子模块路径访问（用于单元测试白盒场景）。
"""

from __future__ import annotations

from src.context.context_gateway import ContextGateway
from src.core_contracts.context_contracts import ContextModelClient


def create_gateway(client: ContextModelClient | None = None) -> ContextGateway:
    """工厂函数：构造全部内部组件并通过依赖注入装配 ContextGateway。

    调用方只需传入可选的模型客户端；TokenEstimator、Snipper、Compactor
    的实例化由本工厂统一负责，外部无需感知任何内部构件。

    Args:
        client (ContextModelClient | None): 可选模型客户端。为 None 时网关仅支持
            预算投影与 snip；compact 与 reactive compact 路径将在调用时抛出 RuntimeError。
    Returns:
        ContextGateway: 完整初始化的上下文治理网关实例。
    Raises:
        无。
    """
    from src.context.token_estimator import TokenEstimator
    from src.context.snipper import Snipper
    from src.context.compactor import Compactor

    estimator = TokenEstimator()
    snipper = Snipper(token_estimator=estimator)
    compactor = Compactor(client=client, token_estimator=estimator) if client is not None else None

    return ContextGateway(
        token_estimator=estimator,
        snipper=snipper,
        compactor=compactor,
    )


__all__ = ["ContextGateway", "create_gateway"]
