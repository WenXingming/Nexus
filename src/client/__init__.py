from src.client.client_gateway import ClientGateway
from src.client.openai_client import OpenAIClient
from src.core_contracts.model_config import ModelConfig


def create_gateway(config: ModelConfig) -> ClientGateway:
    """根据 ModelConfig 创建 ClientGateway 实例的工厂函数。

    Args:
        config: LLM 客户端配置契约。

    Returns:
        ClientGateway: 注入了 OpenAIClient 的门面网关实例。

    Raises:
        LlmInvalidRequestError: 当 config.api_key 为空时抛出。
    """
    openai_client = OpenAIClient(config)
    client_gateway = ClientGateway(openai_client)
    return client_gateway


__all__ = ["ClientGateway", "create_gateway"]
