from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from src.config import LLMConfig


def create_llm(config: LLMConfig | None = None) -> BaseChatModel:
    if config is None:
        config = LLMConfig()

    kwargs: dict = {
        "model_provider": config.provider,
        "model": config.model_name,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url

    return init_chat_model(**kwargs)
