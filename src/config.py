from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()


def _safe_env_float(key: str, default: str) -> float:
    try:
        return float(os.getenv(key, default))
    except (ValueError, TypeError):
        return float(default)


def _safe_env_int(key: str, default: str) -> int:
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return int(default)


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))
    api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    base_url: str | None = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or None)
    model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL_NAME", "claude-sonnet-4-6"))
    temperature: float = field(default_factory=lambda: _safe_env_float("LLM_TEMPERATURE", "0.7"))
    max_tokens: int = field(default_factory=lambda: _safe_env_int("LLM_MAX_TOKENS", "64000"))

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model_name}"

    def __repr__(self) -> str:
        masked = (self.api_key[:4] + "..." + self.api_key[-4:]) if self.api_key and len(self.api_key) > 8 else "***"
        return (
            f"LLMConfig(provider={self.provider!r}, model_name={self.model_name!r}, "
            f"api_key={masked!r}, temperature={self.temperature}, max_tokens={self.max_tokens})"
        )
