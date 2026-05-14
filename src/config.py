from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))
    api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    base_url: str | None = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or None)
    model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL_NAME", "claude-sonnet-4-6"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "64000")))

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model_name}"
