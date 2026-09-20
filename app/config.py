from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Vendor Invoice Extraction"
    database_path: str = "data/vendor_invoices.db"
    upload_dir: str = "data/uploads"

    llm_provider: str = "unconfigured"
    llm_model: str = "configure-me"
    llm_api_key: str = ""
    llm_chat_completions_url: str = "https://api.openai.com/v1/chat/completions"
    llm_extra_headers_json: str = "{}"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_max_agent_steps: int = 8
    llm_temperature: float = 0.0

    @property
    def database_file(self) -> Path:
        path = Path(self.database_path)
        return path if path.is_absolute() else ROOT_DIR / path

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        return path if path.is_absolute() else ROOT_DIR / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
