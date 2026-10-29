from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./memorae.db"
    telegram_bot_token: str = ""
    telegram_webhook_secret: str = Field(default="change-me", min_length=1)
    telegram_api_timeout_seconds: float = Field(default=10.0, gt=0)
    log_level: str = "INFO"
    log_format: str = "text"
    log_include_traceback: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()
