from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./memorae.db"
    telegram_bot_token: str = ""
    telegram_webhook_secret: str = Field(default="change-me", min_length=1)
    telegram_api_timeout_seconds: float = Field(default=10.0, gt=0)
    telegram_confirmation_max_attempts: int = Field(default=3, ge=1, le=5)
    telegram_confirmation_retry_delay_seconds: float = Field(default=0.25, ge=0, le=5)
    telegram_confirmation_background_max_attempts: int = Field(default=3, ge=1, le=3)
    telegram_confirmation_background_retry_base_delay_seconds: float = Field(
        default=30.0, gt=0, le=3600
    )
    worker_poll_interval_seconds: float = Field(default=5.0, gt=0, le=60)
    worker_claim_lease_seconds: float = Field(default=60.0, gt=10, le=600)
    notification_retry_max_attempts: int = Field(default=3, ge=1, le=10)
    notification_uncertain_max_retries: int = Field(default=2, ge=0, le=2)
    notification_retry_base_delay_seconds: float = Field(default=30.0, gt=0, le=3600)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    llm_extract_model: str = "gpt-5.4-mini-2026-03-17"
    llm_intent_model: str = "openai/gpt-oss-20b"
    llm_timeout_seconds: float = Field(default=10.0, gt=0)
    conversation_session_ttl_seconds: int = Field(default=900, ge=60, le=86400)
    phone_default_region: str | None = None
    log_level: str = "INFO"
    log_format: str = "text"
    log_color: bool = True
    log_include_traceback: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)


def validate_runtime_settings(settings: Settings, *, allow_local_database: bool = False) -> None:
    """Reject deployment defaults that are unsafe for a production process."""
    if allow_local_database:
        return

    errors: list[str] = []
    bot_token = settings.telegram_bot_token.strip()
    if not bot_token or bot_token.lower() in {"replace-with-bot-token", "test-token"}:
        errors.append("TELEGRAM_BOT_TOKEN must be configured before production startup")

    secret = settings.telegram_webhook_secret.strip()
    if secret.lower() in {"change-me", "replace-with-random-secret"} or len(secret) < 32:
        errors.append("TELEGRAM_WEBHOOK_SECRET must be a unique value of at least 32 characters")

    scheme = urlparse(settings.database_url).scheme.lower()
    if not scheme.startswith("postgresql"):
        errors.append("DATABASE_URL must use PostgreSQL in production")

    if errors:
        raise RuntimeError("Invalid production configuration: " + "; ".join(errors))


@lru_cache
def get_settings() -> Settings:
    return Settings()
