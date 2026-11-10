from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./memorae.db"
    telegram_bot_token: str = ""
    telegram_webhook_secret: str = Field(default="change-me", min_length=1)
    telegram_api_timeout_seconds: float = Field(default=10.0, gt=0)
    telegram_voice_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1024, le=50 * 1024 * 1024)
    telegram_voice_max_duration_seconds: int = Field(default=300, ge=1, le=3600)
    telegram_voice_log_transcript: bool = False
    telegram_confirmation_max_attempts: int = Field(default=3, ge=1, le=5)
    telegram_confirmation_retry_delay_seconds: float = Field(default=0.25, ge=0, le=5)
    telegram_confirmation_background_max_attempts: int = Field(default=3, ge=1, le=3)
    telegram_confirmation_background_retry_base_delay_seconds: float = Field(
        default=30.0, gt=0, le=3600
    )
    worker_poll_interval_seconds: float = Field(default=5.0, gt=0, le=60)
    worker_claim_lease_seconds: float = Field(default=60.0, gt=10, le=600)
    worker_delivery_grace_seconds: float = Field(default=15.0, ge=0, le=60)
    recurrence_materialization_horizon_days: int = Field(default=35, ge=7, le=366)
    recurrence_materialization_batch_size: int = Field(default=100, ge=1, le=1000)
    notification_retry_max_attempts: int = Field(default=3, ge=1, le=10)
    notification_uncertain_max_retries: int = Field(default=2, ge=0, le=2)
    notification_retry_base_delay_seconds: float = Field(default=30.0, gt=0, le=3600)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    llm_extract_model: str = "openai/gpt-oss-120b"
    llm_intent_model: str = "openai/gpt-oss-20b"
    llm_timeout_seconds: float = Field(default=10.0, gt=0)
    stt_base_url: str = "https://api.groq.com/openai/v1"
    stt_api_key: str | None = None
    stt_model: str = "whisper-large-v3-turbo"
    stt_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    embedding_provider: str = "disabled"
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    embedding_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_gemini_model: str = "gemini-embedding-001"
    embedding_dimensions: int = Field(default=1536, ge=1)
    embedding_timeout_seconds: float = Field(default=10.0, gt=0)
    embedding_allow_sensitive_external: bool = False
    memory_lexical_candidate_limit: int = Field(default=20, ge=2, le=50)
    memory_semantic_candidate_limit: int = Field(default=3, ge=2, le=20)
    memory_semantic_relevance_threshold: float = Field(default=0.68, ge=-1, le=1)
    memory_semantic_ambiguity_margin: float = Field(default=0.04, ge=0, le=1)
    conversation_session_ttl_seconds: int = Field(default=900, ge=60, le=86400)
    phone_default_region: str | None = None
    log_level: str = "INFO"
    log_format: str = "text"
    log_color: bool = True
    log_include_traceback: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)


def validate_runtime_settings(settings: Settings, *, allow_local_database: bool = False) -> None:
    """Reject unsafe production settings while preserving local/test mode."""
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

    if not (settings.llm_api_key or "").strip():
        errors.append("LLM_API_KEY must be configured before production startup")

    if not (settings.stt_api_key or "").strip():
        errors.append("STT_API_KEY must be configured before production startup")

    embedding_provider = settings.embedding_provider.strip().casefold()
    if embedding_provider not in {"disabled", "openai_compatible", "gemini"}:
        errors.append("EMBEDDING_PROVIDER must be disabled, openai_compatible, or gemini")
    if (
        embedding_provider in {"openai_compatible", "gemini"}
        and not (settings.embedding_api_key or "").strip()
    ):
        errors.append("EMBEDDING_API_KEY must be configured for semantic indexing")
    if embedding_provider == "gemini" and settings.embedding_gemini_model != "gemini-embedding-001":
        errors.append("EMBEDDING_GEMINI_MODEL must be gemini-embedding-001")
    if settings.embedding_dimensions != 1536:
        errors.append("EMBEDDING_DIMENSIONS must match the migrated vector(1536) schema")

    if errors:
        raise RuntimeError("Invalid production configuration: " + "; ".join(errors))


@lru_cache
def get_settings() -> Settings:
    return Settings()
