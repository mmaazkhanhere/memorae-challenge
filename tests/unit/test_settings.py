import pytest

from reminder_agent.settings import Settings, validate_runtime_settings


def test_production_configuration_rejects_placeholder_secret_and_sqlite() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(RuntimeError, match="TELEGRAM_WEBHOOK_SECRET"):
        validate_runtime_settings(settings)


def test_production_configuration_accepts_postgresql_and_strong_secret() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://user:password@db.example/app",
        telegram_bot_token="123456:production-token",
        telegram_webhook_secret="s" * 32,
        llm_api_key="production-llm-key",
    )

    validate_runtime_settings(settings)


def test_production_configuration_rejects_missing_bot_token() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://user:password@db.example/app",
        telegram_webhook_secret="s" * 32,
    )

    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        validate_runtime_settings(settings)


def test_production_configuration_rejects_missing_llm_credentials() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://user:password@db.example/app",
        telegram_bot_token="123456:production-token",
        telegram_webhook_secret="s" * 32,
    )

    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        validate_runtime_settings(settings)


def test_local_test_mode_allows_llm_to_be_disabled() -> None:
    validate_runtime_settings(Settings(_env_file=None), allow_local_database=True)


def test_production_embedding_provider_requires_its_own_credentials() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://user:password@db.example/app",
        telegram_bot_token="123456:production-token",
        telegram_webhook_secret="s" * 32,
        llm_api_key="production-llm-key",
        embedding_provider="openai_compatible",
    )

    with pytest.raises(RuntimeError, match="EMBEDDING_API_KEY"):
        validate_runtime_settings(settings)


def test_production_gemini_embedding_provider_requires_supported_model() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://user:password@db.example/app",
        telegram_bot_token="123456:production-token",
        telegram_webhook_secret="s" * 32,
        llm_api_key="production-llm-key",
        embedding_provider="gemini",
        embedding_api_key="gemini-key",
        embedding_gemini_model="gemini-embedding-2",
    )

    with pytest.raises(RuntimeError, match="EMBEDDING_GEMINI_MODEL"):
        validate_runtime_settings(settings)


def test_embedding_dimensions_must_match_migrated_vector_schema() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://user:password@db.example/app",
        telegram_bot_token="123456:production-token",
        telegram_webhook_secret="s" * 32,
        llm_api_key="production-llm-key",
        embedding_dimensions=768,
    )

    with pytest.raises(RuntimeError, match=r"vector\(1536\)"):
        validate_runtime_settings(settings)
