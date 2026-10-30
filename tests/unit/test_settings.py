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
