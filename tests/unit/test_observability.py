import logging

from colorama import Fore

from reminder_agent.infrastructure.observability import (
    JsonFormatter,
    ReadableFormatter,
    exception_details,
    log_event,
)


def _record(level: int = logging.INFO) -> logging.LogRecord:
    logger = logging.getLogger("test_observability")
    record = logger.makeRecord(
        logger.name,
        level,
        __file__,
        1,
        "telegram_update_received",
        (),
        None,
    )
    record.event_name = "telegram_update_received"
    record.context = {"update_id": 101, "has_message": True}
    return record


def test_readable_logs_use_colorama_and_human_readable_event_names() -> None:
    rendered = ReadableFormatter(use_color=True).format(_record())

    assert Fore.GREEN in rendered
    assert "telegram update received" in rendered
    assert "update_id=101" in rendered


def test_readable_logs_can_disable_color_for_non_interactive_sinks() -> None:
    rendered = ReadableFormatter(use_color=False).format(_record(logging.WARNING))

    assert "\x1b[" not in rendered
    assert "WARNING" in rendered


def test_error_logs_are_fully_red() -> None:
    rendered = ReadableFormatter(use_color=True).format(_record(logging.ERROR))

    assert Fore.RED in rendered
    assert rendered.endswith("\x1b[0m")

    failed = _record(logging.WARNING)
    failed.event_name = "memory_answer_provider_failed"
    failed_rendered = ReadableFormatter(use_color=True).format(failed)
    assert Fore.RED in failed_rendered


def test_exception_details_include_http_diagnostics_without_response_body() -> None:
    import httpx

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request, text="secret response body")
    error = httpx.HTTPStatusError("provider rejected request", request=request, response=response)

    details = exception_details(error)

    assert details["exception_type"] == "HTTPStatusError"
    assert details["status_code"] == 400
    assert details["request_method"] == "POST"
    assert details["request_host"] == "api.groq.com"
    assert "secret response body" not in str(details)


def test_json_logs_stay_structured_and_uncolored() -> None:
    rendered = JsonFormatter().format(_record())

    assert '"event":"telegram_update_received"' in rendered
    assert "\x1b[" not in rendered


def test_log_event_attaches_event_name_and_context() -> None:
    logger = logging.getLogger("test_observability.event")
    captured: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        log_event(logger, logging.INFO, "reminder_notification_scheduled", reminder_id="safe-id")
    finally:
        logger.removeHandler(handler)

    assert captured[0].event_name == "reminder_notification_scheduled"
    assert captured[0].context == {"reminder_id": "safe-id"}
