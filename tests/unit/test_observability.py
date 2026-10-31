import logging

from colorama import Fore

from reminder_agent.observability import JsonFormatter, ReadableFormatter, log_event


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
