"""Compatibility alias for the Telegram webhook adapter."""

import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("reminder_agent.channels.telegram.webhook")
