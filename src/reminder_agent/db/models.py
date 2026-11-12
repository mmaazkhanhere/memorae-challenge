"""Compatibility alias for infrastructure database models."""

import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("reminder_agent.infrastructure.database.models")
