"""Reminder scheduling, retrieval, and lifecycle operations.

``service`` is the shared, owner-scoped façade for database-backed reminder
commands.  Operation packages contain deterministic parsing, validation, and
clarification orchestration; they do not take transaction ownership.
"""
