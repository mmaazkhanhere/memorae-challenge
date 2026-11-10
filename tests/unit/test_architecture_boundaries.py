"""Static dependency rules for the feature-oriented package boundaries."""

import ast
from dataclasses import fields
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "reminder_agent"
CANONICAL_PACKAGE_PATHS = (
    SOURCE_ROOT / "application",
    SOURCE_ROOT / "features",
    SOURCE_ROOT / "channels",
    SOURCE_ROOT / "infrastructure",
    SOURCE_ROOT / "shared",
)
CHANNEL_NEUTRAL_PATHS = (
    SOURCE_ROOT / "application",
    SOURCE_ROOT / "features",
    SOURCE_ROOT / "shared",
)
CHANNEL_NEUTRAL_EXCEPTIONS = {
    SOURCE_ROOT / "application" / "bootstrap.py",
    # This compatibility coordinator still owns Telegram transaction
    # composition; its provider-specific adapter implementation lives under
    # channels/telegram.
    SOURCE_ROOT / "application" / "conversation_coordinator.py",
}
TELEGRAM_PREFIXES = ("reminder_agent.channels.telegram",)
APPLICATION_PREFIX = "reminder_agent.application"
FEATURE_PREFIX = "reminder_agent.features"
INFRASTRUCTURE_PREFIX = "reminder_agent.infrastructure"
CHANNEL_PREFIX = "reminder_agent.channels"

# These are existing composition/legacy edges.  They are deliberately keyed by
# the importing module, so a new module cannot inherit a package-wide exception.
INFRASTRUCTURE_CHANNEL_ALLOWLIST = {
    ("reminder_agent.infrastructure.runtime.worker", "reminder_agent.channels.telegram.provider"),
    (
        "reminder_agent.infrastructure.runtime.scheduling",
        "reminder_agent.channels.telegram.callbacks",
    ),
    (
        "reminder_agent.infrastructure.runtime.scheduling",
        "reminder_agent.channels.telegram.provider",
    ),
    (
        "reminder_agent.infrastructure.runtime.confirmations",
        "reminder_agent.channels.telegram.provider",
    ),
}

# Feature code is being prepared for incremental extraction.  Keep the current
# provider/database edges visible without allowing new ones to spread.
FEATURE_INFRASTRUCTURE_ALLOWLIST = {
    ("reminder_agent.features.intent", "reminder_agent.infrastructure"),
    ("reminder_agent.features.intent.__init__", "reminder_agent.infrastructure"),
    ("reminder_agent.features.intent.__init__", "reminder_agent.infrastructure.providers"),
    ("reminder_agent.features.intent.pipeline", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.intent.model.router", "reminder_agent.infrastructure.providers.llm"),
    ("reminder_agent.features.intent.model.router", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.intent.deterministic.interpreter", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.intent.routing", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.intent.model.interpreter", "reminder_agent.infrastructure.providers.llm"),
    ("reminder_agent.features.intent.model.interpreter", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.memory.persistence", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.memory.persistence", "reminder_agent.infrastructure.providers.embeddings"),
    ("reminder_agent.features.memory.persistence", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.memory.saving", "reminder_agent.infrastructure.providers.embeddings"),
    ("reminder_agent.features.memory.saving", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.memory.retrieve.answering", "reminder_agent.infrastructure.providers.llm"),
    ("reminder_agent.features.memory.retrieve.answering", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.reminders.listing.status", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.voice.telegram", "reminder_agent.infrastructure.providers.transcription"),
    ("reminder_agent.features.voice.telegram", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.memory.retrieve.persistence", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.memory.retrieve.persistence", "reminder_agent.infrastructure.providers.embeddings"),
    ("reminder_agent.features.memory.retrieve.persistence", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.memory.retrieve.workflow", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.memory.retrieve.workflow", "reminder_agent.infrastructure.providers.embeddings"),
    ("reminder_agent.features.memory.retrieve.workflow", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.memory.backfill", "reminder_agent.infrastructure.database.session"),
    ("reminder_agent.features.memory.backfill", "reminder_agent.infrastructure.providers.embeddings"),
    ("reminder_agent.features.memory.backfill", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.memory.backfill", "reminder_agent.infrastructure.config.settings"),
    ("reminder_agent.features.memory.adapters", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.memory.adapters", "reminder_agent.infrastructure.providers.embeddings"),
    ("reminder_agent.features.memory.adapters", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.reminders.cancel.decisions", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.complete.operations", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.complete.operations", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.reminders.complete.resolution", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.recurrence_service", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.recurrence_service", "reminder_agent.infrastructure.database.session"),
    ("reminder_agent.features.reminders.recurrence_service", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.reminders.create.reminder_creation_persistence", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.create.reminder_creation_persistence", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.reminders.listing.filters", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.snooze", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.snooze", "reminder_agent.infrastructure.observability"),
    # Reminder operations now live under the feature façade.  Keep their
    # database/observability edges explicit while channel adapters continue to
    # call the application boundary.
    ("reminder_agent.features.reminders.service", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.cancel.decisions", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.cancel.service", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.reminders.cancel.service", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.complete.clarification", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.complete.clarification", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.reminders.complete.operations", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.complete.operations", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.reminders.complete.pending", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.complete.resolution", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.complete.service", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.complete.service", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.reminders.create.service", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.create.service", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.reminders.listing.filters", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.listing.service", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.reschedule.decisions", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.reschedule.service", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.reminders.reschedule.service", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.snooze.operations", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.snooze.operations", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.reminders.snooze.service", "reminder_agent.infrastructure.observability"),
    ("reminder_agent.features.reminders.snooze.service", "reminder_agent.infrastructure.database.models"),
    ("reminder_agent.features.reminders.summary.telegram_adapter", "reminder_agent.infrastructure.database.models"),
}

# Memory adapters are feature-owned handlers, but they compose the application
# workflows and durable response adapters at the authenticated boundary.
MEMORY_ADAPTER_APPLICATION_ALLOWLIST = {
    ("reminder_agent.features.memory.adapters", "reminder_agent.application.conversations"),
    ("reminder_agent.features.memory.adapters", "reminder_agent.application.response_staging"),
    ("reminder_agent.features.memory.saving", "reminder_agent.application.contracts"),
    ("reminder_agent.features.memory.saving", "reminder_agent.application.identity.context"),
    ("reminder_agent.features.memory.saving", "reminder_agent.application.conversations"),
    ("reminder_agent.features.memory.retrieve.workflow", "reminder_agent.application.conversations"),
}

# The reminder feature is the shared application-service façade requested by
# the feature split.  Its orchestration retains trusted application contracts,
# sessions, and response staging while its deterministic operation modules stay
# feature-local.
REMINDER_APPLICATION_PREFIX = "reminder_agent.features.reminders"
FEATURE_DEPENDENCY_RULES = {
    "reminder_agent.features.memory": (
        "reminder_agent.features.intent",
        "reminder_agent.features.reminders",
        "reminder_agent.channels",
    ),
    "reminder_agent.features.onboarding": (
        "reminder_agent.features.reminders",
        "reminder_agent.features.memory",
        "reminder_agent.features.intent",
        "reminder_agent.channels",
    ),
    "reminder_agent.features.intent": (
        "reminder_agent.features.memory",
        "reminder_agent.channels",
    ),
}
FEATURE_DEPENDENCY_ALLOWLIST = {
    # Intent formats validated memory prompts but must not own their text.
    ("reminder_agent.features.intent.deterministic.interpreter", "reminder_agent.features.memory.messages"),
    ("reminder_agent.features.intent.model.interpreter", "reminder_agent.features.memory.messages"),
}
LEGACY_MODULE_PREFIXES = (
    "reminder_agent.api",
    "reminder_agent.telegram",
    "reminder_agent.onboarding",
    "reminder_agent.cancellation",
    "reminder_agent.completion",
    "reminder_agent.conversation_router",
    "reminder_agent.conversations",
    "reminder_agent.db",
    "reminder_agent.embeddings",
    "reminder_agent.idempotency",
    "reminder_agent.intent",
    "reminder_agent.memories",
    "reminder_agent.observability",
    "reminder_agent.recurrence",
    "reminder_agent.reminder_",
    "reminder_agent.reminders",
    "reminder_agent.request_policy",
    "reminder_agent.rescheduling",
    "reminder_agent.settings",
    "reminder_agent.snooze",
    "reminder_agent.transcription",
    "reminder_agent.worker",
)


def _paths_to_check(roots: tuple[Path, ...]) -> list[Path]:
    paths: list[Path] = []
    for path in roots:
        if path.is_dir():
            paths.extend(path.rglob("*.py"))
        elif path.is_file():
            paths.append(path)
    return paths


def _imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def _module_name(path: Path) -> str:
    return "reminder_agent." + path.relative_to(SOURCE_ROOT).with_suffix("").as_posix().replace("/", ".")


def _qualified_imports(path: Path) -> list[str]:
    """Return imports with enough context to make relative imports auditable."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    return _imported_modules(tree)


def _boundary_violations(
    roots: tuple[Path, ...], predicate,
) -> list[str]:
    violations: list[str] = []
    for path in _paths_to_check(roots):
        importer = _module_name(path)
        for imported in _qualified_imports(path):
            if predicate(importer, imported):
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)} -> {imported}")
    return violations


def _is_prefixed(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes)


def test_channel_neutral_modules_do_not_import_telegram_infrastructure() -> None:
    violations: list[str] = []
    for path in _paths_to_check(CHANNEL_NEUTRAL_PATHS):
        if path in CHANNEL_NEUTRAL_EXCEPTIONS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for module in _imported_modules(tree):
            if _is_prefixed(module, TELEGRAM_PREFIXES):
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)} -> {module}")
    assert violations == [], "channel boundary violations:\n" + "\n".join(violations)


def test_domain_features_do_not_import_application_or_channels() -> None:
    violations = _boundary_violations(
        tuple(
            path
            for path in (SOURCE_ROOT / "features").iterdir()
            if path.name != "onboarding"
        ),
        lambda importer, imported: _is_prefixed(imported, (APPLICATION_PREFIX, CHANNEL_PREFIX))
        and not _is_prefixed(importer, (REMINDER_APPLICATION_PREFIX,))
        and (importer, imported) not in MEMORY_ADAPTER_APPLICATION_ALLOWLIST,
    )
    assert violations == [], "domain-to-orchestration/channel violations:\n" + "\n".join(violations)


def test_onboarding_decision_workflow_is_provider_and_infrastructure_neutral() -> None:
    path = SOURCE_ROOT / "features" / "onboarding" / "phone_linking.py"
    imported = _qualified_imports(path)
    forbidden = (APPLICATION_PREFIX, INFRASTRUCTURE_PREFIX, CHANNEL_PREFIX)
    violations = [f"{path.relative_to(REPOSITORY_ROOT)} -> {module}" for module in imported
                  if _is_prefixed(module, forbidden)]
    assert violations == [], "onboarding decision workflow has external dependencies:\n" + "\n".join(
        violations
    )


def test_new_feature_infrastructure_edges_are_rejected() -> None:
    violations = _boundary_violations(
        tuple(
            path
            for path in (SOURCE_ROOT / "features").iterdir()
            if path.name != "onboarding"
        ),
        lambda importer, imported: _is_prefixed(imported, (INFRASTRUCTURE_PREFIX,))
        and (importer, imported) not in FEATURE_INFRASTRUCTURE_ALLOWLIST,
    )
    assert violations == [], "undocumented feature-to-infrastructure edges:\n" + "\n".join(violations)


def test_infrastructure_channel_edges_are_limited_to_delivery_composition() -> None:
    violations = _boundary_violations(
        (SOURCE_ROOT / "infrastructure",),
        lambda importer, imported: _is_prefixed(imported, (CHANNEL_PREFIX,))
        and (importer, imported) not in INFRASTRUCTURE_CHANNEL_ALLOWLIST,
    )
    assert violations == [], "infrastructure-to-channel violations:\n" + "\n".join(violations)


def test_channel_adapters_use_application_boundary_or_documented_legacy_edges() -> None:
    # Compatibility import paths are intentionally thin aliases.  No Telegram
    # module may import feature code directly; application workflows own that
    # orchestration.
    legacy_importers: set[str] = set()
    violations = _boundary_violations(
        (SOURCE_ROOT / "channels",),
        lambda importer, imported: _is_prefixed(imported, (FEATURE_PREFIX,))
        and not _is_prefixed(imported, ("reminder_agent.features.onboarding",))
            and importer not in legacy_importers
            and imported != "reminder_agent.features.reminders.create.presentation",
    )
    assert violations == [], "channel adapters bypass application services:\n" + "\n".join(violations)


def test_telegram_adapters_do_not_perform_database_business_operations() -> None:
    """Telegram may pass a session to application services, but not query or mutate it."""

    forbidden_methods = {"add", "delete", "execute", "get", "scalar", "scalars"}
    violations: list[str] = []
    for path in _paths_to_check((SOURCE_ROOT / "channels" / "telegram",)):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in forbidden_methods:
                continue
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and receiver.id in {"db", "session"}:
                violations.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno} -> {receiver.id}.{node.func.attr}()"
                )
    assert violations == [], "Telegram adapters own database operations:\n" + "\n".join(violations)


def test_telegram_adapters_do_not_import_feature_implementation_modules() -> None:
    violations = _boundary_violations(
        (SOURCE_ROOT / "channels" / "telegram",),
        lambda _importer, imported: _is_prefixed(imported, (FEATURE_PREFIX,))
        and not _is_prefixed(imported, ("reminder_agent.features.onboarding",))
        and imported != "reminder_agent.features.reminders.create.presentation",
    )
    assert violations == [], "Telegram adapters import feature implementations:\n" + "\n".join(
        violations
    )


def test_canonical_outbound_response_has_no_provider_fields() -> None:
    from reminder_agent.application.contracts import OutboundResponse

    field_names = {field.name for field in fields(OutboundResponse)}
    assert field_names == {
        "content",
        "response_kind",
        "resource_id",
        "affected_count",
        "presentation_hint",
        "idempotency_key",
        "should_send",
        "is_duplicate",
    }
    assert not field_names.intersection(
        {"chat_id", "reply_markup", "callback_data", "telegram_message_id"}
    )


def test_canonical_modules_do_not_import_legacy_compatibility_paths() -> None:
    violations: list[str] = []
    for path in _paths_to_check(CANONICAL_PACKAGE_PATHS):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for module in _imported_modules(tree):
            if _is_prefixed(module, LEGACY_MODULE_PREFIXES):
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)} -> {module}")
    assert violations == [], "canonical modules importing compatibility paths:\n" + "\n".join(
        violations
    )


def test_feature_dependencies_do_not_cross_private_boundaries() -> None:
    violations: list[str] = []
    for path in _paths_to_check((SOURCE_ROOT / "features",)):
        module = "reminder_agent." + path.relative_to(SOURCE_ROOT).with_suffix(
            ""
        ).as_posix().replace("/", ".")
        owner = next(
            (prefix for prefix in FEATURE_DEPENDENCY_RULES if _is_prefixed(module, (prefix,))),
            None,
        )
        if _is_prefixed(module, ("reminder_agent.features.onboarding",)):
            # Onboarding is a feature workflow package, not a leaf domain package;
            # its orchestration intentionally composes identity, intent, memory,
            # and reminder services.
            continue
        if (
            owner is None
            or path.name == "__init__.py"
            and owner == "reminder_agent.features.intent"
            or module == "reminder_agent.features.intent.routing"
        ):
            # The intent package facade retains a documented compatibility export
            # for the old memory-answer provider identity.
            continue
        forbidden = FEATURE_DEPENDENCY_RULES[owner]
        for imported in _imported_modules(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if (
                _is_prefixed(imported, forbidden)
                and (module, imported) not in FEATURE_DEPENDENCY_ALLOWLIST
            ):
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)} -> {imported}")
    assert violations == [], "feature dependency violations:\n" + "\n".join(violations)


def test_provider_and_telegram_callback_ownership_is_at_the_boundary() -> None:
    assert (SOURCE_ROOT / "infrastructure" / "providers" / "llm.py").is_file()
    assert (SOURCE_ROOT / "channels" / "telegram" / "callbacks.py").is_file()
    assert not (SOURCE_ROOT / "features" / "intent" / "intent_provider.py").exists()
    assert not (SOURCE_ROOT / "features" / "intent" / "intent_routing_rules.py").exists()
    completion = (SOURCE_ROOT / "features" / "reminders" / "complete" / "operations.py").read_text(
        encoding="utf-8-sig"
    )
    assert "def callback_data" not in completion
    assert "def parse_callback_data" not in completion


def test_canonical_package_directories_exist() -> None:
    assert all(path.is_dir() for path in CANONICAL_PACKAGE_PATHS)
    assert (SOURCE_ROOT / "features" / "reminders").is_dir()
    assert (SOURCE_ROOT / "features" / "memory").is_dir()
    assert (SOURCE_ROOT / "features" / "intent").is_dir()
    assert (SOURCE_ROOT / "features" / "onboarding").is_dir()
    assert (SOURCE_ROOT / "channels" / "telegram").is_dir()


def test_root_compatibility_alias_files_are_removed() -> None:
    assert {path.name for path in SOURCE_ROOT.glob("*.py")} == {"__init__.py"}


def test_canonical_internal_import_graph_has_no_cycles() -> None:
    modules: dict[str, set[str]] = {}
    for path in _paths_to_check(CANONICAL_PACKAGE_PATHS):
        module = "reminder_agent." + path.relative_to(SOURCE_ROOT).with_suffix(
            ""
        ).as_posix().replace("/", ".")
        modules[module] = set()

    for path in _paths_to_check(CANONICAL_PACKAGE_PATHS):
        module = "reminder_agent." + path.relative_to(SOURCE_ROOT).with_suffix(
            ""
        ).as_posix().replace("/", ".")
        for imported in _imported_modules(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if imported in modules:
                modules[module].add(imported)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visiting:
            raise AssertionError(f"import cycle detected at {module}")
        if module in visited:
            return
        visiting.add(module)
        for dependency in modules[module]:
            visit(dependency)
        visiting.remove(module)
        visited.add(module)

    for module in modules:
        visit(module)
