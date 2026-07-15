"""Global project configuration and module feature flags."""

from __future__ import annotations

ENABLED_MODULES = [
    "multi_agent",
    "role_based_access",
    "self_diagnostics",
    "background_worker",
    "web_admin",
    "monitoring",
    "web_search",
    "conversation_memory",
]


def is_module_enabled(module_name: str) -> bool:
    """Return True when module is enabled in feature flag list."""
    return module_name.strip().lower() in {name.lower() for name in ENABLED_MODULES}
