"""Global project configuration and module feature flags."""

from __future__ import annotations

import os

# Only modules that are actually gated at runtime.
# Admin panel / Prometheus / Grafana / RBAC are always-on via compose & code paths.
ENABLED_MODULES = [
    "self_diagnostics",
    "background_worker",
    "web_search",
]


def _modules_from_env() -> set[str] | None:
    raw = os.getenv("ENABLED_MODULES", "").strip()
    if not raw:
        return None
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def is_module_enabled(module_name: str) -> bool:
    """Return True when module is enabled in feature flag list (or ENABLED_MODULES env)."""
    override = _modules_from_env()
    enabled = override if override is not None else {name.lower() for name in ENABLED_MODULES}
    return module_name.strip().lower() in enabled
