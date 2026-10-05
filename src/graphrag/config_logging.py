"""Credential-safe copies for configuration diagnostics (no runtime mutation).

Keep this module dependency-free so both ArchRAG and the vendored GraphRAG can
use it. Pass mappings, argparse namespaces, or a model's ``model_dump()`` here;
never stringify an unfiltered configuration first.
"""

import argparse
import json
import re
from collections.abc import Mapping
from enum import Enum

REDACTED = "[REDACTED]"


def _sensitive_field(name: str) -> bool:
    # Handle API_KEY, api-key, apiKey and provider-prefixed variants alike.
    name = re.sub(r"[^a-z0-9]", "", name.lower())
    return (
        any(part in name for part in ("password", "passwd", "secret", "credential"))
        or name.endswith(
            (
                "key", "keys", "token", "authorization", "auth", "headers",
                "cookie", "cookies", "connectionstring", "proxy", "url",
                "endpoint", "apibase",
            )
        )
        or name in {"tokens", "organization", "containername", "connection"}
    )


def safe_config(value, *, _depth=0):
    """Return JSON-compatible diagnostics with sensitive field values removed.

    Nested mappings, namespaces, lists and tuples are copied. Endpoints/proxies
    are hidden in full because URLs can contain credentials in userinfo, paths,
    queries or fragments. Unsupported objects and excessive nesting are hidden
    without calling their potentially secret-bearing ``str``/``repr`` methods.
    This is field-based redaction, not a filter for secrets in arbitrary prose.
    """
    if _depth >= 20:
        return REDACTED
    if isinstance(value, argparse.Namespace):
        value = vars(value)
    if isinstance(value, Mapping):
        return {
            key: REDACTED if _sensitive_field(key) else safe_config(item, _depth=_depth + 1)
            for key, item in value.items()
            if isinstance(key, str)
        }
    if isinstance(value, (list, tuple)):
        return [safe_config(item, _depth=_depth + 1) for item in value]
    if isinstance(value, Enum):
        return safe_config(value.value, _depth=_depth + 1)
    if value is None or type(value) in (str, int, float, bool):
        return value
    return REDACTED


def format_config(value) -> str:
    """Format only the sanitized copy, never the original configuration."""
    return json.dumps(safe_config(value), indent=4)
