"""One explicit runtime contract for the locked local graph/model services.

The values here describe the repository's digest-pinned certification stack,
not production credentials.  Every setup, readiness, doctor, public
construction, and certification path resolves this same contract so an omitted
environment variable cannot silently select a different endpoint or database.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Literal, cast

from kg_construction.domain.models import canonical_json


def _load_contract_values() -> dict[str, str]:
    raw = files("kg_construction").joinpath(
        "resources/service-runtime.env"
    ).read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for line_number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, value = stripped.partition("=")
        if not separator or not name or not value:
            raise RuntimeError(
                "invalid locked service runtime line "
                f"{line_number}: expected NAME=value"
            )
        if name in values:
            raise RuntimeError(f"duplicate locked service runtime key: {name}")
        values[name] = value
    expected = {
        "SERVICE_RUNTIME_CONTRACT_VERSION",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "NEO4J_DATABASE",
        "MEMGRAPH_URI",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
    }
    if set(values) != expected:
        raise RuntimeError(
            "locked service runtime keys differ: "
            f"missing={sorted(expected - set(values))}, "
            f"unknown={sorted(set(values) - expected)}"
        )
    if values["OLLAMA_MODEL"] != "qwen3:1.7b-q4_K_M":
        raise RuntimeError(
            "service-runtime.env Ollama model is outside the typed release contract"
        )
    return values


_CONTRACT_VALUES = _load_contract_values()
DEFAULT_NEO4J_URI = _CONTRACT_VALUES["NEO4J_URI"]
DEFAULT_NEO4J_USER = _CONTRACT_VALUES["NEO4J_USER"]
DEFAULT_NEO4J_PASSWORD = _CONTRACT_VALUES["NEO4J_PASSWORD"]
DEFAULT_NEO4J_DATABASE = _CONTRACT_VALUES["NEO4J_DATABASE"]
DEFAULT_MEMGRAPH_URI = _CONTRACT_VALUES["MEMGRAPH_URI"]
DEFAULT_OLLAMA_BASE_URL = _CONTRACT_VALUES["OLLAMA_BASE_URL"]
DEFAULT_OLLAMA_MODEL: Literal["qwen3:1.7b-q4_K_M"] = cast(
    Literal["qwen3:1.7b-q4_K_M"],
    _CONTRACT_VALUES["OLLAMA_MODEL"],
)
SERVICE_RUNTIME_CONTRACT_VERSION = _CONTRACT_VALUES[
    "SERVICE_RUNTIME_CONTRACT_VERSION"
]

_DEFAULTS = {
    "NEO4J_URI": DEFAULT_NEO4J_URI,
    "NEO4J_USER": DEFAULT_NEO4J_USER,
    "NEO4J_PASSWORD": DEFAULT_NEO4J_PASSWORD,
    "NEO4J_DATABASE": DEFAULT_NEO4J_DATABASE,
    "MEMGRAPH_URI": DEFAULT_MEMGRAPH_URI,
    "OLLAMA_BASE_URL": DEFAULT_OLLAMA_BASE_URL,
    "OLLAMA_MODEL": DEFAULT_OLLAMA_MODEL,
}
_LEGACY_CERT_ALIASES = {
    "NEO4J_URI": "KG_CERT_NEO4J_URI",
    "NEO4J_USER": "KG_CERT_NEO4J_USER",
    "NEO4J_PASSWORD": "KG_CERT_NEO4J_PASSWORD",
    "NEO4J_DATABASE": "KG_CERT_NEO4J_DATABASE",
    "MEMGRAPH_URI": "KG_CERT_MEMGRAPH_URI",
    "OLLAMA_BASE_URL": "KG_CERT_OLLAMA_URL",
    "OLLAMA_MODEL": "KG_CERT_OLLAMA_MODEL",
}


@dataclass(frozen=True, slots=True)
class LockedServiceRuntime:
    """Resolved local-service endpoints and credentials for one process."""

    neo4j_uri: str = DEFAULT_NEO4J_URI
    neo4j_user: str = DEFAULT_NEO4J_USER
    neo4j_password: str = field(
        default=DEFAULT_NEO4J_PASSWORD,
        repr=False,
    )
    neo4j_database: str = DEFAULT_NEO4J_DATABASE
    memgraph_uri: str = DEFAULT_MEMGRAPH_URI
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    version: str = SERVICE_RUNTIME_CONTRACT_VERSION

    def __post_init__(self) -> None:
        values = {
            "neo4j_uri": self.neo4j_uri,
            "neo4j_user": self.neo4j_user,
            "neo4j_password": self.neo4j_password,
            "neo4j_database": self.neo4j_database,
            "memgraph_uri": self.memgraph_uri,
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
        }
        empty = sorted(name for name, value in values.items() if not value.strip())
        if empty:
            raise ValueError(
                f"locked service runtime values must not be empty: {empty}"
            )
        if not self.neo4j_uri.startswith(("bolt://", "neo4j://")):
            raise ValueError("NEO4J_URI must use bolt:// or neo4j://")
        if not self.memgraph_uri.startswith("bolt://"):
            raise ValueError("MEMGRAPH_URI must use bolt://")
        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL must use http:// or https://")
        if self.version != SERVICE_RUNTIME_CONTRACT_VERSION:
            raise ValueError(
                "service runtime contract version does not match this release"
            )

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> LockedServiceRuntime:
        """Resolve canonical variables, rejecting conflicting legacy aliases."""

        if environ is None:
            import os

            environ = os.environ
        resolved = {
            name: _resolve_environment_value(name, environ)
            for name in _DEFAULTS
        }
        return cls(
            neo4j_uri=resolved["NEO4J_URI"],
            neo4j_user=resolved["NEO4J_USER"],
            neo4j_password=resolved["NEO4J_PASSWORD"],
            neo4j_database=resolved["NEO4J_DATABASE"],
            memgraph_uri=resolved["MEMGRAPH_URI"],
            ollama_base_url=resolved["OLLAMA_BASE_URL"].rstrip("/"),
            ollama_model=resolved["OLLAMA_MODEL"],
        )

    def as_environment(self) -> dict[str, str]:
        """Return the canonical variables inherited by every child command."""

        return {
            "NEO4J_URI": self.neo4j_uri,
            "NEO4J_USER": self.neo4j_user,
            "NEO4J_PASSWORD": self.neo4j_password,
            "NEO4J_DATABASE": self.neo4j_database,
            "MEMGRAPH_URI": self.memgraph_uri,
            "OLLAMA_BASE_URL": self.ollama_base_url,
            "OLLAMA_MODEL": self.ollama_model,
        }

    @property
    def contract_sha256(self) -> str:
        """Hash non-secret topology plus the local credential contract."""

        return hashlib.sha256(
            canonical_json(
                {
                    "version": self.version,
                    **self.as_environment(),
                }
            ).encode("utf-8")
        ).hexdigest()


def _resolve_environment_value(
    canonical_name: str,
    environ: Mapping[str, str],
) -> str:
    alias = _LEGACY_CERT_ALIASES[canonical_name]
    canonical = environ.get(canonical_name)
    legacy = environ.get(alias)
    if canonical is not None and not canonical.strip():
        raise ValueError(f"{canonical_name} is set but empty")
    if legacy is not None and not legacy.strip():
        raise ValueError(f"{alias} is set but empty")
    if canonical is not None and legacy is not None and canonical != legacy:
        raise ValueError(
            f"conflicting service runtime variables: {canonical_name} and {alias}"
        )
    return canonical or legacy or _DEFAULTS[canonical_name]


__all__ = [
    "DEFAULT_MEMGRAPH_URI",
    "DEFAULT_NEO4J_DATABASE",
    "DEFAULT_NEO4J_PASSWORD",
    "DEFAULT_NEO4J_URI",
    "DEFAULT_NEO4J_USER",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "LockedServiceRuntime",
    "SERVICE_RUNTIME_CONTRACT_VERSION",
]
