"""Generate the machine-readable capability manifest from the live registry.

The manifest freezes the release scope: every technique, dependency, service,
store, report, and notebook is classified as exactly one of

* ``required`` — installable and executable in the release with no optional
  extra and no silent skip, or
* ``removed`` — a non-implemented roadmap idea that must live in documentation,
  not as an executable product capability.

Doctrine (full release): every implemented or adapter technique is ``required``;
only non-implemented stubs are ``removed``. Regenerate after registry changes;
``tests/test_capability_manifest.py`` fails if the committed manifest drifts from
the registry or violates the doctrine.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import yaml

from kg_construction.application.registry import TechniqueDescriptor, default_registry

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "capability_manifest.yaml"

# Composite pipelines are wired from other techniques rather than a single class.
_COMPOSITION_OWNER = "composition"

# Runtime services required for a zero-skip release run.
_SERVICES = [
    {"name": "ollama", "kind": "external_service", "disposition": "required",
     "provisioning": "A local Ollama server must be running for llm/ollama techniques."},
    {"name": "neo4j", "kind": "external_service", "disposition": "required",
     "provisioning": "A Neo4j instance must be reachable for Neo4j export/ingestion."},
    {"name": "memgraph", "kind": "external_service", "disposition": "required",
     "provisioning": "A Memgraph instance must be reachable for Memgraph export."},
    {"name": "kuzu", "kind": "embedded", "disposition": "required",
     "provisioning": "Embedded on-disk database; no server process required."},
]

# Graph store backends the release writes to.
_STORES = [
    {"name": name, "disposition": "required"}
    for name in ("neo4j", "memgraph", "kuzu", "networkx", "rdf")
]


def _owner(descriptor: TechniqueDescriptor) -> str | None:
    if descriptor.adapter is not None:
        cls = descriptor.adapter
        return f"{cls.__module__}.{cls.__qualname__}"
    if descriptor.category == "hybrid":
        return _COMPOSITION_OWNER
    return None


def _disposition(descriptor: TechniqueDescriptor) -> str:
    return "removed" if descriptor.status == "stub" else "required"


def _techniques() -> list[dict[str, Any]]:
    registry = default_registry()
    rows: list[dict[str, Any]] = []
    for descriptor in sorted(registry.techniques, key=lambda item: item.name):
        rows.append(
            {
                "name": descriptor.name,
                "category": descriptor.category,
                "method_type": descriptor.method_type,
                "status": descriptor.status,
                "disposition": _disposition(descriptor),
                "owner": _owner(descriptor),
            }
        )
    return rows


def _dependencies() -> list[dict[str, Any]]:
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    rows: list[dict[str, Any]] = [
        {"group": "core", "kind": "runtime", "disposition": "required",
         "packages": list(project.get("dependencies", []))}
    ]
    tooling = {"dev", "notebook"}
    for group, packages in sorted(project.get("optional-dependencies", {}).items()):
        rows.append(
            {
                "group": group,
                "kind": "tooling" if group in tooling else "runtime",
                "disposition": "required",
                "packages": list(packages),
            }
        )
    return rows


def _files(subdir: str, patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    base = _ROOT / subdir
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(base.glob(pattern))
    return [
        {"path": str(path.relative_to(_ROOT)), "disposition": "required"}
        for path in sorted(paths)
    ]


def build_manifest() -> dict[str, Any]:
    techniques = _techniques()
    dependencies = _dependencies()
    reports = _files("reports", ("*.md", "*.json", "*.csv"))
    notebooks = _files("notebooks", ("*.ipynb",))

    def _count(rows: list[dict[str, Any]]) -> dict[str, int]:
        required = sum(1 for row in rows if row["disposition"] == "required")
        return {"total": len(rows), "required": required, "removed": len(rows) - required}

    return {
        "version": 1,
        "doctrine": "full_release",
        "generated_by": "scripts/generate_capability_manifest.py",
        "summary": {
            "techniques": _count(techniques),
            "dependencies": _count(dependencies),
            "services": _count(_SERVICES),
            "stores": _count(_STORES),
            "reports": _count(reports),
            "notebooks": _count(notebooks),
        },
        "techniques": techniques,
        "dependencies": dependencies,
        "services": _SERVICES,
        "stores": _STORES,
        "reports": reports,
        "notebooks": notebooks,
    }


_HEADER = (
    "# Capability manifest — the frozen release scope.\n"
    "#\n"
    "# Each capability is classified as exactly one of:\n"
    "#   required  installable and executable in the release, no optional extra, no silent skip\n"
    "#   removed   a non-implemented roadmap idea that lives in docs, not as a product capability\n"
    "#\n"
    "# Doctrine (full_release): every implemented or adapter technique is required; only\n"
    "# non-implemented stubs are removed. Regenerate with scripts/generate_capability_manifest.py;\n"
    "# tests/test_capability_manifest.py fails if this file drifts from the live registry.\n"
)


def write_manifest() -> Path:
    manifest = build_manifest()
    body = yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False, width=100)
    _MANIFEST.write_text(_HEADER + body, encoding="utf-8")
    return _MANIFEST


def main() -> None:
    path = write_manifest()
    manifest = build_manifest()
    summary = manifest["summary"]["techniques"]
    print(
        f"capability-manifest-ok: {path.name} "
        f"({summary['total']} techniques, {summary['required']} required, {summary['removed']} removed)"
    )


if __name__ == "__main__":
    main()
