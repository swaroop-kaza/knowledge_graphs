"""Generate the mandatory executable capability and roadmap inventories.

``capability_manifest.yaml`` is deliberately *not* a catalogue of aspirations.
It contains only executable release capabilities and every row is mandatory.
Research and roadmap entries are written to
``certification/research_inventory.yaml`` so they remain visible without
contaminating certification counts.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import yaml

from kg_construction.certification.registry import (
    TechniqueDescriptor,
    default_registry,
    research_registry,
)

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "capability_manifest.yaml"
_RESEARCH_INVENTORY = _ROOT / "certification" / "research_inventory.yaml"

_CORE_REPORTS = (
    "reports/current_registry_matrix.csv",
    "reports/current_registry_summary.json",
    "reports/current_repo_capability_summary.md",
    "reports/modelling_correctness.json",
    "reports/modelling_correctness.md",
    "reports/schema_quality.json",
    "reports/schema_quality.md",
)
_RESEARCH_REPORTS = (
    "reports/benchmark_leaderboard.csv",
    "reports/benchmark_leaderboard.md",
    "reports/cross_source_identity_benchmark.csv",
    "reports/cross_source_identity_benchmark.json",
    "reports/cross_source_identity_benchmark.md",
    "reports/gold_standard_benchmark_story.csv",
    "reports/gold_standard_benchmark_story.json",
    "reports/gold_standard_benchmark_story.md",
)

# Composite pipelines are wired from other techniques rather than a single class.
_COMPOSITION_OWNER = "composition"

# Runtime services required for a zero-skip release run.
_SERVICES = [
    {"name": "ollama", "kind": "external_service", "disposition": "mandatory",
     "provisioning": "A local Ollama server must be running for llm/ollama techniques."},
    {"name": "neo4j", "kind": "external_service", "disposition": "mandatory",
     "provisioning": "A Neo4j instance must be reachable for Neo4j export/ingestion."},
    {"name": "memgraph", "kind": "external_service", "disposition": "mandatory",
     "provisioning": "A Memgraph instance must be reachable for Memgraph export."},
    {"name": "ladybug", "kind": "embedded", "disposition": "mandatory",
     "provisioning": "Embedded on-disk database; no server process required."},
]

# Graph store backends the release writes to.
_STORES = [
    {"name": name, "disposition": "mandatory"}
    for name in ("neo4j", "memgraph", "ladybug", "networkx", "rdf")
]

_RENAMED_CAPABILITIES = {
    "kuzu_exporter": "ladybug_exporter",
    "kuzu_ingestion_adapter": "ladybug_ingestion_adapter",
}


def _owner(descriptor: TechniqueDescriptor) -> str | None:
    if descriptor.adapter is not None:
        cls = descriptor.adapter
        return f"{cls.__module__}.{cls.__qualname__}"
    if descriptor.category == "hybrid":
        return _COMPOSITION_OWNER
    return None


def _techniques() -> list[dict[str, Any]]:
    registry = default_registry()
    rows: list[dict[str, Any]] = []
    for descriptor in sorted(registry.techniques, key=lambda item: item.name):
        if descriptor.status == "stub":
            continue
        owner = _owner(descriptor)
        if owner:
            owner = owner.replace("Kuzu", "Ladybug")
        rows.append(
            {
                "name": _RENAMED_CAPABILITIES.get(descriptor.name, descriptor.name),
                "category": descriptor.category,
                "method_type": descriptor.method_type,
                "status": descriptor.status,
                "disposition": "mandatory",
                "profiles": list(descriptor.profiles),
                "owner": owner,
            }
        )
    return rows


def build_research_inventory() -> dict[str, Any]:
    """Return every research/roadmap entry outside the release registry."""

    release_names = default_registry().names()
    rows = []
    for descriptor in sorted(research_registry().techniques, key=lambda item: item.name):
        if descriptor.name in release_names:
            continue
        owner = _owner(descriptor)
        if owner:
            owner = owner.replace("Kuzu", "Ladybug")
        rows.append(
            {
                "name": _RENAMED_CAPABILITIES.get(descriptor.name, descriptor.name),
                "category": descriptor.category,
                "method_type": descriptor.method_type,
                "status": descriptor.status,
                "support_level": descriptor.public_support_level,
                "profiles": list(descriptor.profiles),
                "owner": owner,
                "certified": False,
            }
        )
    research_artifacts = [
        {
            "path": path,
            "kind": "report",
            "certified": False,
        }
        for path in _RESEARCH_REPORTS
    ]
    research_artifacts.extend(
        {
            "path": str(path.relative_to(_ROOT)),
            "kind": "notebook",
            "certified": False,
        }
        for path in sorted((_ROOT / "notebooks").glob("*.ipynb"))
    )
    return {
        "version": 1,
        "purpose": "research_and_roadmap_only",
        "generated_by": "scripts/generate_capability_manifest.py",
        "count": len(rows),
        "artifact_count": len(research_artifacts),
        "techniques": rows,
        "artifacts": research_artifacts,
    }


def _dependencies() -> list[dict[str, Any]]:
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    rows: list[dict[str, Any]] = [
        {"group": "core", "kind": "runtime", "disposition": "mandatory",
         "packages": list(project.get("dependencies", []))}
    ]
    # Product dependencies are all required (no optional extras); tooling lives in
    # PEP 735 dependency-groups and is part of the certified hard pass.
    groups = pyproject.get("dependency-groups", {})

    def resolve_group(group: str, parents: tuple[str, ...] = ()) -> list[str]:
        if group in parents:
            cycle = " -> ".join((*parents, group))
            raise ValueError(f"dependency-group include cycle: {cycle}")
        packages: list[str] = []
        for item in groups.get(group, []):
            if isinstance(item, str):
                packages.append(item)
            elif isinstance(item, dict) and isinstance(item.get("include-group"), str):
                packages.extend(
                    resolve_group(
                        item["include-group"],
                        (*parents, group),
                    )
                )
            else:
                raise ValueError(
                    f"unsupported dependency-group member in {group!r}: {item!r}"
                )
        return list(dict.fromkeys(packages))

    for group in sorted(groups):
        packages = resolve_group(group)
        rows.append(
            {"group": group, "kind": "tooling", "disposition": "mandatory", "packages": packages}
        )
    return rows


def _mandatory_files(paths: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {"path": path, "disposition": "mandatory"}
        for path in paths
    ]


def build_manifest() -> dict[str, Any]:
    techniques = _techniques()
    dependencies = _dependencies()
    reports = _mandatory_files(_CORE_REPORTS)
    notebooks: list[dict[str, Any]] = []

    def _count(rows: list[dict[str, Any]]) -> dict[str, int]:
        mandatory = sum(1 for row in rows if row["disposition"] == "mandatory")
        return {"total": len(rows), "mandatory": mandatory}

    return {
        "version": 1,
        "doctrine": "mandatory_executable_release",
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
    "# Capability manifest — mandatory executable release scope.\n"
    "#\n"
    "# Every row is mandatory. Research and roadmap entries are kept separately in\n"
    "# certification/research_inventory.yaml and never enter certification counts.\n"
    "# Regenerate with scripts/generate_capability_manifest.py.\n"
)


def write_manifest() -> Path:
    manifest = build_manifest()
    body = yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False, width=100)
    _MANIFEST.write_text(_HEADER + body, encoding="utf-8")
    _RESEARCH_INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    roadmap = yaml.safe_dump(
        build_research_inventory(),
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )
    _RESEARCH_INVENTORY.write_text(
        "# Research and roadmap inventory; excluded from release certification.\n" + roadmap,
        encoding="utf-8",
    )
    return _MANIFEST


def main() -> None:
    path = write_manifest()
    manifest = build_manifest()
    summary = manifest["summary"]["techniques"]
    print(
        f"capability-manifest-ok: {path.name} "
        f"({summary['total']} mandatory executable techniques; "
        f"{build_research_inventory()['count']} research/roadmap entries)"
    )


if __name__ == "__main__":
    main()
