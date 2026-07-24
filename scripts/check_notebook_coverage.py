"""Check release coverage while allowing documented research techniques."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
COVERAGE_PATH = NOTEBOOK_DIR / "capability_coverage.yaml"

sys.path.insert(0, str(ROOT / "src"))

from kg_construction.application.registry import (  # noqa: E402
    default_registry,
    research_registry,
)

REQUIRED_STATUSES = {"implemented", "adapter"}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def _as_name_list(value: Any, *, section: str, notebook: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{section}.{notebook} must be a list.")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{section}.{notebook} contains a non-string technique name.")
        names.append(item)
    return names


def main() -> int:
    failures: list[str] = []
    if not COVERAGE_PATH.exists():
        print(f"{COVERAGE_PATH}: missing coverage map")
        return 1

    try:
        payload = _load_yaml(COVERAGE_PATH)
    except Exception as exc:
        print(f"{COVERAGE_PATH}: {exc}")
        return 1

    coverage = payload.get("coverage", {})
    extension_points = payload.get("extension_points", {})
    if not isinstance(coverage, dict):
        failures.append("coverage must be a mapping of notebook filename to technique names.")
        coverage = {}
    if not isinstance(extension_points, dict):
        failures.append("extension_points must be a mapping of notebook filename to technique names.")
        extension_points = {}

    release_registry = default_registry()
    inventory_registry = research_registry()
    by_name = {
        technique.name: technique for technique in inventory_registry.techniques
    }
    required = {
        technique.name
        for technique in release_registry.techniques
        if technique.status in REQUIRED_STATUSES
    }

    covered: set[str] = set()
    documented_extension_points: set[str] = set()

    for section_name, section, destination in [
        ("coverage", coverage, covered),
        ("extension_points", extension_points, documented_extension_points),
    ]:
        for notebook_name, names_payload in section.items():
            if not isinstance(notebook_name, str):
                failures.append(f"{section_name}: notebook key must be a string.")
                continue
            notebook_path = NOTEBOOK_DIR / notebook_name
            if not notebook_path.exists():
                failures.append(f"{section_name}.{notebook_name}: notebook file does not exist.")
            try:
                names = _as_name_list(names_payload, section=section_name, notebook=notebook_name)
            except ValueError as exc:
                failures.append(str(exc))
                continue
            for name in names:
                if name not in by_name:
                    failures.append(f"{section_name}.{notebook_name}: unknown technique {name!r}.")
                    continue
                destination.add(name)

    missing = sorted(required - covered)
    for name in missing:
        technique = by_name[name]
        failures.append(
            f"missing notebook coverage for {name!r} "
            f"(category={technique.category}, status={technique.status})."
        )

    stub_in_required_coverage = sorted(
        name for name in covered if name in by_name and by_name[name].status == "stub"
    )
    for name in stub_in_required_coverage:
        failures.append(
            f"{name!r} is a stub; document it under extension_points instead of coverage."
        )

    undocumented_stubs = {
        technique.name
        for technique in inventory_registry.techniques
        if technique.status == "stub"
    } - documented_extension_points
    if undocumented_stubs:
        failures.append(
            "undocumented stub extension points: "
            + ", ".join(sorted(undocumented_stubs))
        )

    if failures:
        for failure in failures:
            print(failure)
        return 1

    print(
        "notebook-coverage-ok: "
        f"{len(covered & required)} required techniques covered, "
        f"{len(documented_extension_points)} stub extension points documented"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
