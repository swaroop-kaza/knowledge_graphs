"""Generate checked-in registry capability reports."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from kg_construction.application.registry import (
    SUPPORT_LEVEL_DEFINITIONS,
    SUPPORT_LEVEL_LABELS,
    SupportLevel,
    TechniqueDescriptor,
    default_registry,
)

REPORTS_DIR = Path("reports")
SUPPORT_LEVEL_ORDER: tuple[SupportLevel, ...] = (
    "mandatory",
    "research",
    "roadmap",
)


def _adapter_name(technique: TechniqueDescriptor) -> str:
    return technique.adapter.__name__ if technique.adapter is not None else ""


def _counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _support_counter_payload(counter: Counter[SupportLevel]) -> dict[str, int]:
    return {level: counter[level] for level in SUPPORT_LEVEL_ORDER}


def _technique_payload(technique: TechniqueDescriptor) -> dict[str, Any]:
    return {
        "name": technique.name,
        "method_type": technique.method_type,
        "status": technique.status,
        "support_level": technique.public_support_level,
        "support_label": technique.public_support_label,
        "priority": technique.priority,
        "input_types": list(technique.input_types),
        "adapter": _adapter_name(technique),
        "tags": list(technique.tags),
        "profiles": list(technique.profiles),
        "notes": technique.notes,
    }


def _write_json_report(techniques: list[TechniqueDescriptor]) -> None:
    status_counts = Counter(technique.status for technique in techniques)
    support_counts = Counter(technique.public_support_level for technique in techniques)
    category_counts = Counter(technique.category for technique in techniques)
    method_type_counts = Counter(technique.method_type for technique in techniques)

    techniques_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for technique in sorted(techniques, key=lambda item: (item.category, item.priority, item.name)):
        techniques_by_category[technique.category].append(_technique_payload(technique))

    payload = {
        "report_date": date.today().isoformat(),
        "total_registered_techniques": len(techniques),
        "status_counts": _counter_payload(status_counts),
        "support_level_counts": _support_counter_payload(support_counts),
        "support_levels": [
            {
                "level": level,
                "label": SUPPORT_LEVEL_LABELS[level],
                "description": SUPPORT_LEVEL_DEFINITIONS[level],
            }
            for level in SUPPORT_LEVEL_ORDER
        ],
        "category_counts": _counter_payload(category_counts),
        "method_type_counts": _counter_payload(method_type_counts),
        "techniques_by_category": dict(techniques_by_category),
        "report_files": [
            "reports/current_repo_capability_summary.md",
            "reports/current_registry_summary.json",
            "reports/current_registry_matrix.csv",
        ],
    }
    (REPORTS_DIR / "current_registry_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv_report(techniques: list[TechniqueDescriptor]) -> None:
    path = REPORTS_DIR / "current_registry_matrix.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "name",
                "method_type",
                "status",
                "support_level",
                "support_label",
                "priority",
                "input_types",
                "adapter",
                "tags",
                "profiles",
            ],
        )
        writer.writeheader()
        for technique in sorted(techniques, key=lambda item: (item.category, item.priority, item.name)):
            writer.writerow(
                {
                    "category": technique.category,
                    "name": technique.name,
                    "method_type": technique.method_type,
                    "status": technique.status,
                    "support_level": technique.public_support_level,
                    "support_label": technique.public_support_label,
                    "priority": technique.priority,
                    "input_types": "|".join(technique.input_types),
                    "adapter": _adapter_name(technique),
                    "tags": "|".join(technique.tags),
                    "profiles": "|".join(technique.profiles),
                }
            )


def _format_list(items: list[TechniqueDescriptor]) -> str:
    if not items:
        return "none"
    return ", ".join(technique.name for technique in sorted(items, key=lambda item: (item.priority, item.name)))


def _write_markdown_report(techniques: list[TechniqueDescriptor]) -> None:
    status_counts = Counter(technique.status for technique in techniques)
    support_counts = Counter(technique.public_support_level for technique in techniques)
    category_counts = Counter(technique.category for technique in techniques)
    method_type_counts = Counter(technique.method_type for technique in techniques)
    by_category: dict[str, list[TechniqueDescriptor]] = defaultdict(list)
    for technique in techniques:
        by_category[technique.category].append(technique)

    lines = [
        "# Current Repo Capability Summary",
        "",
        f"Report date: {date.today().isoformat()}",
        "",
        "This report is generated from the live `kg_construction.application.registry.default_registry()` implementation registry. It replaces old article-specific and run-specific outputs with a compact source-of-truth view of the current repo capabilities.",
        "",
        "## Registry totals",
        "",
        f"- Total registered techniques: {len(techniques)}",
    ]
    for status, count in _counter_payload(status_counts).items():
        lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "## Public support levels",
            "",
            "The package is still `0.1.x` and classified as Alpha, but each registered technique has a user-facing support level:",
            "",
            "| Support level | Count | Meaning |",
            "| --- | ---: | --- |",
        ]
    )
    for level in SUPPORT_LEVEL_ORDER:
        lines.append(
            f"| {SUPPORT_LEVEL_LABELS[level]} | {support_counts[level]} | {SUPPORT_LEVEL_DEFINITIONS[level]} |"
        )

    lines.extend(["", "## Categories", "", "| Category | Count |", "| --- | ---: |"])
    for category, count in _counter_payload(category_counts).items():
        lines.append(f"| {category} | {count} |")

    lines.extend(["", "## Method types", "", "| Method type | Count |", "| --- | ---: |"])
    for method_type, count in _counter_payload(method_type_counts).items():
        lines.append(f"| {method_type} | {count} |")

    lines.extend(["", "## Capability groups", ""])
    for category in sorted(by_category):
        category_techniques = by_category[category]
        lines.extend([f"### {category}", ""])
        for level in SUPPORT_LEVEL_ORDER:
            items = [item for item in category_techniques if item.public_support_level == level]
            lines.append(f"- {SUPPORT_LEVEL_LABELS[level]}: {_format_list(items)}")
        lines.append("")

    lines.extend(
        [
            "## Artifact policy",
            "",
            "- Keep source code, configs, examples, notebooks, and concise current reports in the repo.",
            "- Treat `data/processed/` as a generated smoke-output workspace.",
            "- Treat `reports/experiment_datasets/` as a generated high-volume experiment-output workspace.",
            "- Regenerate bulky experiment outputs with `scripts/run_experiment_datasets.py` instead of storing stale graph dumps.",
            "- Keep article-specific narrative assets outside the repo unless they directly document current implementation behavior.",
            "",
            "## Current report files",
            "",
            "- `reports/current_repo_capability_summary.md`",
            "- `reports/current_registry_summary.json`",
            "- `reports/current_registry_matrix.csv`",
        ]
    )
    (REPORTS_DIR / "current_repo_capability_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Write all checked-in registry reports."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    techniques = default_registry().techniques
    _write_json_report(techniques)
    _write_csv_report(techniques)
    _write_markdown_report(techniques)
    print(f"Generated registry reports for {len(techniques)} techniques.")


if __name__ == "__main__":
    main()
