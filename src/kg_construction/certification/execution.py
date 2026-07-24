"""Execution-certificate validation for the mandatory capability set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

STAGES = ("declared", "configured", "instantiated", "executed", "evaluated", "passed")


@dataclass(frozen=True)
class ExecutionValidation:
    """Result of validating one complete certification execution report."""

    passed: bool
    detail: str
    declared: frozenset[str]
    completed: frozenset[str]


def validate_execution_report(
    manifest_path: Path,
    report_path: Path,
    *,
    expected_profile: str,
) -> ExecutionValidation:
    """Require exact capability coverage and every stage to be true.

    A report is valid only when the same set of names appears at all six stages:
    ``declared == configured == instantiated == executed == evaluated == passed``.
    Skips and fallbacks are prohibited even when a stage otherwise reports success.
    """

    manifest = _mapping(manifest_path)
    report = _mapping(report_path)
    declared_rows = manifest.get("techniques", [])
    if not isinstance(declared_rows, list):
        return ExecutionValidation(False, "manifest techniques must be a list", frozenset(), frozenset())
    all_declared = [
        str(row.get("name", ""))
        for row in declared_rows
        if isinstance(row, dict)
    ]
    if (
        "" in all_declared
        or not all_declared
        or len(all_declared) != len(set(all_declared))
    ):
        return ExecutionValidation(
            False,
            "manifest contains an empty/duplicate name or no techniques",
            frozenset(all_declared),
            frozenset(),
        )
    declared = frozenset(
        str(row["name"])
        for row in declared_rows
        if isinstance(row, dict) and _capability_applies(row, expected_profile)
    )
    if report.get("profile") != expected_profile:
        return ExecutionValidation(
            False,
            f"execution profile {report.get('profile')!r} does not match {expected_profile!r}",
            declared,
            frozenset(),
        )
    attestation = report.get("attestation")
    if (
        not isinstance(attestation, dict)
        or attestation.get("generated_by")
        != "kg_construction.certification.runner.dispatch_capabilities"
        or not attestation.get("run_id")
        or not attestation.get("manifest_sha256")
        or not attestation.get("plan_sha256")
    ):
        return ExecutionValidation(
            False,
            "report is not attested by the live capability dispatcher",
            declared,
            frozenset(),
        )

    rows = report.get("capabilities", [])
    if not isinstance(rows, list):
        return ExecutionValidation(False, "report capabilities must be a list", declared, frozenset())
    by_name: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("name"):
            return ExecutionValidation(False, "each report capability must be a named mapping", declared, frozenset())
        name = str(row["name"])
        if name in by_name:
            duplicates.append(name)
        by_name[name] = row
    if duplicates:
        return ExecutionValidation(False, f"duplicate capability rows: {sorted(set(duplicates))}", declared, frozenset())

    reported = frozenset(by_name)
    if reported != declared:
        missing = sorted(declared - reported)
        unexpected = sorted(reported - declared)
        return ExecutionValidation(
            False,
            f"capability coverage mismatch: missing={missing}, unexpected={unexpected}",
            declared,
            frozenset(),
        )

    stage_sets: dict[str, frozenset[str]] = {}
    violations: list[str] = []
    for stage in STAGES:
        stage_sets[stage] = frozenset(name for name, row in by_name.items() if row.get(stage) is True)
    for name, row in by_name.items():
        if row.get("skipped") is not False:
            violations.append(f"{name}: skipped must be false")
        if row.get("fallback") is not False:
            violations.append(f"{name}: fallback must be false")
        evidence = row.get("evidence")
        if not isinstance(evidence, dict) or not all(evidence.get(stage) for stage in STAGES[1:]):
            violations.append(f"{name}: non-empty evidence is required for configured..passed")
        process = row.get("process_evidence")
        if (
            not isinstance(process, dict)
            or process.get("returncode") != 0
            or not process.get("command")
            or not process.get("artifact_sha256")
            or not process.get("stdout_sha256")
            or not process.get("stderr_sha256")
        ):
            violations.append(f"{name}: fresh executable process evidence is required")
    unequal = {stage: sorted(declared - names) for stage, names in stage_sets.items() if names != declared}
    if unequal:
        violations.append(f"incomplete stages={unequal}")
    if violations:
        return ExecutionValidation(False, "; ".join(violations), declared, stage_sets["passed"])
    return ExecutionValidation(
        True,
        f"{len(declared)} capabilities completed every mandatory stage with zero skips/fallbacks",
        declared,
        stage_sets["passed"],
    )


def _mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _capability_applies(row: dict[str, Any], profile: str) -> bool:
    """Treat a missing profile list as compatible with every release profile."""

    profiles = row.get("profiles")
    return profiles is None or (
        isinstance(profiles, list) and profile in profiles
    )


__all__ = ["ExecutionValidation", "STAGES", "validate_execution_report"]
