"""Dispatch trusted per-capability certifiers and capture fresh evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from kg_construction.certification.execution import STAGES


def dispatch_capabilities(
    manifest_path: Path,
    plan_path: Path,
    *,
    profile: str,
    output_path: Path,
) -> dict[str, Any]:
    """Run every declared certifier; a pre-existing report is never accepted."""

    manifest = _mapping(manifest_path)
    plan = _mapping(plan_path)
    declared = [
        row["name"]
        for row in manifest.get("techniques", [])
        if isinstance(row, dict) and _capability_applies(row, profile)
    ]
    contracts = plan.get("capabilities", {})
    if not isinstance(contracts, dict):
        contracts = {}
    run_id = uuid.uuid4().hex
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"kg-cert-{run_id[:8]}-") as directory:
        evidence_root = Path(directory)
        for name in declared:
            contract = contracts.get(name)
            rows.append(
                _dispatch_one(
                    name,
                    contract if isinstance(contract, dict) else None,
                    profile=profile,
                    run_id=run_id,
                    evidence_root=evidence_root,
                )
            )

    report = {
        "version": 2,
        "profile": profile,
        "attestation": {
            "generated_by": "kg_construction.certification.runner.dispatch_capabilities",
            "run_id": run_id,
            "manifest_sha256": _file_sha256(manifest_path),
            "plan_sha256": _file_sha256(plan_path) if plan_path.is_file() else None,
            "applicable_capability_count": len(declared),
        },
        "capabilities": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(report, sort_keys=False, width=100), encoding="utf-8")
    return report


def _capability_applies(row: dict[str, Any], profile: str) -> bool:
    """Return profile applicability; absent metadata preserves compatibility."""

    profiles = row.get("profiles")
    return profiles is None or (
        isinstance(profiles, list) and profile in profiles
    )


def _dispatch_one(
    name: str,
    contract: dict[str, Any] | None,
    *,
    profile: str,
    run_id: str,
    evidence_root: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        **{stage: False for stage in STAGES},
        "skipped": False,
        "fallback": False,
        "evidence": {},
    }
    row["declared"] = True
    if contract is None:
        row["failure"] = "no executable certifier contract"
        return row
    command = contract.get("command")
    profiles = contract.get("profiles", [])
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
        or profile not in profiles
    ):
        row["failure"] = "certifier command/profile contract is invalid"
        return row
    row["configured"] = True
    row["evidence"]["configured"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()

    evidence_path = evidence_root / f"{len(list(evidence_root.iterdir())):04d}.json"
    environment = dict(os.environ)
    environment.update(
        {
            "KG_CERT_CAPABILITY": name,
            "KG_CERT_PROFILE": profile,
            "KG_CERT_RUN_ID": run_id,
            "KG_CERT_EVIDENCE_PATH": str(evidence_path),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "UV_OFFLINE": "1",
        }
    )
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=contract.get("cwd"),
            env=environment,
            capture_output=True,
            text=True,
            timeout=float(contract.get("timeout_seconds", 300)),
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        row["failure"] = f"certifier dispatch failed: {type(exc).__name__}: {exc}"
        return row
    duration = time.monotonic() - started
    row["process_evidence"] = {
        "command": command,
        "returncode": completed.returncode,
        "duration_seconds": round(duration, 6),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
    }
    if completed.returncode != 0:
        row["failure"] = f"certifier returned {completed.returncode}"
        return row
    if not evidence_path.is_file():
        row["failure"] = "certifier produced no fresh evidence file"
        return row
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        row["failure"] = f"invalid evidence: {type(exc).__name__}: {exc}"
        return row
    if (
        evidence.get("capability") != name
        or evidence.get("profile") != profile
        or evidence.get("run_id") != run_id
    ):
        row["failure"] = "evidence identity/run nonce mismatch"
        return row
    observations = evidence.get("observations")
    if not isinstance(observations, dict):
        row["failure"] = "evidence observations are missing"
        return row
    evidence_problem = _semantic_evidence_problem(observations)
    if evidence_problem:
        row["failure"] = evidence_problem
        return row
    row["instantiated"] = observations.get("instantiated") is True
    row["executed"] = observations.get("executed") is True
    row["evaluated"] = observations.get("evaluated") is True
    row["passed"] = (
        observations.get("passed") is True
        and completed.returncode == 0
        and row["instantiated"]
        and row["executed"]
        and row["evaluated"]
    )
    row["skipped"] = observations.get("skipped") is not False
    row["fallback"] = observations.get("fallback") is not False
    for stage in STAGES[1:]:
        value = observations.get(f"{stage}_evidence")
        if stage == "configured":
            continue
        row["evidence"][stage] = value
    row["process_evidence"]["artifact_sha256"] = _file_sha256(evidence_path)
    if not row["passed"] or row["skipped"] or row["fallback"]:
        row["failure"] = "semantic evidence did not prove a zero-skip, zero-fallback pass"
    return row


def _semantic_evidence_problem(observations: dict[str, Any]) -> str | None:
    instantiated = observations.get("instantiated_evidence")
    if not isinstance(instantiated, dict) or not instantiated.get("implementation"):
        return "instantiation evidence must identify the concrete implementation"
    executed = observations.get("executed_evidence")
    if (
        not isinstance(executed, dict)
        or not _is_sha256(executed.get("fixture_sha256"))
        or not _is_sha256(executed.get("output_sha256"))
    ):
        return "execution evidence must bind immutable fixture and output SHA-256 values"
    evaluated = observations.get("evaluated_evidence")
    if (
        not isinstance(evaluated, dict)
        or not isinstance(evaluated.get("metrics"), dict)
        or not evaluated["metrics"]
        or not isinstance(evaluated.get("thresholds"), dict)
        or not evaluated["thresholds"]
    ):
        return "evaluation evidence must contain metrics and thresholds"
    passed = observations.get("passed_evidence")
    assertions = passed.get("assertions") if isinstance(passed, dict) else None
    if (
        not isinstance(assertions, list)
        or not assertions
        or not all(isinstance(item, dict) and item.get("passed") is True for item in assertions)
    ):
        return "pass evidence must contain successful named assertions"
    return None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["dispatch_capabilities"]
