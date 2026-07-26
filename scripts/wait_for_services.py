"""Require every locked certification service and its semantic smoke query."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from kg_construction.certification.profiles import select_profile
from kg_construction.service_runtime import LockedServiceRuntime

_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=float(os.getenv("SERVICE_WAIT_TIMEOUT", "120")))
    parser.add_argument("--profile", default=os.getenv("KG_CERT_PROFILE"))
    parser.add_argument(
        "--ollama-url",
        default=None,
        help=(
            "Compatibility assertion only; it must match OLLAMA_BASE_URL from "
            "the locked service environment."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        runtime = LockedServiceRuntime.from_environment()
    except ValueError as exc:
        print(f"[FAIL] locked-service-runtime: {exc}")
        return 1
    if (
        args.ollama_url is not None
        and args.ollama_url.rstrip("/") != runtime.ollama_base_url
    ):
        print(
            "[FAIL] locked-service-runtime: --ollama-url must match "
            "OLLAMA_BASE_URL"
        )
        return 1
    manifest = _mapping(_ROOT / "models.manifest.yaml")
    lock = _mapping(_ROOT / "models.lock.yaml")
    try:
        profile = select_profile(manifest, args.profile)
    except ValueError as exc:
        print(f"[FAIL] platform-profile: {exc}")
        return 1
    required = set(profile.mandatory_services)
    probes: dict[str, Callable[[], str]] = {
        "ollama": lambda: _probe_ollama(
            runtime.ollama_base_url,
            lock,
            profile.name,
        ),
        "neo4j": lambda: _probe_neo4j(runtime),
        "memgraph": lambda: _probe_memgraph(runtime),
        "ladybug": _probe_ladybug,
    }
    missing_probe = sorted(required - set(probes))
    if missing_probe:
        print(f"[FAIL] no readiness probe for mandatory services: {missing_probe}")
        return 1

    pending = {name: probes[name] for name in required}
    errors: dict[str, str] = {}
    deadline = time.monotonic() + max(args.timeout, 0.1)
    while pending and time.monotonic() < deadline:
        for name, probe in list(pending.items()):
            try:
                detail = probe()
            except Exception as exc:  # noqa: BLE001 - retries preserve the last concrete error.
                errors[name] = f"{type(exc).__name__}: {exc}"
            else:
                print(f"[PASS] {name}: {detail}")
                pending.pop(name)
                errors.pop(name, None)
        if pending:
            time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
    for name in sorted(pending):
        print(f"[FAIL] {name}: {errors.get(name, 'timed out')}")
    return 1 if pending else 0


def _probe_ollama(base_url: str, lock: dict[str, Any], profile: str) -> str:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=3) as response:
        tags_payload = json.loads(response.read().decode("utf-8"))
    tags = {
        str(row.get("name") or row.get("model")): row
        for row in tags_payload.get("models", [])
        if row.get("name") or row.get("model")
    }
    selected = [
        row
        for row in lock.get("service_models", [])
        if profile in row.get("profiles", []) and row.get("service") == "ollama"
    ]
    if not selected:
        raise RuntimeError("no locked Ollama model for this profile")
    for row in selected:
        actual = _canonical_ollama_digest(
            tags.get(row["model"], {}).get("digest")
        )
        if actual != row.get("digest"):
            raise RuntimeError(f"{row['model']} digest mismatch: expected {row.get('digest')}, got {actual}")
        payload = json.dumps(
            {
                "model": row["model"],
                "stream": False,
                "format": "json",
                "messages": [{"role": "user", "content": row["smoke"]["prompt"]}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = str(body.get("message", {}).get("content", ""))
        if row["smoke"]["expect_contains"].lower() not in content.lower():
            raise RuntimeError(f"{row['model']} semantic smoke did not satisfy its contract")
    return f"{len(selected)} locked model digest(s) and semantic smoke(s) passed"


def _canonical_ollama_digest(value: object) -> str:
    """Normalize Ollama's raw content hash to the lock representation."""

    digest = str(value or "").lower()
    if len(digest) == 64 and all(character in "0123456789abcdef" for character in digest):
        return f"sha256:{digest}"
    return digest


def _probe_neo4j(runtime: LockedServiceRuntime | None = None) -> str:
    from neo4j import GraphDatabase

    resolved = runtime or LockedServiceRuntime.from_environment()
    driver = GraphDatabase.driver(
        resolved.neo4j_uri,
        auth=(resolved.neo4j_user, resolved.neo4j_password),
        connection_timeout=3,
    )
    try:
        driver.verify_connectivity()
        with driver.session(database=resolved.neo4j_database) as session:
            record = session.run("RETURN 1 AS ok").single()
    finally:
        driver.close()
    if record is None or record["ok"] != 1:
        raise RuntimeError("Neo4j semantic query returned the wrong value")
    return "authenticated query passed"


def _probe_memgraph(runtime: LockedServiceRuntime | None = None) -> str:
    from neo4j import GraphDatabase

    resolved = runtime or LockedServiceRuntime.from_environment()
    driver = GraphDatabase.driver(
        resolved.memgraph_uri,
        auth=None,
        connection_timeout=3,
    )
    try:
        with driver.session() as session:
            record = session.run("RETURN 1 AS ok").single()
    finally:
        driver.close()
    if record is None or record["ok"] != 1:
        raise RuntimeError("Memgraph semantic query returned the wrong value")
    return "Bolt query passed"


def _probe_ladybug() -> str:
    import ladybug

    with tempfile.TemporaryDirectory(prefix="kg-cert-ladybug-") as directory:
        database_path = str(Path(directory) / "doctor.lbdb")
        database = ladybug.Database(database_path)
        connection = ladybug.Connection(database)
        result: Any = connection.execute("RETURN 1 AS ok")
        if not result.has_next() or result.get_next()[0] != 1:
            raise RuntimeError("Ladybug semantic query returned the wrong value")
    return "embedded query passed"


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required file: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a YAML mapping")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
