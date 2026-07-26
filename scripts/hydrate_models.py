"""Hydrate locked models once, or prove exact artifacts without registry access.

``--offline`` disables package/model registry access. Local Ollama remains a
mandatory service and is still queried for its locked model digest.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from kg_construction.certification.preflight import (
    check_model_lock,
    check_model_manifest,
    check_models_present,
)
from kg_construction.certification.profiles import select_profile
from kg_construction.service_runtime import DEFAULT_OLLAMA_BASE_URL

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "models.manifest.yaml"
_LOCK = _ROOT / "models.lock.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Must match the detected mandatory platform profile.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid package/model registry access; local Ollama is still required.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Hash and validate local artifacts without attempting hydration.",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not _LOCK.is_file():
        print("models-hydration-failed: models.lock.yaml is missing")
        return 1
    manifest = _mapping(_MANIFEST)
    lock = _mapping(_LOCK)
    try:
        profile = select_profile(manifest, args.profile).name
    except ValueError as exc:
        print(f"models-hydration-failed: {exc}")
        return 1

    manifest_check = check_model_manifest(_MANIFEST)
    print(f"[{'PASS' if manifest_check.passed else 'FAIL'}] {manifest_check.name}: {manifest_check.detail}")
    if not manifest_check.passed:
        return 1
    lock_check = check_model_lock(_MANIFEST, _LOCK, profile=profile)
    print(f"[{'PASS' if lock_check.passed else 'FAIL'}] {lock_check.name}: {lock_check.detail}")
    if not lock_check.passed:
        return 1

    offline = args.offline or args.validate_only
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not args.validate_only:
        if not _hydrate_snapshots(lock, profile=profile, offline=offline):
            return 1
        if not _hydrate_service_models(
            lock,
            profile=profile,
            base_url=args.ollama_url,
            offline=offline,
        ):
            return 1

    check = check_models_present(_LOCK, profile=profile)
    print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    service_failures = _validate_service_models(lock, profile=profile, base_url=args.ollama_url)
    if service_failures:
        print(f"[FAIL] service_models: {service_failures}")
        return 1
    print(f"[PASS] service_models: exact Ollama digest validated for {profile}")
    return 0 if check.passed else 1


def _hydrate_snapshots(lock: dict[str, Any], *, profile: str, offline: bool) -> bool:
    from huggingface_hub import snapshot_download

    seen: set[tuple[str, str]] = set()
    for row in lock.get("models", []):
        if profile not in row.get("profiles", []):
            continue
        key = (row["repo_id"], row["revision"])
        if key in seen:
            continue
        seen.add(key)
        try:
            path = snapshot_download(
                row["repo_id"],
                revision=row["revision"],
                local_files_only=offline,
            )
        except Exception as exc:  # noqa: BLE001 - bootstrap reports the exact artifact failure.
            print(f"models-hydration-failed: {key[0]}@{key[1]}: {type(exc).__name__}: {exc}")
            return False
        print(f"hydrated {key[0]}@{key[1]} -> {path}")
    return True


def _hydrate_service_models(
    lock: dict[str, Any],
    *,
    profile: str,
    base_url: str,
    offline: bool,
) -> bool:
    try:
        tags = _ollama_tags(base_url)
    except Exception as exc:  # noqa: BLE001 - bootstrap service failures are hard failures.
        print(f"models-hydration-failed: Ollama unavailable: {type(exc).__name__}: {exc}")
        return False
    for row in lock.get("service_models", []):
        if profile not in row.get("profiles", []):
            continue
        tag = row["model"]
        expected_digest = row["digest"]
        if tag in tags:
            actual_digest = tags[tag].get("digest")
            if actual_digest != expected_digest:
                print(
                    f"models-hydration-failed: locked Ollama model {tag!r} has "
                    f"digest {actual_digest!r}, expected {expected_digest!r}"
                )
                return False
            continue
        if offline:
            print(f"models-hydration-failed: locked Ollama model {tag!r} is absent offline")
            return False
        try:
            _pull_ollama(base_url, tag)
        except Exception as exc:  # noqa: BLE001 - explicit online bootstrap failure.
            print(f"models-hydration-failed: Ollama API pull {tag}: {type(exc).__name__}: {exc}")
            return False
        tags = _ollama_tags(base_url)
        actual_digest = tags.get(tag, {}).get("digest")
        if actual_digest != expected_digest:
            print(
                f"models-hydration-failed: registry tag drift for {tag!r}: "
                f"expected {expected_digest!r}, pulled {actual_digest!r}"
            )
            return False
    return True


def _validate_service_models(
    lock: dict[str, Any],
    *,
    profile: str,
    base_url: str,
) -> list[str]:
    try:
        tags = _ollama_tags(base_url)
    except Exception as exc:  # noqa: BLE001 - validation must turn service errors into hard failures.
        return [f"Ollama unavailable: {type(exc).__name__}: {exc}"]
    failures: list[str] = []
    for row in lock.get("service_models", []):
        if profile not in row.get("profiles", []):
            continue
        actual = tags.get(row["model"], {}).get("digest")
        if actual != row.get("digest"):
            failures.append(f"{row['model']}: expected {row.get('digest')}, got {actual}")
    return failures


def _ollama_tags(base_url: str) -> dict[str, dict[str, Any]]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models", []), list):
        raise RuntimeError("Ollama /api/tags returned an invalid payload")
    return {
        str(row.get("name") or row.get("model")): {
            **row,
            "digest": _canonical_ollama_digest(row.get("digest")),
        }
        for row in payload.get("models", [])
        if isinstance(row, dict) and (row.get("name") or row.get("model"))
    }


def _canonical_ollama_digest(value: object) -> str:
    """Normalize Ollama's raw content hash to the lock's algorithm-qualified form."""

    digest = str(value or "").lower()
    if len(digest) == 64 and all(character in "0123456789abcdef" for character in digest):
        return f"sha256:{digest}"
    return digest


def _pull_ollama(base_url: str, tag: str) -> None:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/pull",
        data=json.dumps({"model": tag, "stream": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "success":
        raise RuntimeError(f"unexpected pull response: {payload}")


def _mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a YAML mapping")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
