"""Resolve every mandatory model to immutable revisions and byte checksums.

This is an explicit online release-provisioning command. Certification itself
never invokes it. Every Hugging Face snapshot across all platform profiles is
resolved, hydrated, and hashed. Every declared Ollama model is pulled when
absent and locked by its local content digest. Partial/profile-specific release
locks are deliberately unsupported.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from kg_construction.certification.preflight import check_model_manifest
from kg_construction.service_runtime import DEFAULT_OLLAMA_BASE_URL

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "models.manifest.yaml"
_LOCK = _ROOT / "models.lock.yaml"
_DIGEST_PREFIX = "sha256:"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
    )
    return parser


def build_lock(*, ollama_url: str = DEFAULT_OLLAMA_BASE_URL) -> dict[str, Any]:
    manifest_check = check_model_manifest(_MANIFEST)
    if not manifest_check.passed:
        raise SystemExit(f"invalid models.manifest.yaml: {manifest_check.detail}")
    manifest = _mapping(_MANIFEST)
    declared_profiles = [row["name"] for row in manifest.get("profiles", [])]

    resolved_snapshots: dict[tuple[str, str, str], tuple[str, Path, str]] = {}
    locked_models: list[dict[str, Any]] = []
    for model in manifest.get("models", []):
        selected = [
            profile
            for profile in declared_profiles
            if profile in model.get("profiles", [])
        ]
        key = (
            model["repo_id"],
            model["ref"],
            str(model["license"]).lower(),
        )
        if key not in resolved_snapshots:
            resolved_snapshots[key] = _resolve_snapshot(
                model["repo_id"],
                model["ref"],
                expected_license=model["license"],
            )
        revision, snapshot, card_license = resolved_snapshots[key]
        files = _hash_snapshot(snapshot)
        for profile in selected:
            locked_models.append(
                {
                    "name": model["name"],
                    "repo_id": model["repo_id"],
                    "ref": model["ref"],
                    "revision": revision,
                    "backend": model["backend"],
                    "task": model["task"],
                    "profiles": [profile],
                    "device": model["device_by_profile"][profile],
                    "accelerator_policy": model.get("accelerator_policy", "native"),
                    "license": model["license"],
                    "model_card_license": card_license,
                    "smoke": copy.deepcopy(model["smoke"]),
                    "files": copy.deepcopy(files),
                }
            )

    ollama_tags = _ollama_tags(ollama_url)
    service_models: list[dict[str, Any]] = []
    for model in manifest.get("service_models", []):
        if model.get("service") != "ollama":
            raise SystemExit(
                f"unsupported mandatory service-model backend: {model.get('service')!r}"
            )
        tag = model["model"]
        record = ollama_tags.get(tag)
        if record is None:
            print(f"pulling mandatory Ollama model {tag}")
            _pull_ollama(ollama_url, tag)
            ollama_tags = _ollama_tags(ollama_url)
            record = ollama_tags.get(tag)
        digest = str((record or {}).get("digest", ""))
        size = (record or {}).get("size")
        if (
            not digest.startswith(_DIGEST_PREFIX)
            or len(digest) != len(_DIGEST_PREFIX) + 64
            or any(character not in "0123456789abcdef" for character in digest[7:])
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise SystemExit(
                f"Ollama model {tag!r} is not present with a valid digest and size"
            )
        service_models.append(
            {
                **model,
                "profiles": list(model["profiles"]),
                "digest": digest,
                "size": size,
            }
        )

    unique_snapshots = {
        (row["repo_id"], row["revision"]): sum(item["size"] for item in row["files"])
        for row in locked_models
    }
    artifact_totals = {
        "unique_huggingface_snapshots": len(unique_snapshots),
        "unique_huggingface_bytes": sum(unique_snapshots.values()),
        "ollama_models": len(service_models),
        "ollama_bytes": sum(row["size"] for row in service_models),
    }
    return {
        "version": 3,
        "generated_by": "scripts/lock_models.py",
        "manifest_sha256": _sha256_file(_MANIFEST),
        "profiles": declared_profiles,
        "artifact_totals": artifact_totals,
        "models": locked_models,
        "service_models": service_models,
    }


def _resolve_snapshot(
    repo_id: str,
    ref: str,
    *,
    expected_license: str,
) -> tuple[str, Path, str]:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(repo_id, revision=ref)
    revision = str(info.sha)
    if len(revision) < 40:
        raise SystemExit(f"{repo_id}@{ref} did not resolve to an immutable revision")
    card_license = ""
    if info.card_data is not None and hasattr(info.card_data, "get"):
        value = info.card_data.get("license")
        card_license = str(value or "").lower()
    if card_license != expected_license.lower():
        raise SystemExit(
            f"license mismatch for {repo_id}: manifest={expected_license!r}, "
            f"model-card={card_license!r}"
        )
    path = Path(snapshot_download(repo_id, revision=revision))
    return revision, path, card_license


def _hash_snapshot(snapshot: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file() or item.is_symlink()):
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        rows.append(
            {
                "path": str(path.relative_to(snapshot)),
                "sha256": digest.hexdigest(),
                "size": size,
            }
        )
    if not rows:
        raise SystemExit(f"resolved model snapshot is empty: {snapshot}")
    return rows


def _ollama_tags(base_url: str) -> dict[str, dict[str, Any]]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models", []), list):
        raise SystemExit("Ollama /api/tags returned an invalid payload")
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("models", []):
        if not isinstance(row, dict):
            continue
        name = row.get("name") or row.get("model")
        if name:
            normalized = dict(row)
            normalized["digest"] = _canonical_ollama_digest(row.get("digest"))
            rows[str(name)] = normalized
    return rows


def _canonical_ollama_digest(value: object) -> str:
    """Normalize Ollama's raw content hash to the lock's algorithm-qualified form."""

    digest = str(value or "").lower()
    if len(digest) == 64 and all(character in "0123456789abcdef" for character in digest):
        return f"{_DIGEST_PREFIX}{digest}"
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
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise SystemExit(f"Ollama failed to pull mandatory model {tag!r}: {payload!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a YAML mapping")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lock = build_lock(ollama_url=args.ollama_url)
    body = yaml.safe_dump(lock, sort_keys=False, width=100)
    _atomic_write(
        _LOCK,
        "# Generated immutable model lock. Do not hand-edit.\n" + body,
    )
    print(
        f"models-lock-ok: {_LOCK.name} "
        f"({len(lock['models'])} profile artifacts, "
        f"{lock['artifact_totals']['unique_huggingface_snapshots']} unique snapshots, "
        f"{_gib(lock['artifact_totals']['unique_huggingface_bytes'])} GiB Hugging Face, "
        f"{_gib(lock['artifact_totals']['ollama_bytes'])} GiB Ollama)"
    )
    return 0


def _atomic_write(path: Path, body: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(body)
        temporary = Path(handle.name)
    try:
        temporary.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _gib(size: int) -> str:
    return f"{size / (1024**3):.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
