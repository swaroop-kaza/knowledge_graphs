"""Resolve version-tagged certification services to a digest-pinned Compose lock."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SERVICES = _ROOT / "certification" / "services.yaml"
_LOCK = _ROOT / "certification" / "services.lock.yaml"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _resolve_digest(image: str) -> str:
    result = subprocess.run(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            image,
            "--format",
            "{{json .Manifest}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    manifest = json.loads(result.stdout)
    digest = str(manifest.get("digest", ""))
    if not _DIGEST.fullmatch(digest):
        raise SystemExit(f"could not resolve an immutable digest for {image}: {digest!r}")
    return digest


def build_lock() -> dict[str, Any]:
    source = yaml.safe_load(_SERVICES.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("services"), dict):
        raise SystemExit("certification/services.yaml must be a Docker Compose mapping")
    locked = copy.deepcopy(source)
    lock_metadata: dict[str, dict[str, str]] = {}
    for name, service in locked["services"].items():
        image = str(service.get("image", ""))
        if "@" in image or not re.search(r":[\w][\w.-]*$", image):
            raise SystemExit(f"service {name!r} source image must use one explicit version tag: {image!r}")
        repository = image.rsplit(":", 1)[0]
        digest = _resolve_digest(image)
        service["image"] = f"{repository}@{digest}"
        lock_metadata[name] = {"source_image": image, "digest": digest}
    locked["x-certification-lock"] = {
        "version": 2,
        "generated_by": "scripts/lock_service_digests.py",
        "source_sha256": _sha256_file(_SERVICES),
        "services": lock_metadata,
    }
    return locked


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    lock = build_lock()
    _atomic_write(
        _LOCK,
        "# Generated digest-pinned Docker Compose lock. Do not hand-edit.\n"
        + yaml.safe_dump(lock, sort_keys=False, width=100),
    )
    print(f"service-digests-ok: {_LOCK.name} ({len(lock['services'])} images pinned)")
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


if __name__ == "__main__":
    raise SystemExit(main())
