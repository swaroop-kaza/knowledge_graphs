"""Platform-mandatory certification profile selection and hardware checks."""

from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlatformProfile:
    name: str
    system: str
    machine: str
    accelerator: str
    mandatory_backends: tuple[str, ...]
    mandatory_services: tuple[str, ...]


def select_profile(manifest: dict[str, Any], requested: str | None = None) -> PlatformProfile:
    """Select exactly one profile and reject hardware/profile fallback."""

    system = platform.system()
    machine = platform.machine().lower()
    cuda = _cuda_available()
    if system == "Darwin" and machine == "arm64":
        detected = "darwin-arm64"
    elif system == "Linux" and machine in {"x86_64", "amd64"}:
        detected = "linux-cuda" if cuda else "linux-cpu"
    else:
        raise ValueError(f"unsupported certification platform: {system}/{machine}")
    if requested and requested != detected:
        raise ValueError(
            f"requested profile {requested!r} does not match detected mandatory profile {detected!r}"
        )

    rows = manifest.get("profiles", [])
    matches = [row for row in rows if isinstance(row, dict) and row.get("name") == detected]
    if len(matches) != 1:
        raise ValueError(f"profile {detected!r} must appear exactly once in models.manifest.yaml")
    row = matches[0]
    selector = row.get("selector", {})
    expected_system = selector.get("system")
    expected_machine = str(selector.get("machine", "")).lower()
    if expected_system != system or expected_machine not in {machine, "x86_64" if machine == "amd64" else machine}:
        raise ValueError(f"profile selector for {detected!r} does not match this host")
    return PlatformProfile(
        name=detected,
        system=system,
        machine=machine,
        accelerator=str(selector.get("accelerator")),
        mandatory_backends=tuple(row.get("mandatory_backends", [])),
        mandatory_services=tuple(row.get("mandatory_services", [])),
    )


def hardware_violations(profile: PlatformProfile) -> list[str]:
    """Return missing mandatory runtime/hardware requirements."""

    problems: list[str] = []
    if profile.name == "darwin-arm64":
        if importlib.util.find_spec("mlx") is None:
            problems.append("mlx is not installed")
        if importlib.util.find_spec("mlx_lm") is None:
            problems.append("mlx-lm is not installed")
        try:
            import torch

            if not torch.backends.mps.is_built():
                problems.append("Torch was not built with MPS")
            elif not torch.backends.mps.is_available():
                problems.append("MPS is not available")
        except ImportError:
            problems.append("torch is not installed")
    elif profile.name == "linux-cuda" and not _cuda_available():
        problems.append("CUDA is not available")
    elif profile.name == "linux-cpu" and _cuda_available():
        problems.append("CUDA is available, so linux-cuda is the mandatory profile")
    return problems


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


__all__ = ["PlatformProfile", "hardware_violations", "select_profile"]
