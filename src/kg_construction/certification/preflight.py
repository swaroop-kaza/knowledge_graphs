"""Hard-fail preflight for the reproducible locked-artifact certification run."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from kg_construction.certification.execution import validate_execution_report
from kg_construction.certification.profiles import hardware_violations, select_profile

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REQUIRED_LOCKS = ("uv.lock", "models.lock.yaml", "certification/services.lock.yaml")
_SOFTENERS = ("|| true", "--skip-services", "--skip-", "|| echo", "set +e")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_CERTIFICATION_PROFILES = frozenset(
    {"darwin-arm64", "linux-cpu", "linux-cuda"}
)
_PROFILE_SELECTORS = {
    "darwin-arm64": ("Darwin", "arm64", "mps"),
    "linux-cpu": ("Linux", "x86_64", "cpu"),
    "linux-cuda": ("Linux", "x86_64", "cuda"),
}
_COMMON_MODEL_INVENTORY = {
    "gliner_entity_extractor": ("gliner", "token-classification"),
    "gliner_relex_extractor": ("gliner", "relation-extraction"),
    "structured_transformer_extractor": ("transformers", "text-generation"),
    "transformer_ner": ("transformers", "token-classification"),
    "embedding_resolver": ("sentence-transformers", "feature-extraction"),
    "cross_encoder_resolver": ("sentence-transformers", "text-ranking"),
    "spacy_pipeline": ("spacy", "sentence-pos-ner"),
}
_DARWIN_MODEL_INVENTORY = {
    "mlx_structured_extractor": ("mlx", "text-generation"),
}
_COMMON_SERVICES = {"ollama", "neo4j", "memgraph", "ladybug"}
Phase = Literal["preflight", "final"]


@dataclass(frozen=True)
class CertCheck:
    name: str
    passed: bool
    detail: str


def check_no_product_optional_extras(pyproject_path: Path) -> CertCheck:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    extras = data.get("project", {}).get("optional-dependencies", {})
    if extras:
        return CertCheck("no_product_optional_extras", False, f"optional extras still present: {sorted(extras)}")
    return CertCheck("no_product_optional_extras", True, "all product dependencies are mandatory")


def check_capability_manifest(manifest_path: Path) -> CertCheck:
    manifest = _load_mapping(manifest_path)
    rows = manifest.get("techniques", [])
    if manifest.get("doctrine") != "mandatory_executable_release" or not isinstance(rows, list) or not rows:
        return CertCheck("capability_manifest", False, "manifest doctrine/technique list is invalid")
    forbidden: list[str] = []
    unimportable: list[str] = []
    invalid_profiles: list[str] = []
    names: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            forbidden.append("<non-mapping>")
            continue
        name = str(row.get("name", ""))
        names.append(name)
        status = str(row.get("status", "")).lower()
        disposition = str(row.get("disposition", "")).lower()
        if status in {"stub", "planned", "optional"} or disposition != "mandatory":
            forbidden.append(name)
        if not row.get("owner"):
            forbidden.append(f"{name}:missing-owner")
        else:
            owner = str(row["owner"])
            try:
                module_name, attribute = owner.rsplit(".", 1)
                module = importlib.import_module(module_name)
                implementation = getattr(module, attribute)
                if not callable(implementation):
                    raise TypeError("owner is not callable")
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                unimportable.append(f"{name}:{type(exc).__name__}")
        if "profiles" in row:
            profiles = row.get("profiles")
            if (
                not isinstance(profiles, list)
                or not profiles
                or not all(isinstance(item, str) and item for item in profiles)
                or len(profiles) != len(set(profiles))
                or not set(profiles).issubset(_CERTIFICATION_PROFILES)
            ):
                invalid_profiles.append(name)
    if forbidden:
        return CertCheck(
            "capability_manifest",
            False,
            f"non-mandatory/non-executable rows: {sorted(forbidden)}",
        )
    if "" in names or len(names) != len(set(names)):
        return CertCheck("capability_manifest", False, "capability names are empty or duplicated")
    if unimportable:
        return CertCheck(
            "capability_manifest",
            False,
            f"implementation owners are not importable/callable: {sorted(unimportable)}",
        )
    if invalid_profiles:
        return CertCheck(
            "capability_manifest",
            False,
            "invalid capability profile applicability: "
            f"{sorted(invalid_profiles)}",
        )
    summary = manifest.get("summary", {}).get("techniques", {})
    if summary != {"total": len(rows), "mandatory": len(rows)}:
        return CertCheck("capability_manifest", False, "technique summary is not total=mandatory")
    return CertCheck("capability_manifest", True, f"{len(rows)} executable capabilities are mandatory")


def check_lockfiles_present(root: Path) -> CertCheck:
    missing = [name for name in _REQUIRED_LOCKS if not (root / name).is_file()]
    if missing:
        return CertCheck("lockfiles_present", False, f"missing locks: {missing}")
    return CertCheck("lockfiles_present", True, "dependency, model, and service locks are present")


def check_model_manifest(models_manifest_path: Path) -> CertCheck:
    """Validate the complete, non-optional platform/model inventory."""

    manifest = _load_mapping(models_manifest_path)
    violations: list[str] = []
    if manifest.get("version") != 2:
        violations.append("manifest version must be 2")

    profile_rows = manifest.get("profiles")
    if not isinstance(profile_rows, list):
        profile_rows = []
        violations.append("profiles must be a list")
    profiles = {
        str(row.get("name")): row
        for row in profile_rows
        if isinstance(row, dict) and row.get("name")
    }
    profile_names = [str(row.get("name", "")) for row in profile_rows if isinstance(row, dict)]
    if (
        set(profiles) != _CERTIFICATION_PROFILES
        or len(profile_names) != len(set(profile_names))
        or "" in profile_names
    ):
        violations.append(
            "profiles must contain darwin-arm64, linux-cpu, and linux-cuda exactly once"
        )
    for name, expected in _PROFILE_SELECTORS.items():
        row = profiles.get(name, {})
        selector = row.get("selector", {}) if isinstance(row, dict) else {}
        actual = (
            selector.get("system"),
            str(selector.get("machine", "")).lower(),
            selector.get("accelerator"),
        )
        if actual != expected:
            violations.append(f"{name}: selector mismatch")
        backends = row.get("mandatory_backends", []) if isinstance(row, dict) else []
        services = row.get("mandatory_services", []) if isinstance(row, dict) else []
        if not _unique_nonempty_strings(backends):
            violations.append(f"{name}: mandatory_backends must be non-empty and unique")
        if (
            not _unique_nonempty_strings(services)
            or set(services) != _COMMON_SERVICES
        ):
            violations.append(f"{name}: mandatory_services must equal {sorted(_COMMON_SERVICES)}")

    model_rows = manifest.get("models")
    if not isinstance(model_rows, list):
        model_rows = []
        violations.append("models must be a list")
    models: dict[str, dict[str, Any]] = {}
    model_names: list[str] = []
    for raw in model_rows:
        if not isinstance(raw, dict):
            violations.append("model row must be a mapping")
            continue
        name = str(raw.get("name", ""))
        model_names.append(name)
        if name:
            models[name] = raw
        selected_profiles = raw.get("profiles", [])
        device_by_profile = raw.get("device_by_profile", {})
        if (
            not name
            or not raw.get("repo_id")
            or not raw.get("ref")
            or not raw.get("backend")
            or not raw.get("task")
            or not raw.get("license")
            or not isinstance(raw.get("smoke"), dict)
            or not raw.get("smoke")
        ):
            violations.append(f"{name or '<unnamed>'}: incomplete model contract")
        if (
            not _unique_nonempty_strings(selected_profiles)
            or not set(selected_profiles).issubset(_CERTIFICATION_PROFILES)
        ):
            violations.append(f"{name or '<unnamed>'}: invalid model profiles")
        if (
            not isinstance(device_by_profile, dict)
            or set(device_by_profile) != set(selected_profiles)
            or not all(isinstance(value, str) and value for value in device_by_profile.values())
        ):
            violations.append(f"{name or '<unnamed>'}: device_by_profile must cover profiles exactly")
        if raw.get("accelerator_policy", "native") not in {"native", "cpu_only"}:
            violations.append(f"{name or '<unnamed>'}: invalid accelerator_policy")
    if "" in model_names or len(model_names) != len(set(model_names)):
        violations.append("model names must be non-empty and unique")

    for profile in sorted(_CERTIFICATION_PROFILES):
        expected_inventory = dict(_COMMON_MODEL_INVENTORY)
        if profile == "darwin-arm64":
            expected_inventory.update(_DARWIN_MODEL_INVENTORY)
        selected_names = {
            name for name, row in models.items() if profile in row.get("profiles", [])
        }
        missing = sorted(set(expected_inventory) - selected_names)
        if missing:
            violations.append(f"{profile}: missing mandatory models {missing}")
        for name, (backend, task) in expected_inventory.items():
            model_row = models.get(name)
            if model_row is not None and (
                model_row.get("backend") != backend or model_row.get("task") != task
            ):
                violations.append(f"{name}: expected backend/task {backend}/{task}")
    mlx_profiles = set(models.get("mlx_structured_extractor", {}).get("profiles", []))
    if mlx_profiles != {"darwin-arm64"}:
        violations.append("mlx_structured_extractor must be mandatory exactly on darwin-arm64")

    service_rows = manifest.get("service_models")
    if not isinstance(service_rows, list):
        service_rows = []
        violations.append("service_models must be a list")
    service_names: list[str] = []
    service_models: dict[str, dict[str, Any]] = {}
    for raw in service_rows:
        if not isinstance(raw, dict):
            violations.append("service model row must be a mapping")
            continue
        name = str(raw.get("name", ""))
        service_names.append(name)
        if name:
            service_models[name] = raw
        selected_profiles = raw.get("profiles", [])
        if (
            not name
            or raw.get("service") != "ollama"
            or not raw.get("model")
            or not raw.get("license")
            or not isinstance(raw.get("smoke"), dict)
            or not raw.get("smoke")
            or not _unique_nonempty_strings(selected_profiles)
            or not set(selected_profiles).issubset(_CERTIFICATION_PROFILES)
        ):
            violations.append(f"{name or '<unnamed>'}: incomplete service-model contract")
    if "" in service_names or len(service_names) != len(set(service_names)):
        violations.append("service-model names must be non-empty and unique")
    ollama = service_models.get("ollama_structured_extractor", {})
    if set(ollama.get("profiles", [])) != _CERTIFICATION_PROFILES:
        violations.append("ollama_structured_extractor must be mandatory on every profile")
    for name in sorted(_CERTIFICATION_PROFILES):
        device_check = check_no_cpu_fallback(model_rows, profile=name)
        if not device_check.passed:
            violations.append(device_check.detail)

    if violations:
        return CertCheck("model_manifest", False, "; ".join(dict.fromkeys(violations)))
    return CertCheck(
        "model_manifest",
        True,
        "GLiNER, GLiNER RelEx, Transformers, spaCy, embeddings, Ollama, and Darwin MLX are mandatory",
    )


def check_makefile_certification_is_strict(makefile_path: Path, *, target: str = "certify") -> CertCheck:
    recipe = _makefile_recipe(makefile_path, target)
    if recipe is None:
        return CertCheck("certification_is_strict", False, f"no {target!r} target found")
    found = [token for token in _SOFTENERS if token in recipe]
    if found:
        return CertCheck("certification_is_strict", False, f"{target} softens with: {found}")
    return CertCheck("certification_is_strict", True, f"{target} has no skip/error softeners")


def check_execution_plan(
    manifest_path: Path,
    plan_path: Path,
    *,
    profile: str | None,
) -> CertCheck:
    manifest = _load_mapping(manifest_path)
    plan = _load_mapping(plan_path)
    declared = {
        str(row.get("name"))
        for row in manifest.get("techniques", [])
        if isinstance(row, dict) and row.get("name")
    }
    contracts = plan.get("capabilities", {})
    if not isinstance(contracts, dict):
        return CertCheck("execution_plan", False, "capabilities must be a mapping")
    configured = set(contracts)
    if configured != declared:
        return CertCheck(
            "execution_plan",
            False,
            f"certifier coverage mismatch: missing={sorted(declared - configured)}, "
            f"unexpected={sorted(configured - declared)}",
        )
    invalid: list[str] = []
    profile_mismatches: list[str] = []
    manifest_rows = {
        str(row.get("name")): row
        for row in manifest.get("techniques", [])
        if isinstance(row, dict) and row.get("name")
    }
    for name, contract in contracts.items():
        if not isinstance(contract, dict):
            invalid.append(name)
            continue
        command = contract.get("command")
        profiles = contract.get("profiles")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
            or not isinstance(profiles, list)
            or not profiles
            or not all(isinstance(item, str) and item for item in profiles)
            or len(profiles) != len(set(profiles))
            or not set(profiles).issubset(_CERTIFICATION_PROFILES)
        ):
            invalid.append(name)
            continue
        manifest_row = manifest_rows[name]
        declared_profiles = manifest_row.get("profiles")
        if isinstance(declared_profiles, list) and set(profiles) != set(
            declared_profiles
        ):
            profile_mismatches.append(name)
            continue
        if (
            profile is not None
            and _capability_applies(manifest_row, profile)
            and profile not in profiles
        ):
            profile_mismatches.append(name)
    if invalid:
        return CertCheck("execution_plan", False, f"invalid certifier contracts: {sorted(invalid)}")
    if profile_mismatches:
        return CertCheck(
            "execution_plan",
            False,
            "certifier profile applicability mismatch: "
            f"{sorted(profile_mismatches)}",
        )
    return CertCheck("execution_plan", True, f"{len(declared)} capabilities have executable certifiers")


def _capability_applies(row: dict[str, Any], profile: str) -> bool:
    """Treat legacy rows without ``profiles`` as compatible everywhere."""

    profiles = row.get("profiles")
    return profiles is None or (
        isinstance(profiles, list) and profile in profiles
    )


def check_profile_contract(models_manifest_path: Path, *, requested: str | None = None) -> tuple[CertCheck, str | None]:
    manifest = _load_mapping(models_manifest_path)
    try:
        profile = select_profile(manifest, requested)
    except ValueError as exc:
        return CertCheck("platform_profile", False, str(exc)), None
    problems = hardware_violations(profile)
    if problems:
        return CertCheck("platform_profile", False, f"{profile.name}: {problems}"), profile.name
    return (
        CertCheck(
            "platform_profile",
            True,
            f"{profile.name}: backends={list(profile.mandatory_backends)}, services={list(profile.mandatory_services)}",
        ),
        profile.name,
    )


def check_no_cpu_fallback(models: list[dict[str, Any]], *, profile: str | None = None) -> CertCheck:
    """Ensure selected devices match the profile and its accelerator."""

    expected = {
        "darwin-arm64": {"mps", "mlx", "cpu"},
        "linux-cpu": {"cpu"},
        "linux-cuda": {"cuda", "cuda:0", "cpu"},
    }
    if profile is None:
        # Backwards-compatible unit-level check for already-resolved lock rows.
        for model in models:
            device = str(model.get("device", "cpu"))
            if device.startswith("cuda") and not _cuda_available():
                return CertCheck("no_device_fallback", False, f"{model.get('name')} requires {device}")
        return CertCheck("no_device_fallback", True, "resolved model devices are available")
    allowed = expected.get(profile)
    if allowed is None:
        return CertCheck("no_device_fallback", False, f"unknown profile {profile!r}")
    violations: list[str] = []
    for model in models:
        if profile not in model.get("profiles", []):
            continue
        device = str(model.get("device_by_profile", {}).get(profile, ""))
        if not device or device not in allowed:
            violations.append(f"{model.get('name')}={device or '<missing>'}")
        if (
            profile in {"darwin-arm64", "linux-cuda"}
            and device == "cpu"
            and model.get("accelerator_policy") != "cpu_only"
        ):
            violations.append(f"{model.get('name')} silently selects CPU")
    if violations:
        return CertCheck("no_device_fallback", False, f"device violations: {violations}")
    return CertCheck("no_device_fallback", True, f"all {profile} model devices are explicit")


def check_model_lock(
    models_manifest_path: Path,
    models_lock_path: Path,
    *,
    profile: str,
) -> CertCheck:
    manifest = _load_mapping(models_manifest_path)
    lock = _load_mapping(models_lock_path)
    declared_profiles = _declared_model_profiles(manifest)
    lock_profiles = lock.get("profiles", [])
    if (
        lock.get("version") != 3
        or lock.get("generated_by") != "scripts/lock_models.py"
        or lock.get("manifest_sha256") != _file_sha256(models_manifest_path)
        or not _unique_nonempty_strings(lock_profiles)
        or set(lock_profiles) != declared_profiles
        or profile not in declared_profiles
    ):
        return CertCheck(
            "model_lock",
            False,
            f"model lock version/source/profile coverage is invalid for {profile}",
        )

    manifest_models = [
        row for row in manifest.get("models", []) if isinstance(row, dict)
    ]
    locked_rows = [row for row in lock.get("models", []) if isinstance(row, dict)]
    violations: list[str] = []
    locked_keys: list[tuple[str, str]] = []
    for row in locked_rows:
        row_profiles = row.get("profiles", [])
        if not isinstance(row_profiles, list) or len(row_profiles) != 1:
            violations.append(f"{row.get('name')}: file-backed lock row must select one profile")
            continue
        locked_keys.append((str(row.get("name", "")), str(row_profiles[0])))
    if len(locked_keys) != len(set(locked_keys)):
        violations.append("duplicate file-backed model/profile lock rows")

    for selected_profile in sorted(declared_profiles):
        selected = {
            str(row.get("name")): row
            for row in manifest_models
            if row.get("name") and selected_profile in row.get("profiles", [])
        }
        matching = [
            row for row in locked_rows if selected_profile in row.get("profiles", [])
        ]
        locked = {
            str(row.get("name")): row for row in matching if row.get("name")
        }
        if len(locked) != len(matching):
            violations.append(f"{selected_profile}: duplicated or unnamed model lock row")
        if set(locked) != set(selected):
            violations.append(
                f"{selected_profile}: locked model set mismatch "
                f"(missing={sorted(set(selected) - set(locked))}, "
                f"unexpected={sorted(set(locked) - set(selected))})"
            )
            continue
        for name, source in selected.items():
            row = locked[name]
            for field in ("repo_id", "ref", "backend", "task", "license", "smoke"):
                if row.get(field) != source.get(field):
                    violations.append(f"{name}/{selected_profile}: {field} mismatch")
            expected_policy = source.get("accelerator_policy", "native")
            if row.get("accelerator_policy") != expected_policy:
                violations.append(f"{name}/{selected_profile}: accelerator_policy mismatch")
            if row.get("model_card_license") != str(source.get("license", "")).lower():
                violations.append(f"{name}/{selected_profile}: model-card license mismatch")
            if not _REVISION.fullmatch(str(row.get("revision", ""))):
                violations.append(f"{name}/{selected_profile}: immutable revision missing")
            if row.get("device") != source.get("device_by_profile", {}).get(selected_profile):
                violations.append(f"{name}/{selected_profile}: device mismatch")
            artifact_problem = _artifact_lock_problem(row.get("files"))
            if artifact_problem:
                violations.append(f"{name}/{selected_profile}: {artifact_problem}")

    manifest_service_rows = [
        row for row in manifest.get("service_models", []) if isinstance(row, dict)
    ]
    locked_service_rows = [
        row for row in lock.get("service_models", []) if isinstance(row, dict)
    ]
    service_names = [str(row.get("name", "")) for row in locked_service_rows]
    if "" in service_names or len(service_names) != len(set(service_names)):
        violations.append("service-model lock names must be non-empty and unique")
    locked_service_by_name = {
        str(row.get("name")): row for row in locked_service_rows if row.get("name")
    }
    manifest_service_by_name = {
        str(row.get("name")): row for row in manifest_service_rows if row.get("name")
    }
    if set(locked_service_by_name) != set(manifest_service_by_name):
        violations.append("service-model set mismatch")
    for name, source in manifest_service_by_name.items():
        locked_service = locked_service_by_name.get(name)
        if locked_service is None:
            continue
        for field in ("service", "model", "profiles", "license", "smoke"):
            if locked_service.get(field) != source.get(field):
                violations.append(f"{name}: {field} mismatch")
        if not _DIGEST.fullmatch(str(locked_service.get("digest", ""))):
            violations.append(f"{name}: service-model digest missing")
        size = locked_service.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            violations.append(f"{name}: service-model size must be positive")

    unique_snapshots: dict[tuple[str, str], tuple[list[dict[str, Any]], int]] = {}
    for row in locked_rows:
        files = row.get("files")
        if _artifact_lock_problem(files) or not isinstance(files, list):
            continue
        key = (str(row.get("repo_id", "")), str(row.get("revision", "")))
        size = sum(int(item["size"]) for item in files)
        previous = unique_snapshots.get(key)
        if previous is not None and previous[0] != files:
            violations.append(f"{key[0]}@{key[1]}: inconsistent files across profiles")
        unique_snapshots[key] = (files, size)
    expected_totals = {
        "unique_huggingface_snapshots": len(unique_snapshots),
        "unique_huggingface_bytes": sum(item[1] for item in unique_snapshots.values()),
        "ollama_models": len(locked_service_rows),
        "ollama_bytes": sum(
            int(row["size"])
            for row in locked_service_rows
            if isinstance(row.get("size"), int) and not isinstance(row.get("size"), bool)
        ),
    }
    if lock.get("artifact_totals") != expected_totals:
        violations.append("artifact_totals mismatch")

    if violations:
        return CertCheck("model_lock", False, "; ".join(dict.fromkeys(violations)))
    selected_count = sum(
        1 for row in manifest_models if profile in row.get("profiles", [])
    )
    selected_service_count = sum(
        1 for row in manifest_service_rows if profile in row.get("profiles", [])
    )
    return CertCheck(
        "model_lock",
        True,
        f"all profiles are locked; {selected_count} file-backed and "
        f"{selected_service_count} service models apply to {profile}",
    )


def check_models_present(
    models_lock_path: Path,
    *,
    profile: str | None = None,
    cache_root: Path | None = None,
) -> CertCheck:
    lock = _load_mapping(models_lock_path)
    rows = [
        row
        for row in lock.get("models", [])
        if isinstance(row, dict) and (profile is None or profile in row.get("profiles", []))
    ]
    failures: list[str] = []
    for row in rows:
        name = str(row.get("name", "<unnamed>"))
        repo_id = row.get("repo_id")
        revision = row.get("revision")
        artifact_problem = _artifact_lock_problem(row.get("files"))
        if (
            not isinstance(repo_id, str)
            or not repo_id
            or not _REVISION.fullmatch(str(revision or ""))
            or artifact_problem
        ):
            failures.append(f"{name}: malformed lock row")
            continue
        snapshot = _snapshot_path(repo_id, str(revision), cache_root=cache_root)
        if not snapshot.is_dir():
            failures.append(f"{name}: snapshot missing")
            continue
        expected = {item["path"]: item for item in row.get("files", [])}
        actual = {
            str(path.relative_to(snapshot))
            for path in snapshot.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if actual != set(expected):
            failures.append(
                f"{name}: artifact set mismatch "
                f"(missing={sorted(set(expected) - actual)}, unexpected={sorted(actual - set(expected))})"
            )
            continue
        for relative, artifact in expected.items():
            path = snapshot / relative
            try:
                digest, size = _sha256(path)
            except OSError as exc:
                failures.append(f"{name}:{relative}: {exc}")
                break
            if digest != artifact["sha256"] or size != artifact["size"]:
                failures.append(f"{name}:{relative}: checksum/size mismatch")
                break
    if failures:
        return CertCheck("models_hydrated", False, "; ".join(failures))
    if not rows:
        return CertCheck("models_hydrated", False, "no locked models selected")
    return CertCheck("models_hydrated", True, f"{len(rows)} model snapshots match every locked byte")


def check_service_lock(services_path: Path, lock_path: Path) -> CertCheck:
    source = _load_mapping(services_path)
    lock = _load_mapping(lock_path)
    source_services = source.get("services", {})
    locked_services = lock.get("services", {})
    if not isinstance(source_services, dict) or not isinstance(locked_services, dict):
        return CertCheck("service_lock", False, "service source/lock must be Compose mappings")
    metadata = lock.get("x-certification-lock", {})
    metadata_services = metadata.get("services", {}) if isinstance(metadata, dict) else {}
    if (
        metadata.get("version") != 2
        or metadata.get("generated_by") != "scripts/lock_service_digests.py"
        or metadata.get("source_sha256") != _file_sha256(services_path)
        or not isinstance(metadata_services, dict)
    ):
        return CertCheck("service_lock", False, "service lock metadata is missing or invalid")
    if set(source_services) != set(locked_services):
        return CertCheck("service_lock", False, "service names differ between source and lock")
    violations: list[str] = []
    for name, service in locked_services.items():
        if not isinstance(service, dict):
            violations.append(f"{name}: invalid service mapping")
            continue
        image = str(service.get("image", ""))
        if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image):
            violations.append(f"{name}: image is not digest-pinned")
        source_image = str(source_services[name].get("image", ""))
        if image.split("@", 1)[0] != source_image.rsplit(":", 1)[0]:
            violations.append(f"{name}: image repository mismatch")
        metadata_row = metadata_services.get(name, {})
        if (
            not isinstance(metadata_row, dict)
            or metadata_row.get("source_image") != source_image
            or image.rsplit("@", 1)[-1] != metadata_row.get("digest")
        ):
            violations.append(f"{name}: lock metadata mismatch")
    if violations:
        return CertCheck("service_lock", False, "; ".join(violations))
    return CertCheck("service_lock", True, f"{len(locked_services)} service images are digest-pinned")


def check_local_service_images(lock_path: Path, *, profile: str | None = None) -> CertCheck:
    lock = _load_mapping(lock_path)
    images = [
        str(row.get("image"))
        for row in lock.get("services", {}).values()
        if isinstance(row, dict) and row.get("image")
    ]
    if not images:
        return CertCheck("service_images_local", False, "no digest-pinned service images found")
    missing: list[str] = []
    wrong_platform: list[str] = []
    expected_platform = {
        "darwin-arm64": "linux/arm64",
        "linux-cpu": "linux/amd64",
        "linux-cuda": "linux/amd64",
    }.get(profile or "")
    for image in images:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    image,
                    "--format",
                    "{{.Os}}/{{.Architecture}}",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            missing.append(image)
        else:
            actual_platform = result.stdout.strip()
            if expected_platform and actual_platform != expected_platform:
                wrong_platform.append(f"{image}={actual_platform}")
    if missing:
        return CertCheck("service_images_local", False, f"locked images absent locally: {missing}")
    if wrong_platform:
        return CertCheck(
            "service_images_local",
            False,
            f"locked image platform mismatch; expected {expected_platform}: {wrong_platform}",
        )
    return CertCheck("service_images_local", True, f"{len(images)} locked service images are local")


def check_offline_environment(environ: dict[str, str] | None = None) -> CertCheck:
    environ = dict(os.environ) if environ is None else environ
    required = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "UV_OFFLINE": "1",
    }
    missing = [name for name, value in required.items() if environ.get(name) != value]
    if missing:
        return CertCheck("offline_environment", False, f"offline flags not enforced: {missing}")
    return CertCheck("offline_environment", True, "dependency and model clients are forced offline")


def run_preflight(
    root: Path | None = None,
    *,
    requested_profile: str | None = None,
    execution_plan: Path | None = None,
    execution_report: Path | None = None,
    phase: Phase = "final",
    verify_local_services: bool = True,
) -> list[CertCheck]:
    """Run the complete preflight or final certification gate."""

    root = root or _REPO_ROOT
    model_manifest = root / "models.manifest.yaml"
    model_lock = root / "models.lock.yaml"
    service_lock = root / "certification" / "services.lock.yaml"
    profile_check, profile = check_profile_contract(model_manifest, requested=requested_profile)
    plan_path = execution_plan or root / "certification" / "execution_plan.yaml"
    checks = [
        check_no_product_optional_extras(root / "pyproject.toml"),
        check_capability_manifest(root / "capability_manifest.yaml"),
        check_lockfiles_present(root),
        check_makefile_certification_is_strict(root / "Makefile"),
        check_model_manifest(model_manifest),
        profile_check,
        check_execution_plan(root / "capability_manifest.yaml", plan_path, profile=profile),
    ]
    manifest = _load_mapping(model_manifest)
    if profile:
        checks.extend(
            [
                check_no_cpu_fallback(manifest.get("models", []), profile=profile),
                check_model_lock(model_manifest, model_lock, profile=profile),
                check_models_present(model_lock, profile=profile),
            ]
        )
    else:
        checks.append(CertCheck("no_device_fallback", False, "platform profile did not resolve"))
        checks.append(CertCheck("model_lock", False, "platform profile did not resolve"))
        checks.append(CertCheck("models_hydrated", False, "platform profile did not resolve"))
    checks.append(check_service_lock(root / "certification" / "services.yaml", service_lock))
    if verify_local_services:
        checks.append(check_local_service_images(service_lock, profile=profile))
    checks.append(check_offline_environment())
    if phase == "final":
        if execution_report is None:
            checks.append(CertCheck("execution_certificate", False, "final gate requires an execution report"))
        elif not profile:
            checks.append(CertCheck("execution_certificate", False, "platform profile did not resolve"))
        else:
            result = validate_execution_report(
                root / "capability_manifest.yaml",
                execution_report,
                expected_profile=profile,
            )
            checks.append(CertCheck("execution_certificate", result.passed, result.detail))
    return checks


def _artifact_lock_problem(files: object) -> str | None:
    if not isinstance(files, list) or not files:
        return "file lock is empty"
    paths: list[str] = []
    for artifact in files:
        if not isinstance(artifact, dict):
            return "artifact row is not a mapping"
        path = artifact.get("path")
        size = artifact.get("size")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not _DIGEST.fullmatch(f"sha256:{artifact.get('sha256', '')}")
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            return "invalid artifact lock"
        paths.append(path)
    if len(paths) != len(set(paths)):
        return "artifact paths are duplicated"
    return None


def _declared_model_profiles(manifest: dict[str, Any]) -> set[str]:
    profile_rows = manifest.get("profiles", [])
    declared = {
        str(row.get("name"))
        for row in profile_rows
        if isinstance(row, dict) and row.get("name")
    }
    if declared:
        return declared
    for collection in ("models", "service_models"):
        for row in manifest.get(collection, []):
            if isinstance(row, dict):
                declared.update(
                    str(item)
                    for item in row.get("profiles", [])
                    if isinstance(item, str) and item
                )
    return declared


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_nonempty_strings(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item for item in value)
        and len(value) == len(set(value))
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _makefile_recipe(makefile_path: Path, target: str) -> str | None:
    lines = makefile_path.read_text(encoding="utf-8").splitlines()
    recipe: list[str] = []
    capturing = False
    for line in lines:
        if re.match(rf"^{re.escape(target)}\s*:", line):
            capturing = True
            continue
        if capturing:
            if line.startswith("\t") or not line.strip():
                recipe.append(line)
            else:
                break
    return "\n".join(recipe) if capturing else None


def _snapshot_path(repo_id: str, revision: str, *, cache_root: Path | None) -> Path:
    if cache_root is None:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            cache_root = Path(HF_HUB_CACHE)
        except ImportError:
            cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repository = "models--" + repo_id.replace("/", "--")
    return cache_root / repository / "snapshots" / revision


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


__all__ = [
    "CertCheck",
    "check_capability_manifest",
    "check_execution_plan",
    "check_local_service_images",
    "check_lockfiles_present",
    "check_makefile_certification_is_strict",
    "check_model_lock",
    "check_model_manifest",
    "check_models_present",
    "check_no_cpu_fallback",
    "check_no_product_optional_extras",
    "check_offline_environment",
    "check_profile_contract",
    "check_service_lock",
    "run_preflight",
]
