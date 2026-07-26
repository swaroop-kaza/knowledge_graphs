"""Installation and runtime diagnostics for kg-construction."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from kg_construction import __all__ as public_exports
from kg_construction.certification.preflight import (
    check_model_lock,
    check_model_manifest,
    check_models_present,
)
from kg_construction.construction.config import DataToKGConfig
from kg_construction.infrastructure.extractors.model_backed.factory import (
    detect_mandatory_profile,
)
from kg_construction.infrastructure.extractors.relations import RelationTypingConfig
from kg_construction.research.benchmark_runner import TechniqueEvaluationConfig
from kg_construction.research.graph_ml import graph_ml_workflow_config_from_dict
from kg_construction.research.hybrid_gliner import HybridGlinerPipelineConfig
from kg_construction.service_runtime import LockedServiceRuntime

DoctorStatus = Literal["ok", "fail"]
JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class DependencyProbe:
    """One distribution/module pair in a mandatory dependency group."""

    distribution: str
    module: str
    required: bool = True


@dataclass(slots=True)
class DoctorCheck:
    """One diagnostic check result."""

    name: str
    category: str
    status: DoctorStatus
    message: str
    hint: str = ""
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class DoctorReport:
    """Complete doctor report."""

    status: DoctorStatus
    checks: list[DoctorCheck]
    summary: dict[str, int]

    def to_dict(self) -> JsonDict:
        return {
            "status": self.status,
            "summary": dict(self.summary),
            "checks": [asdict(check) for check in self.checks],
        }


CORE_IMPORTS: tuple[DependencyProbe, ...] = (
    DependencyProbe("pydantic", "pydantic"),
    DependencyProbe("PyYAML", "yaml"),
    DependencyProbe("jsonpath-ng", "jsonpath_ng"),
    DependencyProbe("openpyxl", "openpyxl"),
    DependencyProbe("SQLAlchemy", "sqlalchemy"),
    DependencyProbe("pypdf", "pypdf"),
    DependencyProbe("rapidfuzz", "rapidfuzz"),
    DependencyProbe("rdflib", "rdflib"),
    DependencyProbe("pyshacl", "pyshacl"),
    DependencyProbe("PyLD", "pyld"),
    DependencyProbe("huggingface-hub", "huggingface_hub"),
    DependencyProbe("pandas", "pandas"),
    DependencyProbe("pyarrow", "pyarrow"),
)

MANDATORY_IMPORT_GROUPS: dict[str, tuple[DependencyProbe, ...]] = {
    "notebook": (
        DependencyProbe("jupyter", "jupyter"),
        DependencyProbe("ipykernel", "ipykernel"),
        DependencyProbe("nbformat", "nbformat"),
    ),
    "gliner": (DependencyProbe("gliner", "gliner"),),
    "llm": (
        DependencyProbe("langchain-ollama", "langchain_ollama"),
        DependencyProbe("langchain-core", "langchain_core"),
        DependencyProbe("langchain-neo4j", "langchain_neo4j"),
    ),
    "bert": (
        DependencyProbe("transformers", "transformers"),
        DependencyProbe("torch", "torch"),
    ),
    "spacy": (DependencyProbe("spacy", "spacy"),),
    "graph": (
        DependencyProbe("networkx", "networkx"),
        DependencyProbe("neo4j", "neo4j"),
        DependencyProbe("numpy", "numpy"),
        DependencyProbe("ladybug", "ladybug"),
    ),
    "rdf": (
        DependencyProbe("rdflib", "rdflib"),
        DependencyProbe("pyshacl", "pyshacl"),
        DependencyProbe("pyld", "pyld"),
    ),
    "linking": (DependencyProbe("sentence-transformers", "sentence_transformers"),),
    "dedup": (
        DependencyProbe("datasketch", "datasketch"),
        DependencyProbe("rapidfuzz", "rapidfuzz"),
        DependencyProbe("splink", "splink"),
        DependencyProbe("dedupe", "dedupe"),
    ),
    "ml": (
        DependencyProbe("numpy", "numpy"),
        DependencyProbe("torch", "torch"),
        DependencyProbe("torch-geometric", "torch_geometric"),
        DependencyProbe("pykeen", "pykeen"),
        DependencyProbe("igraph", "igraph"),
        DependencyProbe("leidenalg", "leidenalg"),
    ),
}

CONFIG_VALIDATORS = {
    "data_to_kg_complete.example.yaml",
    "evaluation.example.yaml",
    "gliner_ollama_relation_typing.example.yaml",
    "graph_ml.example.yaml",
    "integration_profiles.example.yaml",
    "ollama_gliner_label_proposal.example.yaml",
    "production_ingestion.example.yaml",
}

STATUS_ORDER: dict[DoctorStatus, int] = {"fail": 0, "ok": 1}


def build_parser() -> argparse.ArgumentParser:
    """Build the doctor CLI parser."""

    parser = argparse.ArgumentParser(description="Diagnose kg-construction installation and runtime setup.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Retained for CLI compatibility; doctor is always strict.",
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="Service timeout in seconds.")
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Must match the locked OLLAMA_BASE_URL environment contract.",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Must match the locked OLLAMA_MODEL environment contract.",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=None,
        help="Must match the locked NEO4J_URI environment contract.",
    )
    parser.add_argument(
        "--neo4j-user",
        default=None,
        help="Must match the locked NEO4J_USER environment contract.",
    )
    parser.add_argument(
        "--neo4j-database",
        default=None,
        help="Must match the locked NEO4J_DATABASE environment contract.",
    )
    parser.add_argument(
        "--memgraph-uri",
        default=None,
        help="Must match the locked MEMGRAPH_URI environment contract.",
    )
    parser.add_argument("--config-dir", default="configs", help="Directory containing public YAML configs.")
    parser.add_argument("--models-manifest", default="models.manifest.yaml")
    parser.add_argument("--models-lock", default="models.lock.yaml")
    parser.add_argument("--model-cache-root", default=os.getenv("HF_HUB_CACHE"))
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Additional YAML config file to validate. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run doctor checks and print a report."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        service_runtime = LockedServiceRuntime.from_environment()
    except ValueError as exc:
        parser.error(str(exc))
    assertions = {
        "--ollama-url": (
            args.ollama_url.rstrip("/") if args.ollama_url else None,
            service_runtime.ollama_base_url,
        ),
        "--ollama-model": (args.ollama_model, service_runtime.ollama_model),
        "--neo4j-uri": (args.neo4j_uri, service_runtime.neo4j_uri),
        "--neo4j-user": (args.neo4j_user, service_runtime.neo4j_user),
        "--neo4j-database": (
            args.neo4j_database,
            service_runtime.neo4j_database,
        ),
        "--memgraph-uri": (args.memgraph_uri, service_runtime.memgraph_uri),
    }
    conflicts = [
        flag
        for flag, (asserted, expected) in assertions.items()
        if asserted is not None and asserted != expected
    ]
    if conflicts:
        parser.error(
            "service CLI assertions conflict with the locked environment: "
            + ", ".join(conflicts)
        )
    report = run_doctor(
        config_dir=Path(args.config_dir),
        config_paths=[Path(path) for path in args.config],
        timeout=max(0.1, args.timeout),
        service_runtime=service_runtime,
        models_manifest_path=Path(args.models_manifest),
        models_lock_path=Path(args.models_lock),
        model_cache_root=(
            Path(args.model_cache_root) if args.model_cache_root else None
        ),
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(_format_report(report))
    return 0 if all(check.status == "ok" for check in report.checks) else 1


def run_doctor(
    *,
    config_dir: Path = Path("configs"),
    config_paths: list[Path] | None = None,
    timeout: float = 2.0,
    service_runtime: LockedServiceRuntime | None = None,
    models_manifest_path: Path = Path("models.manifest.yaml"),
    models_lock_path: Path = Path("models.lock.yaml"),
    model_cache_root: Path | None = None,
) -> DoctorReport:
    """Run all doctor checks and return a structured report."""

    runtime = service_runtime or LockedServiceRuntime.from_environment()
    locked_models = _check_locked_model_artifacts(
        models_manifest_path,
        models_lock_path,
        cache_root=model_cache_root,
    )
    expected_ollama_digest = (
        str(locked_models.metadata.get("ollama_digest"))
        if locked_models.status == "ok"
        and locked_models.metadata.get("ollama_digest")
        else None
    )
    checks: list[DoctorCheck] = [
        _check_python_version(),
        _check_distribution(),
        _check_core_imports(),
        _check_public_api_exports(),
        _check_mandatory_dependency_groups(),
        _check_platform_runtime(),
        _check_rdf_shacl(),
        _check_notebook_dependencies(),
        _check_config_validity(config_dir, config_paths or []),
        locked_models,
        _check_ollama(
            runtime.ollama_base_url,
            model=runtime.ollama_model,
            expected_digest=expected_ollama_digest,
            timeout=timeout,
        ),
        _check_neo4j(
            runtime.neo4j_uri,
            neo4j_user=runtime.neo4j_user,
            neo4j_password=runtime.neo4j_password,
            database=runtime.neo4j_database,
            timeout=timeout,
        ),
        _check_memgraph(runtime.memgraph_uri, timeout=timeout),
        _check_ladybug(),
    ]

    summary: dict[str, int] = {
        status: sum(1 for check in checks if check.status == status) for status in STATUS_ORDER
    }
    overall = _overall_status(checks)
    return DoctorReport(status=overall, checks=checks, summary=summary)


def _check_python_version() -> DoctorCheck:
    version = sys.version_info
    metadata = {
        "version": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    if version < (3, 11):
        return DoctorCheck(
            "python_version",
            "runtime",
            "fail",
            f"Python {platform.python_version()} is below the required 3.11 minimum.",
            hint="Create a Python 3.11 or 3.12 virtual environment.",
            metadata=metadata,
        )
    message = f"Python {platform.python_version()} satisfies the >=3.11 requirement."
    if version[:2] not in {(3, 11), (3, 12)}:
        message += " This version is newer than the CI-tested 3.11/3.12 classifiers."
    return DoctorCheck("python_version", "runtime", "ok", message, metadata=metadata)


def _check_distribution() -> DoctorCheck:
    try:
        version = importlib.metadata.version("kg-construction")
    except importlib.metadata.PackageNotFoundError:
        return DoctorCheck(
            "package_distribution",
            "package",
            "fail",
            "kg-construction is importable from source, but no installed distribution metadata was found.",
            hint="Run `uv sync --frozen --all-groups` from the repo root.",
        )
    return DoctorCheck(
        "package_distribution",
        "package",
        "ok",
        f"Installed kg-construction distribution version {version}.",
        metadata={"version": version},
    )


def _check_core_imports() -> DoctorCheck:
    results = [_probe_dependency(item) for item in CORE_IMPORTS]
    missing = [item for item in results if not item["available"]]
    if missing:
        return DoctorCheck(
            "core_imports",
            "dependencies",
            "fail",
            f"{len(missing)} core dependency import(s) are missing.",
            hint="Run `python -m pip install -e .`.",
            metadata={"dependencies": results},
        )
    return DoctorCheck(
        "core_imports",
        "dependencies",
        "ok",
        f"All {len(results)} core dependency imports are available.",
        metadata={"dependencies": results},
    )


def _check_public_api_exports() -> DoctorCheck:
    required = {
        "ConstructionResult",
        "ConstructionWorkflow",
        "DataToKGConfig",
        "VersionedGraphEvaluationSuite",
        "construct_kg",
        "convert",
        "load_sources",
    }
    missing = sorted(required - set(public_exports))
    if missing:
        return DoctorCheck(
            "public_api_exports",
            "package",
            "fail",
            f"Missing public API export(s): {', '.join(missing)}.",
        )
    return DoctorCheck(
        "public_api_exports",
        "package",
        "ok",
        "Public construction and evaluation API exports are present.",
        metadata={"checked": sorted(required)},
    )


def _check_mandatory_dependency_groups() -> DoctorCheck:
    payload: dict[str, JsonDict] = {}
    installed: list[str] = []
    partial: list[str] = []
    missing: list[str] = []
    for group_name, probes in MANDATORY_IMPORT_GROUPS.items():
        results = [_probe_dependency(item) for item in probes]
        available = sum(1 for item in results if item["available"])
        if available == len(results):
            installed.append(group_name)
            status = "installed"
        elif available == 0:
            missing.append(group_name)
            status = "not_installed"
        else:
            partial.append(group_name)
            status = "partial"
        payload[group_name] = {"status": status, "dependencies": results}

    message = f"Mandatory dependency groups installed: {installed or ['none']}."
    if partial:
        message += f" Partial groups: {partial}."
    if missing:
        message += f" Missing groups: {missing}."
    return DoctorCheck(
        "mandatory_dependency_groups",
        "dependencies",
        "fail" if partial or missing else "ok",
        message,
        metadata={
            "groups": payload,
            "installed": installed,
            "partial": partial,
            "missing": missing,
        },
    )


def _check_platform_runtime() -> DoctorCheck:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine == "arm64":
        missing = [
            module
            for module in ("mlx", "mlx_lm")
            if not _module_available(module)
        ]
        try:
            import torch

            mps_built = bool(torch.backends.mps.is_built())
            mps_available = bool(torch.backends.mps.is_available())
        except ImportError:
            mps_built = False
            mps_available = False
        if missing or not mps_built or not mps_available:
            return DoctorCheck(
                "platform_runtime",
                "runtime",
                "fail",
                "Darwin ARM64 requires MLX, MLX-LM, and an available Torch MPS device.",
                metadata={
                    "profile": "darwin-arm64",
                    "missing_modules": missing,
                    "mps_built": mps_built,
                    "mps_available": mps_available,
                },
            )
        return DoctorCheck(
            "platform_runtime",
            "runtime",
            "ok",
            "Darwin ARM64 MLX and MPS runtimes are available.",
            metadata={"profile": "darwin-arm64"},
        )
    if system == "Linux" and machine in {"x86_64", "amd64"}:
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
        except ImportError:
            cuda = False
        profile_name = "linux-cuda" if cuda else "linux-cpu"
        return DoctorCheck(
            "platform_runtime",
            "runtime",
            "ok",
            f"{profile_name} runtime selected without fallback.",
            metadata={"profile": profile_name, "cuda_available": cuda},
        )
    return DoctorCheck(
        "platform_runtime",
        "runtime",
        "fail",
        f"No mandatory certification profile exists for {system}/{machine}.",
    )


def _check_rdf_shacl() -> DoctorCheck:
    try:
        from pyshacl import validate
        from rdflib import Graph, Literal, Namespace
        from rdflib.namespace import RDF

        ex = Namespace("https://example.org/kg-doctor/")
        graph = Graph()
        graph.add((ex.node, RDF.type, ex.Thing))
        graph.add((ex.node, ex.name, Literal("doctor")))
        shacl_graph = Graph()
        shacl_graph.parse(
            data="""
                @prefix sh: <http://www.w3.org/ns/shacl#> .
                @prefix ex: <https://example.org/kg-doctor/> .
                ex:ThingShape a sh:NodeShape ;
                  sh:targetClass ex:Thing ;
                  sh:property [ sh:path ex:name ; sh:minCount 1 ] .
            """,
            format="turtle",
        )
        conforms, _, _ = validate(graph, shacl_graph=shacl_graph)
    except Exception as exc:  # noqa: BLE001 - doctor should convert diagnostics into report entries.
        return DoctorCheck(
            "rdf_shacl",
            "standards",
            "fail",
            f"RDF/SHACL validation failed: {type(exc).__name__}: {exc}",
            hint="Run `uv sync --frozen --all-groups`.",
        )
    if not conforms:
        return DoctorCheck("rdf_shacl", "standards", "fail", "pySHACL returned a non-conforming smoke graph.")
    return DoctorCheck("rdf_shacl", "standards", "ok", "RDFLib and pySHACL validated a smoke graph.")


def _check_notebook_dependencies() -> DoctorCheck:
    probes = MANDATORY_IMPORT_GROUPS["notebook"]
    results = [_probe_dependency(item) for item in probes]
    missing = [item["distribution"] for item in results if not item["available"]]
    if missing:
        return DoctorCheck(
            "notebook_dependencies",
            "dependencies",
            "fail",
            f"Mandatory notebook dependencies are missing: {', '.join(missing)}.",
            hint="Run `uv sync --frozen --all-groups`.",
            metadata={"dependencies": results},
        )
    return DoctorCheck(
        "notebook_dependencies",
        "dependencies",
        "ok",
        "Notebook dependencies are available.",
        metadata={"dependencies": results},
    )


def _check_config_validity(config_dir: Path, extra_paths: list[Path]) -> DoctorCheck:
    paths: list[Path] = []
    if config_dir.exists():
        paths.extend(sorted(config_dir.glob("*.yaml")))
    paths.extend(extra_paths)
    if not paths:
        return DoctorCheck(
            "config_validity",
            "config",
            "fail",
            f"No YAML configs found under {config_dir}.",
        )

    results: list[JsonDict] = []
    failures: list[str] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            payload = _load_yaml(path)
            _validate_config_payload(path.name, payload)
        except Exception as exc:  # noqa: BLE001 - doctor reports config failures uniformly.
            message = f"{path}: {type(exc).__name__}: {exc}"
            failures.append(message)
            results.append({"path": str(path), "status": "fail", "error": message})
        else:
            results.append({"path": str(path), "status": "ok"})

    if failures:
        return DoctorCheck(
            "config_validity",
            "config",
            "fail",
            f"{len(failures)} config file(s) failed validation.",
            hint="Run `make config-check` for the stricter public config checker.",
            metadata={"configs": results},
        )
    known = sorted(path.name for path in paths if path.name in CONFIG_VALIDATORS)
    return DoctorCheck(
        "config_validity",
        "config",
        "ok",
        f"{len(results)} YAML config file(s) validated.",
        metadata={"configs": results, "known_configs": known},
    )


def _check_locked_model_artifacts(
    manifest_path: Path,
    lock_path: Path,
    *,
    cache_root: Path | None,
) -> DoctorCheck:
    """Verify the selected profile's manifest, lock, file set, bytes, and hashes."""

    try:
        profile = detect_mandatory_profile()
        manifest = check_model_manifest(manifest_path)
        lock = check_model_lock(manifest_path, lock_path, profile=profile)
        hydrated = check_models_present(
            lock_path,
            profile=profile,
            cache_root=cache_root,
        )
        payload = _load_yaml(lock_path)
    except Exception as exc:  # noqa: BLE001 - doctor reports exact lock failures
        return DoctorCheck(
            "locked_model_artifacts",
            "models",
            "fail",
            f"Locked model verification failed: {type(exc).__name__}: {exc}",
            hint="Run scripts/hydrate_models.py --offline --validate-only.",
        )
    selected = [
        row
        for row in payload.get("models", [])
        if isinstance(row, dict) and profile in row.get("profiles", [])
    ]
    service_models = [
        row
        for row in payload.get("service_models", [])
        if isinstance(row, dict) and profile in row.get("profiles", [])
    ]
    failed = [check for check in (manifest, lock, hydrated) if not check.passed]
    if failed:
        return DoctorCheck(
            "locked_model_artifacts",
            "models",
            "fail",
            "; ".join(check.detail for check in failed),
            hint="Re-lock online, hydrate once, then validate the exact bytes offline.",
            metadata={
                "profile": profile,
                "checks": {
                    check.name: {"passed": check.passed, "detail": check.detail}
                    for check in (manifest, lock, hydrated)
                },
            },
        )
    file_count = sum(
        len(row.get("files", []))
        for row in selected
        if isinstance(row.get("files", []), list)
    )
    byte_count = sum(
        int(file_row.get("size", 0))
        for row in selected
        for file_row in row.get("files", [])
        if isinstance(file_row, dict)
    )
    ollama = next(
        (
            row
            for row in service_models
            if row.get("service") == "ollama"
        ),
        {},
    )
    return DoctorCheck(
        "locked_model_artifacts",
        "models",
        "ok",
        f"{len(selected)} selected snapshots, {file_count} files, and "
        f"{byte_count} bytes match their locked SHA-256 values.",
        metadata={
            "profile": profile,
            "snapshot_count": len(selected),
            "file_count": file_count,
            "byte_count": byte_count,
            "selected_models": [
                {
                    "name": row.get("name"),
                    "revision": row.get("revision"),
                    "device": row.get("device"),
                }
                for row in selected
            ],
            "ollama_model": ollama.get("model"),
            "ollama_digest": ollama.get("digest"),
        },
    )


def _check_ollama(
    base_url: str,
    *,
    model: str,
    timeout: float,
    expected_digest: str | None = None,
) -> DoctorCheck:
    url = f"{base_url.rstrip('/')}/api/tags"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return DoctorCheck(
            "ollama",
            "service",
            "fail",
            f"Ollama is not reachable at {base_url}: {type(exc).__name__}: {exc}",
            hint="Start the locked Ollama service and hydrate its locked model.",
            metadata={"base_url": base_url},
        )
    models = payload.get("models", []) if isinstance(payload, dict) else []
    names = {
        str(row.get("name") or row.get("model"))
        for row in models
        if isinstance(row, dict) and (row.get("name") or row.get("model"))
    }
    if model not in names:
        return DoctorCheck(
            "ollama",
            "service",
            "fail",
            f"Ollama is reachable, but mandatory model {model!r} is absent.",
            hint="Run scripts/hydrate_models.py during the online provisioning phase.",
            metadata={"base_url": base_url, "models": sorted(names), "required_model": model},
        )
    selected = next(
        (
            row
            for row in models
            if isinstance(row, dict)
            and str(row.get("name") or row.get("model")) == model
        ),
        {},
    )
    actual_digest = _canonical_ollama_digest(selected.get("digest"))
    if expected_digest is not None and actual_digest != expected_digest:
        return DoctorCheck(
            "ollama",
            "service",
            "fail",
            f"Ollama model {model!r} digest does not match the model lock.",
            hint="Rehydrate the locked Ollama model; do not accept tag drift.",
            metadata={
                "base_url": base_url,
                "required_model": model,
                "expected_digest": expected_digest,
                "actual_digest": actual_digest,
            },
        )
    return DoctorCheck(
        "ollama",
        "service",
        "ok",
        f"Ollama is reachable and reports mandatory model {model!r}.",
        metadata={
            "base_url": base_url,
            "model_count": len(models),
            "required_model": model,
            "digest": actual_digest,
        },
    )


def _canonical_ollama_digest(value: object) -> str:
    digest = str(value or "").lower()
    if len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    ):
        return f"sha256:{digest}"
    return digest


def _check_neo4j(
    uri: str | None,
    *,
    neo4j_user: str | None,
    neo4j_password: str | None,
    database: str,
    timeout: float,
) -> DoctorCheck:
    if not _module_available("neo4j"):
        return DoctorCheck(
            "neo4j",
            "service",
            "fail",
            "Neo4j Python driver is not installed.",
            hint="Run `uv sync --frozen --all-groups`.",
        )
    if not uri:
        return DoctorCheck(
            "neo4j",
            "service",
            "fail",
            "NEO4J_URI is not set.",
            hint="Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD to verify live connectivity.",
        )
    if neo4j_user and not neo4j_password:
        return DoctorCheck(
            "neo4j",
            "service",
            "fail",
            "Neo4j URI/user are configured, but the locked password is empty.",
            hint="Set NEO4J_PASSWORD in the shared locked-service environment.",
            metadata={"uri": uri, "user": neo4j_user, "database": database},
        )
    try:
        from neo4j import GraphDatabase

        auth = (
            (neo4j_user, neo4j_password)
            if neo4j_user and neo4j_password
            else None
        )
        try:
            driver = GraphDatabase.driver(uri, auth=auth, connection_timeout=timeout)
        except TypeError:
            driver = GraphDatabase.driver(uri, auth=auth)
        try:
            driver.verify_connectivity()
            with driver.session(database=database) as session:
                record = session.run("RETURN 1 AS ok").single()
                if record is None or record["ok"] != 1:
                    raise RuntimeError(
                        "Neo4j semantic query returned the wrong value"
                    )
        finally:
            driver.close()
    except Exception as exc:  # noqa: BLE001 - doctor reports service failures.
        return DoctorCheck(
            "neo4j",
            "service",
            "fail",
            f"Neo4j driver is installed, but connectivity failed: {type(exc).__name__}: {exc}",
            hint="Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, and whether the database is running.",
            metadata={"uri": uri, "user": neo4j_user, "database": database},
        )
    return DoctorCheck(
        "neo4j",
        "service",
        "ok",
        "Neo4j authenticated database query passed.",
        metadata={"uri": uri, "user": neo4j_user, "database": database},
    )


def _check_memgraph(uri: str, *, timeout: float) -> DoctorCheck:
    if not _module_available("neo4j"):
        return DoctorCheck(
            "memgraph",
            "service",
            "fail",
            "Neo4j-compatible Bolt driver is not installed.",
            hint="Run `uv sync --frozen --all-groups`.",
        )
    try:
        from neo4j import GraphDatabase

        try:
            driver = GraphDatabase.driver(uri, auth=None, connection_timeout=timeout)
        except TypeError:
            driver = GraphDatabase.driver(uri, auth=None)
        try:
            with driver.session() as session:
                record = session.run("RETURN 1 AS ok").single()
        finally:
            driver.close()
        if record is None or record["ok"] != 1:
            raise RuntimeError("semantic query returned the wrong result")
    except Exception as exc:  # noqa: BLE001 - every service error is a hard diagnostic failure.
        return DoctorCheck(
            "memgraph",
            "service",
            "fail",
            f"Memgraph connectivity/query failed: {type(exc).__name__}: {exc}",
            metadata={"uri": uri},
        )
    return DoctorCheck(
        "memgraph",
        "service",
        "ok",
        "Memgraph connectivity and semantic query succeeded.",
        metadata={"uri": uri},
    )


def _check_ladybug() -> DoctorCheck:
    if not _module_available("ladybug"):
        return DoctorCheck(
            "ladybug",
            "service",
            "fail",
            "Ladybug Python package is not installed.",
            hint="Run `uv sync --frozen --all-groups`.",
        )
    try:
        import ladybug

        with tempfile.TemporaryDirectory(prefix="kg-construction-ladybug-doctor-") as directory:
            database = ladybug.Database(str(Path(directory) / "doctor.lbdb"))
            connection = ladybug.Connection(database)
            result = connection.execute("RETURN 1 AS ok")
            if isinstance(result, list):
                raise RuntimeError("semantic query unexpectedly returned multiple results")
            if not result.has_next():
                raise RuntimeError("semantic query returned no result")
            row = result.get_next()
            value = row.get("ok") if isinstance(row, dict) else row[0]
            if value != 1:
                raise RuntimeError("semantic query returned the wrong result")
    except Exception as exc:  # noqa: BLE001 - every service error is a hard diagnostic failure.
        return DoctorCheck(
            "ladybug",
            "service",
            "fail",
            f"Ladybug embedded smoke query failed: {type(exc).__name__}: {exc}",
            hint="Check the locked Ladybug package and local filesystem permissions.",
        )
    return DoctorCheck(
        "ladybug",
        "service",
        "ok",
        "Ladybug embedded database smoke query succeeded.",
    )


def _probe_dependency(probe: DependencyProbe) -> JsonDict:
    version = _distribution_version(probe.distribution)
    module_available = _module_available(probe.module)
    return {
        "distribution": probe.distribution,
        "module": probe.module,
        "available": bool(version or module_available),
        "distribution_version": version,
        "module_available": module_available,
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _load_yaml(path: Path) -> JsonDict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("top-level YAML document must be a mapping")
    return payload


def _validate_config_payload(filename: str, payload: JsonDict) -> None:
    if filename == "data_to_kg_complete.example.yaml" or "pipeline" in payload:
        DataToKGConfig.model_validate(payload)
    elif filename == "evaluation.example.yaml" or {"evaluation", "weights", "penalties"} <= set(payload):
        TechniqueEvaluationConfig.model_validate(payload)
    elif filename == "ollama_gliner_label_proposal.example.yaml":
        HybridGlinerPipelineConfig.model_validate(payload)
    elif filename == "gliner_ollama_relation_typing.example.yaml":
        RelationTypingConfig.model_validate(payload.get("relation_typing", payload))
    elif filename == "graph_ml.example.yaml" or "graph_ml" in payload:
        graph_ml_workflow_config_from_dict(payload)
    elif filename == "production_ingestion.example.yaml":
        _require_sections(
            payload,
            "production_ingestion",
            "retry",
            "dead_letter",
            "neo4j",
            "ladybug",
            "replay",
        )
    elif filename == "integration_profiles.example.yaml":
        _require_sections(
            payload,
            "integration_profiles",
            "neo4j_live",
            "ladybug_live",
            "graph_store_batch_benchmark",
        )
    else:
        if filename.endswith(".yaml") and not payload:
            raise ValueError("YAML config must not be empty")


def _require_sections(payload: JsonDict, root_key: str, *sections: str) -> None:
    root = payload.get(root_key)
    if not isinstance(root, dict):
        raise ValueError(f"missing top-level mapping {root_key!r}")
    missing = [section for section in sections if section not in root]
    if missing:
        raise ValueError(f"{root_key} missing required section(s): {missing}")


def _overall_status(checks: list[DoctorCheck]) -> DoctorStatus:
    if any(check.status == "fail" for check in checks):
        return "fail"
    return "ok"


def _format_report(report: DoctorReport) -> str:
    lines = ["kg-construction doctor", ""]
    for check in sorted(report.checks, key=lambda item: (item.category, STATUS_ORDER[item.status], item.name)):
        label = check.status.upper()
        lines.append(f"[{label}] {check.category}/{check.name}: {check.message}")
        if check.hint:
            lines.append(f"       hint: {check.hint}")
    lines.extend(
        [
            "",
            "Summary: "
            + ", ".join(f"{status}={report.summary.get(status, 0)}" for status in ("ok", "fail")),
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DoctorCheck",
    "DoctorReport",
    "build_parser",
    "main",
    "run_doctor",
]


if __name__ == "__main__":
    raise SystemExit(main())
