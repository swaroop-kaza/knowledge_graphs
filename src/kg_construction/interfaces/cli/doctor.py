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

from kg_construction.application import __all__ as application_exports
from kg_construction.application.benchmark_runner import TechniqueEvaluationConfig
from kg_construction.application.config import DataToKGConfig
from kg_construction.application.graph_ml import graph_ml_workflow_config_from_dict
from kg_construction.application.hybrid_gliner import HybridGlinerPipelineConfig
from kg_construction.infrastructure.extractors.relations import RelationTypingConfig

DoctorStatus = Literal["ok", "warn", "fail", "skip"]
JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class DependencyProbe:
    """One distribution/module pair used to infer extra availability."""

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
)

EXTRA_IMPORTS: dict[str, tuple[DependencyProbe, ...]] = {
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
        DependencyProbe("kuzu", "kuzu"),
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

STATUS_ORDER: dict[DoctorStatus, int] = {"fail": 0, "warn": 1, "ok": 2, "skip": 3}


def build_parser() -> argparse.ArgumentParser:
    """Build the doctor CLI parser."""

    parser = argparse.ArgumentParser(description="Diagnose kg-construction installation and runtime setup.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when warnings are present.")
    parser.add_argument("--skip-services", action="store_true", help="Skip Ollama, Neo4j, and Kuzu runtime checks.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Service timeout in seconds.")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME"))
    parser.add_argument("--neo4j-password-env", default="NEO4J_PASSWORD")
    parser.add_argument("--config-dir", default="configs", help="Directory containing public YAML configs.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Additional YAML config file to validate. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run doctor checks and print a report."""

    args = build_parser().parse_args(argv)
    report = run_doctor(
        config_dir=Path(args.config_dir),
        config_paths=[Path(path) for path in args.config],
        timeout=max(0.1, args.timeout),
        ollama_url=args.ollama_url,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password_env=args.neo4j_password_env,
        skip_services=args.skip_services,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(_format_report(report))
    if report.summary.get("fail", 0) > 0:
        return 1
    if args.strict and report.summary.get("warn", 0) > 0:
        return 1
    return 0


def run_doctor(
    *,
    config_dir: Path = Path("configs"),
    config_paths: list[Path] | None = None,
    timeout: float = 2.0,
    ollama_url: str = "http://localhost:11434",
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password_env: str = "NEO4J_PASSWORD",
    skip_services: bool = False,
) -> DoctorReport:
    """Run all doctor checks and return a structured report."""

    checks: list[DoctorCheck] = [
        _check_python_version(),
        _check_distribution(),
        _check_core_imports(),
        _check_public_api_exports(),
        _check_installed_extras(),
        _check_optional_imports(),
        _check_rdf_shacl(),
        _check_notebook_dependencies(),
        _check_config_validity(config_dir, config_paths or []),
    ]
    if skip_services:
        checks.extend(
            [
                DoctorCheck("ollama", "service", "skip", "Service checks were skipped."),
                DoctorCheck("neo4j", "service", "skip", "Service checks were skipped."),
                DoctorCheck("kuzu", "service", "skip", "Service checks were skipped."),
            ]
        )
    else:
        checks.extend(
            [
                _check_ollama(ollama_url, timeout=timeout),
                _check_neo4j(
                    neo4j_uri,
                    neo4j_user=neo4j_user,
                    password_env=neo4j_password_env,
                    timeout=timeout,
                ),
                _check_kuzu(),
            ]
        )

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
            "warn",
            "kg-construction is importable from source, but no installed distribution metadata was found.",
            hint="Run `python -m pip install -e .` from the repo root.",
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
        "DataToKGConfig",
        "ConstructionWorkflow",
        "ConstructionResult",
        "TechniqueEvaluationConfig",
        "TechniqueEvaluationReport",
        "default_registry",
    }
    missing = sorted(required - set(application_exports))
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
        "Public application API exports are present.",
        metadata={"checked": sorted(required)},
    )


def _check_installed_extras() -> DoctorCheck:
    payload: dict[str, JsonDict] = {}
    installed: list[str] = []
    partial: list[str] = []
    missing: list[str] = []
    for extra_name, probes in EXTRA_IMPORTS.items():
        results = [_probe_dependency(item) for item in probes]
        available = sum(1 for item in results if item["available"])
        if available == len(results):
            installed.append(extra_name)
            status = "installed"
        elif available == 0:
            missing.append(extra_name)
            status = "not_installed"
        else:
            partial.append(extra_name)
            status = "partial"
        payload[extra_name] = {"status": status, "dependencies": results}

    message = f"Inferred extras installed: {installed or ['none']}."
    if partial:
        message += f" Partial extras: {partial}."
    return DoctorCheck(
        "installed_extras",
        "dependencies",
        "ok",
        message,
        metadata={"extras": payload, "installed": installed, "partial": partial, "missing": missing},
    )


def _check_optional_imports() -> DoctorCheck:
    groups: dict[str, JsonDict] = {}
    available_count = 0
    total_count = 0
    for group_name, probes in EXTRA_IMPORTS.items():
        results = [_probe_dependency(item) for item in probes]
        available_count += sum(1 for item in results if item["available"])
        total_count += len(results)
        groups[group_name] = {"dependencies": results}
    return DoctorCheck(
        "optional_imports",
        "dependencies",
        "ok",
        f"{available_count}/{total_count} optional imports are available.",
        metadata={"groups": groups},
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
            hint="Install the RDF dependencies with `python -m pip install -e .` or `.[rdf]`.",
        )
    if not conforms:
        return DoctorCheck("rdf_shacl", "standards", "fail", "pySHACL returned a non-conforming smoke graph.")
    return DoctorCheck("rdf_shacl", "standards", "ok", "RDFLib and pySHACL validated a smoke graph.")


def _check_notebook_dependencies() -> DoctorCheck:
    probes = EXTRA_IMPORTS["notebook"]
    results = [_probe_dependency(item) for item in probes]
    missing = [item["distribution"] for item in results if not item["available"]]
    if missing:
        return DoctorCheck(
            "notebook_dependencies",
            "dependencies",
            "skip",
            f"Notebook extra is not fully installed. Missing: {', '.join(missing)}.",
            hint='Install with `python -m pip install -e ".[notebook]"`.',
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
            "skip",
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


def _check_ollama(base_url: str, *, timeout: float) -> DoctorCheck:
    url = f"{base_url.rstrip('/')}/api/tags"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return DoctorCheck(
            "ollama",
            "service",
            "skip",
            f"Ollama is not reachable at {base_url}: {type(exc).__name__}: {exc}",
            hint="Start Ollama or set OLLAMA_BASE_URL. This is optional unless you use LLM extractors.",
            metadata={"base_url": base_url},
        )
    models = payload.get("models", []) if isinstance(payload, dict) else []
    return DoctorCheck(
        "ollama",
        "service",
        "ok",
        f"Ollama is reachable at {base_url}; {len(models)} model(s) reported.",
        metadata={"base_url": base_url, "model_count": len(models)},
    )


def _check_neo4j(
    uri: str | None,
    *,
    neo4j_user: str | None,
    password_env: str,
    timeout: float,
) -> DoctorCheck:
    if not _module_available("neo4j"):
        return DoctorCheck(
            "neo4j",
            "service",
            "skip",
            "Neo4j Python driver is not installed.",
            hint='Install graph extras with `python -m pip install -e ".[graph]"`.',
        )
    if not uri:
        return DoctorCheck(
            "neo4j",
            "service",
            "skip",
            "NEO4J_URI is not set.",
            hint="Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD to verify live connectivity.",
        )
    password = os.getenv(password_env)
    if neo4j_user and not password:
        return DoctorCheck(
            "neo4j",
            "service",
            "warn",
            f"Neo4j URI/user are configured, but {password_env} is not set.",
            hint=f"Export {password_env} before running doctor.",
            metadata={"uri": uri, "user": neo4j_user, "password_env": password_env},
        )
    try:
        from neo4j import GraphDatabase

        auth = (neo4j_user, password) if neo4j_user and password else None
        try:
            driver = GraphDatabase.driver(uri, auth=auth, connection_timeout=timeout)
        except TypeError:
            driver = GraphDatabase.driver(uri, auth=auth)
        try:
            driver.verify_connectivity()
        finally:
            driver.close()
    except Exception as exc:  # noqa: BLE001 - doctor reports optional service failures.
        return DoctorCheck(
            "neo4j",
            "service",
            "warn",
            f"Neo4j driver is installed, but connectivity failed: {type(exc).__name__}: {exc}",
            hint="Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, and whether the database is running.",
            metadata={"uri": uri, "user": neo4j_user},
        )
    return DoctorCheck(
        "neo4j",
        "service",
        "ok",
        "Neo4j connectivity verified with the Python driver.",
        metadata={"uri": uri, "user": neo4j_user},
    )


def _check_kuzu() -> DoctorCheck:
    if not _module_available("kuzu"):
        return DoctorCheck(
            "kuzu",
            "service",
            "skip",
            "Kuzu Python package is not installed.",
            hint='Install graph extras with `python -m pip install -e ".[graph]"`.',
        )
    try:
        import kuzu

        with tempfile.TemporaryDirectory(prefix="kg-construction-kuzu-doctor-") as directory:
            database = kuzu.Database(directory)
            connection = kuzu.Connection(database)
            connection.execute("RETURN 1 AS ok")
    except Exception as exc:  # noqa: BLE001 - doctor reports optional backend failures.
        return DoctorCheck(
            "kuzu",
            "service",
            "warn",
            f"Kuzu is installed, but an embedded smoke query failed: {type(exc).__name__}: {exc}",
            hint="Check the installed Kuzu package and local filesystem permissions.",
        )
    return DoctorCheck("kuzu", "service", "ok", "Kuzu embedded database smoke query succeeded.")


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
        _require_sections(payload, "production_ingestion", "retry", "dead_letter", "neo4j", "kuzu", "replay")
    elif filename == "integration_profiles.example.yaml":
        _require_sections(payload, "integration_profiles", "neo4j_live", "kuzu_live", "graph_store_batch_benchmark")
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
    if any(check.status == "warn" for check in checks):
        return "warn"
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
            + ", ".join(f"{status}={report.summary.get(status, 0)}" for status in ("ok", "warn", "fail", "skip")),
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
