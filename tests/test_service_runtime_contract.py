from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from scripts import wait_for_services

from kg_construction.construction.config import DataToKGConfig
from kg_construction.interfaces.cli import doctor
from kg_construction.service_runtime import (
    DEFAULT_MEMGRAPH_URI,
    DEFAULT_NEO4J_DATABASE,
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USER,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    LockedServiceRuntime,
)

_ROOT = Path(__file__).resolve().parents[1]


def test_package_env_is_the_single_locked_default_contract() -> None:
    runtime = LockedServiceRuntime.from_environment({})

    assert runtime.as_environment() == {
        "NEO4J_URI": DEFAULT_NEO4J_URI,
        "NEO4J_USER": DEFAULT_NEO4J_USER,
        "NEO4J_PASSWORD": DEFAULT_NEO4J_PASSWORD,
        "NEO4J_DATABASE": DEFAULT_NEO4J_DATABASE,
        "MEMGRAPH_URI": DEFAULT_MEMGRAPH_URI,
        "OLLAMA_BASE_URL": DEFAULT_OLLAMA_BASE_URL,
        "OLLAMA_MODEL": DEFAULT_OLLAMA_MODEL,
    }
    assert len(runtime.contract_sha256) == 64
    assert "certification" not in repr(runtime)


def test_conflicting_or_empty_legacy_environment_fails_closed() -> None:
    with pytest.raises(ValueError, match="conflicting"):
        LockedServiceRuntime.from_environment(
            {
                "NEO4J_URI": DEFAULT_NEO4J_URI,
                "KG_CERT_NEO4J_URI": "bolt://elsewhere:7687",
            }
        )
    with pytest.raises(ValueError, match="set but empty"):
        LockedServiceRuntime.from_environment({"NEO4J_PASSWORD": ""})


def test_legacy_certification_names_resolve_only_when_unambiguous() -> None:
    runtime = LockedServiceRuntime.from_environment(
        {
            "KG_CERT_NEO4J_URI": "bolt://graph.internal:7687",
            "KG_CERT_NEO4J_USER": "locked-user",
            "KG_CERT_NEO4J_PASSWORD": "locked-password",
            "KG_CERT_NEO4J_DATABASE": "locked-database",
            "KG_CERT_MEMGRAPH_URI": "bolt://memgraph.internal:7687",
            "KG_CERT_OLLAMA_URL": "http://ollama.internal:11434",
            "KG_CERT_OLLAMA_MODEL": DEFAULT_OLLAMA_MODEL,
        }
    )

    assert runtime.neo4j_uri == "bolt://graph.internal:7687"
    assert runtime.neo4j_database == "locked-database"
    assert runtime.memgraph_uri == "bolt://memgraph.internal:7687"
    assert runtime.ollama_base_url == "http://ollama.internal:11434"


def test_compose_and_shell_hard_pass_match_the_package_contract() -> None:
    runtime = LockedServiceRuntime.from_environment({})
    compose = yaml.safe_load(
        (_ROOT / "certification/services.lock.yaml").read_text(encoding="utf-8")
    )
    services = compose["services"]
    run_script = (_ROOT / "run_all_approaches.sh").read_text(encoding="utf-8")
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "7687:7687" in services["neo4j"]["ports"]
    assert services["neo4j"]["environment"]["NEO4J_AUTH"] == (
        f"{runtime.neo4j_user}/{runtime.neo4j_password}"
    )
    assert "7688:7687" in services["memgraph"]["ports"]
    assert "11434:11434" in services["ollama"]["ports"]
    assert "source src/kg_construction/resources/service-runtime.env" in run_script
    assert "include src/kg_construction/resources/service-runtime.env" in makefile


def test_doctor_passes_one_runtime_to_every_service_probe() -> None:
    runtime = LockedServiceRuntime(
        neo4j_uri="bolt://graph.internal:7687",
        neo4j_user="doctor-user",
        neo4j_password="doctor-password",
        neo4j_database="doctor-database",
        memgraph_uri="bolt://memgraph.internal:7687",
        ollama_base_url="http://ollama.internal:11434",
    )
    ok = doctor.DoctorCheck("ok", "test", "ok", "passed")

    with (
        patch.object(doctor, "_check_ollama", return_value=ok) as ollama,
        patch.object(doctor, "_check_neo4j", return_value=ok) as neo4j,
        patch.object(doctor, "_check_memgraph", return_value=ok) as memgraph,
        patch.object(
            doctor,
            "_check_locked_model_artifacts",
            return_value=ok,
        ),
        patch.object(doctor, "_check_distribution", return_value=ok),
        patch.object(doctor, "_check_core_imports", return_value=ok),
        patch.object(doctor, "_check_public_api_exports", return_value=ok),
        patch.object(doctor, "_check_mandatory_dependency_groups", return_value=ok),
        patch.object(doctor, "_check_platform_runtime", return_value=ok),
        patch.object(doctor, "_check_rdf_shacl", return_value=ok),
        patch.object(doctor, "_check_notebook_dependencies", return_value=ok),
        patch.object(doctor, "_check_config_validity", return_value=ok),
        patch.object(doctor, "_check_ladybug", return_value=ok),
    ):
        report = doctor.run_doctor(service_runtime=runtime)

    assert report.status == "ok"
    assert ollama.call_args.args[0] == runtime.ollama_base_url
    assert ollama.call_args.kwargs["model"] == runtime.ollama_model
    assert neo4j.call_args.args[0] == runtime.neo4j_uri
    assert neo4j.call_args.kwargs == {
        "neo4j_user": runtime.neo4j_user,
        "neo4j_password": runtime.neo4j_password,
        "database": runtime.neo4j_database,
        "timeout": 2.0,
    }
    assert memgraph.call_args.args[0] == runtime.memgraph_uri


def test_product_config_and_locked_ollama_runtime_must_agree() -> None:
    from kg_construction.construction.api import _validate_service_runtime

    config = DataToKGConfig()
    _validate_service_runtime(config, LockedServiceRuntime.from_environment({}))

    with pytest.raises(ValueError, match="llm.base_url"):
        _validate_service_runtime(
            config,
            LockedServiceRuntime(ollama_base_url="http://other:11434"),
        )


def test_readiness_uses_the_locked_neo4j_database_and_credentials() -> None:
    runtime = LockedServiceRuntime(
        neo4j_uri="bolt://graph.internal:7687",
        neo4j_user="readiness-user",
        neo4j_password="readiness-password",
        neo4j_database="readiness-database",
    )
    record = {"ok": 1}
    session = MagicMock()
    session.run.return_value.single.return_value = record
    session_context = MagicMock()
    session_context.__enter__.return_value = session
    driver = MagicMock()
    driver.session.return_value = session_context

    with patch(
        "neo4j.GraphDatabase.driver",
        return_value=driver,
    ) as graph_driver:
        detail = wait_for_services._probe_neo4j(runtime)

    assert detail == "authenticated query passed"
    graph_driver.assert_called_once_with(
        runtime.neo4j_uri,
        auth=(runtime.neo4j_user, runtime.neo4j_password),
        connection_timeout=3,
    )
    driver.session.assert_called_once_with(database=runtime.neo4j_database)
    assert session.run.call_args.args == ("RETURN 1 AS ok",)


def test_certification_has_no_divergent_service_environment_namespace() -> None:
    source = (
        _ROOT / "src/kg_construction/certification/capabilities.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "KG_CERT_NEO4J_URI",
        "KG_CERT_NEO4J_USER",
        "KG_CERT_NEO4J_PASSWORD",
        "KG_CERT_NEO4J_DATABASE",
        "KG_CERT_MEMGRAPH_URI",
        "KG_CERT_OLLAMA_URL",
        "KG_CERT_OLLAMA_MODEL",
    )

    assert not any(name in source for name in forbidden)
