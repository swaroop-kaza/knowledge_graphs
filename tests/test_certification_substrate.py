"""Focused invariants for mandatory locked-artifact, evidence-backed certification."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import textwrap
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from kg_construction.certification.execution import validate_execution_report
from kg_construction.certification.preflight import (
    check_capability_manifest,
    check_execution_plan,
    check_lockfiles_present,
    check_makefile_certification_is_strict,
    check_model_lock,
    check_model_manifest,
    check_models_present,
    check_no_cpu_fallback,
    check_no_product_optional_extras,
    check_offline_environment,
    check_service_lock,
)
from kg_construction.certification.profiles import select_profile
from kg_construction.certification.runner import dispatch_capabilities

_ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


class DependencySubstrateTests(unittest.TestCase):
    def test_no_product_optional_extras(self) -> None:
        self.assertTrue(check_no_product_optional_extras(_ROOT / "pyproject.toml").passed)

    def test_every_product_backend_is_required(self) -> None:
        deps = " ".join(_pyproject()["project"]["dependencies"])
        for package in (
            "torch",
            "transformers",
            "gliner",
            "spacy",
            "neo4j",
            "ladybug",
            "sentence-transformers",
            "pykeen",
            "mlx",
            "mlx-lm",
            "pandas",
            "pyarrow",
        ):
            self.assertIn(package, deps, f"{package} must be a required dependency")

    def test_tooling_groups_are_mandatory(self) -> None:
        groups = _pyproject().get("dependency-groups", {})
        self.assertEqual({"dev", "notebook", "benchmark", "evaluation", "all"}, set(groups))


class ManifestAndProfileTests(unittest.TestCase):
    def test_certified_manifest_contains_only_mandatory_executables(self) -> None:
        result = check_capability_manifest(_ROOT / "capability_manifest.yaml")
        self.assertTrue(result.passed, result.detail)

    def test_exact_platform_profiles_exist(self) -> None:
        manifest = yaml.safe_load((_ROOT / "models.manifest.yaml").read_text(encoding="utf-8"))
        profiles = {row["name"]: row for row in manifest["profiles"]}
        self.assertEqual({"darwin-arm64", "linux-cpu", "linux-cuda"}, set(profiles))
        self.assertTrue({"mlx", "mps", "ollama"} <= set(profiles["darwin-arm64"]["mandatory_backends"]))
        for row in profiles.values():
            self.assertEqual(
                {"ollama", "neo4j", "memgraph", "ladybug"},
                set(row["mandatory_services"]),
            )

    @patch("kg_construction.certification.profiles.platform.machine", return_value="x86_64")
    @patch("kg_construction.certification.profiles.platform.system", return_value="Linux")
    @patch("kg_construction.certification.profiles._cuda_available", return_value=False)
    def test_linux_cpu_profile_is_selected_without_fallback(self, *_mocks) -> None:
        manifest = yaml.safe_load((_ROOT / "models.manifest.yaml").read_text(encoding="utf-8"))
        self.assertEqual(select_profile(manifest).name, "linux-cpu")
        with self.assertRaisesRegex(ValueError, "does not match"):
            select_profile(manifest, "linux-cuda")

    def test_qwen3_model_set_is_consistent_across_backends(self) -> None:
        manifest = yaml.safe_load((_ROOT / "models.manifest.yaml").read_text(encoding="utf-8"))
        repo_ids = {row["repo_id"] for row in manifest["models"]}
        tags = {row["model"] for row in manifest["service_models"]}
        self.assertIn("Qwen/Qwen3-1.7B", repo_ids)
        self.assertIn("mlx-community/Qwen3-1.7B-4bit", repo_ids)
        self.assertIn("qwen3:1.7b-q4_K_M", tags)

    def test_model_devices_are_explicit_for_every_profile(self) -> None:
        manifest = yaml.safe_load((_ROOT / "models.manifest.yaml").read_text(encoding="utf-8"))
        for profile in ("darwin-arm64", "linux-cpu", "linux-cuda"):
            check = check_no_cpu_fallback(manifest["models"], profile=profile)
            self.assertTrue(check.passed, check.detail)

    def test_complete_mandatory_model_manifest_is_enforced(self) -> None:
        result = check_model_manifest(_ROOT / "models.manifest.yaml")
        self.assertTrue(result.passed, result.detail)
        manifest = yaml.safe_load((_ROOT / "models.manifest.yaml").read_text(encoding="utf-8"))
        manifest["models"] = [
            row
            for row in manifest["models"]
            if row["name"] != "gliner_relex_extractor"
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.manifest.yaml"
            path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
            failed = check_model_manifest(path)
        self.assertFalse(failed.passed)
        self.assertIn("gliner_relex_extractor", failed.detail)


class CertificationRecipeTests(unittest.TestCase):
    def test_certify_target_is_strict(self) -> None:
        result = check_makefile_certification_is_strict(_ROOT / "Makefile")
        self.assertTrue(result.passed, result.detail)

    def test_softened_certify_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Makefile"
            path.write_text("certify:\n\tpython scripts/certify.py || true\n", encoding="utf-8")
            self.assertFalse(check_makefile_certification_is_strict(path).passed)

    def test_missing_locks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(check_lockfiles_present(Path(directory)).passed)

    def test_offline_flags_are_all_required(self) -> None:
        self.assertFalse(check_offline_environment({}).passed)
        self.assertTrue(
            check_offline_environment(
                {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "UV_OFFLINE": "1"}
            ).passed
        )


class ArtifactLockTests(unittest.TestCase):
    def test_model_lock_and_local_bytes_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "hub"
            revision = "a" * 40
            snapshot = cache / "models--example--tiny" / "snapshots" / revision
            snapshot.mkdir(parents=True)
            artifact = snapshot / "config.json"
            artifact.write_bytes(b'{"model":"tiny"}')
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            manifest = {
                "models": [
                    {
                        "name": "tiny",
                        "repo_id": "example/tiny",
                        "ref": "main",
                        "backend": "transformers",
                        "task": "test",
                        "profiles": ["linux-cpu"],
                        "device_by_profile": {"linux-cpu": "cpu"},
                        "license": "apache-2.0",
                        "smoke": {"expect": "ok"},
                    }
                ],
                "service_models": [],
            }
            lock = {
                "version": 3,
                "generated_by": "scripts/lock_models.py",
                "profiles": ["linux-cpu"],
                "artifact_totals": {
                    "unique_huggingface_snapshots": 1,
                    "unique_huggingface_bytes": artifact.stat().st_size,
                    "ollama_models": 0,
                    "ollama_bytes": 0,
                },
                "models": [
                    {
                        "name": "tiny",
                        "repo_id": "example/tiny",
                        "ref": "main",
                        "backend": "transformers",
                        "task": "test",
                        "revision": revision,
                        "profiles": ["linux-cpu"],
                        "device": "cpu",
                        "accelerator_policy": "native",
                        "license": "apache-2.0",
                        "model_card_license": "apache-2.0",
                        "smoke": {"expect": "ok"},
                        "files": [
                            {"path": "config.json", "sha256": digest, "size": artifact.stat().st_size}
                        ],
                    }
                ],
                "service_models": [],
            }
            manifest_path = root / "models.manifest.yaml"
            lock_path = root / "models.lock.yaml"
            manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
            lock["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            lock_path.write_text(yaml.safe_dump(lock), encoding="utf-8")
            self.assertTrue(
                check_model_lock(manifest_path, lock_path, profile="linux-cpu").passed
            )
            self.assertTrue(
                check_models_present(lock_path, profile="linux-cpu", cache_root=cache).passed
            )
            artifact.write_bytes(b"tampered")
            self.assertFalse(
                check_models_present(lock_path, profile="linux-cpu", cache_root=cache).passed
            )

    def test_service_compose_lock_requires_digest_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "services.yaml"
            lock = root / "services.lock.yaml"
            source.write_text(
                yaml.safe_dump({"services": {"graph": {"image": "vendor/graph:1.2.3"}}}),
                encoding="utf-8",
            )
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            lock.write_text(
                yaml.safe_dump(
                    {
                        "services": {
                            "graph": {"image": f"vendor/graph@sha256:{'b' * 64}"}
                        },
                        "x-certification-lock": {
                            "version": 2,
                            "generated_by": "scripts/lock_service_digests.py",
                            "source_sha256": source_sha256,
                            "services": {
                                "graph": {
                                    "source_image": "vendor/graph:1.2.3",
                                    "digest": f"sha256:{'b' * 64}",
                                }
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(check_service_lock(source, lock).passed)
            lock.write_text(
                yaml.safe_dump({"services": {"graph": {"image": "vendor/graph:latest"}}}),
                encoding="utf-8",
            )
            self.assertFalse(check_service_lock(source, lock).passed)


class LiveExecutionAttestationTests(unittest.TestCase):
    def test_preflight_rejects_missing_capability_certifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.yaml"
            plan = root / "plan.yaml"
            manifest.write_text(
                yaml.safe_dump({"techniques": [{"name": "required"}]}),
                encoding="utf-8",
            )
            plan.write_text("capabilities: {}\n", encoding="utf-8")
            self.assertFalse(
                check_execution_plan(manifest, plan, profile="linux-cpu").passed
            )
            plan.write_text(
                yaml.safe_dump(
                    {
                        "capabilities": {
                            "required": {
                                "profiles": ["linux-cpu"],
                                "command": [sys.executable, "-c", "raise SystemExit(0)"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                check_execution_plan(manifest, plan, profile="linux-cpu").passed
            )

    def test_hand_authored_success_report_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.yaml"
            report = root / "report.yaml"
            manifest.write_text(
                yaml.safe_dump({"techniques": [{"name": "real_capability"}]}),
                encoding="utf-8",
            )
            report.write_text(
                yaml.safe_dump(
                    {
                        "profile": "linux-cpu",
                        "capabilities": [
                            {
                                "name": "real_capability",
                                "declared": True,
                                "configured": True,
                                "instantiated": True,
                                "executed": True,
                                "evaluated": True,
                                "passed": True,
                                "skipped": False,
                                "fallback": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(
                validate_execution_report(
                    manifest,
                    report,
                    expected_profile="linux-cpu",
                ).passed
            )

    def test_dispatcher_generates_fresh_per_capability_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.yaml"
            plan = root / "plan.yaml"
            report = root / "report.yaml"
            certifier = root / "certifier.py"
            manifest.write_text(
                yaml.safe_dump({"techniques": [{"name": "real_capability"}]}),
                encoding="utf-8",
            )
            certifier.write_text(
                textwrap.dedent(
                    """
                    import json
                    import os
                    from pathlib import Path

                    name = os.environ["KG_CERT_CAPABILITY"]
                    profile = os.environ["KG_CERT_PROFILE"]
                    run_id = os.environ["KG_CERT_RUN_ID"]
                    observations = {
                        "instantiated": True,
                        "executed": True,
                        "evaluated": True,
                        "passed": True,
                        "skipped": False,
                        "fallback": False,
                        "instantiated_evidence": {
                            "implementation": "tests.RealCapability"
                        },
                        "executed_evidence": {
                            "fixture_sha256": "a" * 64,
                            "output_sha256": "b" * 64,
                        },
                        "evaluated_evidence": {
                            "metrics": {"f1": 1.0},
                            "thresholds": {"f1": 1.0},
                        },
                        "passed_evidence": {
                            "assertions": [{"name": "f1", "passed": True}]
                        },
                    }
                    Path(os.environ["KG_CERT_EVIDENCE_PATH"]).write_text(json.dumps({
                        "capability": name,
                        "profile": profile,
                        "run_id": run_id,
                        "observations": observations,
                    }), encoding="utf-8")
                    """
                ),
                encoding="utf-8",
            )
            plan.write_text(
                yaml.safe_dump(
                    {
                        "capabilities": {
                            "real_capability": {
                                "profiles": ["linux-cpu"],
                                "command": [sys.executable, str(certifier)],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            dispatch_capabilities(
                manifest,
                plan,
                profile="linux-cpu",
                output_path=report,
            )
            result = validate_execution_report(
                manifest,
                report,
                expected_profile="linux-cpu",
            )
            self.assertTrue(result.passed, result.detail)

    def test_missing_certifier_never_becomes_a_skip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.yaml"
            plan = root / "plan.yaml"
            report = root / "report.yaml"
            manifest.write_text(
                yaml.safe_dump({"techniques": [{"name": "unconfigured"}]}),
                encoding="utf-8",
            )
            plan.write_text("capabilities: {}\n", encoding="utf-8")
            generated = dispatch_capabilities(
                manifest,
                plan,
                profile="linux-cpu",
                output_path=report,
            )
            row = generated["capabilities"][0]
            self.assertFalse(row["configured"])
            self.assertFalse(row["skipped"])
            self.assertFalse(
                validate_execution_report(
                    manifest,
                    report,
                    expected_profile="linux-cpu",
                ).passed
            )


if __name__ == "__main__":
    unittest.main()
