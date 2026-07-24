from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from kg_construction.interfaces.cli import construct, doctor
from kg_construction.interfaces.cli.doctor import DoctorCheck, run_doctor


def _ok(name: str, category: str) -> DoctorCheck:
    return DoctorCheck(name, category, "ok", f"{name} passed")


class DoctorCliTests(unittest.TestCase):
    def _mandatory_runtime_patches(self, *, include_config: bool = True):
        patches = [
            patch.object(doctor, "_check_distribution", return_value=_ok("package_distribution", "package")),
            patch.object(doctor, "_check_core_imports", return_value=_ok("core_imports", "dependencies")),
            patch.object(
                doctor,
                "_check_installed_extras",
                return_value=_ok("mandatory_dependency_groups", "dependencies"),
            ),
            patch.object(doctor, "_check_platform_runtime", return_value=_ok("platform_runtime", "runtime")),
            patch.object(
                doctor,
                "_check_locked_model_artifacts",
                return_value=_ok("locked_model_artifacts", "models"),
            ),
            patch.object(
                doctor,
                "_check_notebook_dependencies",
                return_value=_ok("notebook_dependencies", "dependencies"),
            ),
            patch.object(doctor, "_check_ollama", return_value=_ok("ollama", "service")),
            patch.object(doctor, "_check_neo4j", return_value=_ok("neo4j", "service")),
            patch.object(doctor, "_check_memgraph", return_value=_ok("memgraph", "service")),
            patch.object(doctor, "_check_ladybug", return_value=_ok("ladybug", "service")),
        ]
        if include_config:
            patches.append(
                patch.object(
                    doctor,
                    "_check_config_validity",
                    return_value=_ok("config_validity", "config"),
                )
            )
        return tuple(patches)

    def test_run_doctor_requires_every_runtime_and_has_zero_skips(self) -> None:
        patches = self._mandatory_runtime_patches()
        for item in patches:
            item.start()
        try:
            report = run_doctor()
        finally:
            for item in reversed(patches):
                item.stop()
        self.assertEqual(report.status, "ok")
        self.assertEqual(set(report.summary), {"ok", "fail"})
        self.assertTrue(all(check.status == "ok" for check in report.checks))
        names = {check.name for check in report.checks}
        self.assertTrue(
            {"locked_model_artifacts", "ollama", "neo4j", "memgraph", "ladybug"}
            <= names
        )

    def test_construct_entrypoint_routes_strict_doctor_json(self) -> None:
        patches = self._mandatory_runtime_patches()
        for item in patches:
            item.start()
        stream = io.StringIO()
        try:
            with redirect_stdout(stream):
                exit_code = construct.main(["doctor", "--json"])
        finally:
            for item in reversed(patches):
                item.stop()
        payload = json.loads(stream.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(set(payload["summary"]), {"ok", "fail"})

    def test_skip_services_switch_is_rejected(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            doctor.build_parser().parse_args(["--skip-services"])

    @patch(
        "kg_construction.interfaces.cli.doctor.urllib.request.urlopen",
        side_effect=urllib.error.URLError("offline"),
    )
    def test_unreachable_ollama_is_a_hard_failure(self, _urlopen) -> None:
        result = doctor._check_ollama(
            "http://localhost:11434",
            model="qwen3:1.7b-q4_K_M",
            timeout=0.1,
        )
        self.assertEqual(result.status, "fail")

    def test_doctor_reports_invalid_extra_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "bad.example.yaml"
            invalid.write_text("[]\n", encoding="utf-8")
            patches = self._mandatory_runtime_patches(include_config=False)
            for item in patches:
                item.start()
            try:
                report = run_doctor(
                    config_dir=Path(directory) / "missing-configs",
                    config_paths=[invalid],
                )
            finally:
                for item in reversed(patches):
                    item.stop()
        by_name = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "fail")
        self.assertEqual(by_name["config_validity"].status, "fail")
        self.assertIn(
            "top-level YAML document must be a mapping",
            by_name["config_validity"].metadata["configs"][0]["error"],
        )


if __name__ == "__main__":
    unittest.main()
