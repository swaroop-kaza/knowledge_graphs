from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from kg_construction.interfaces.cli import construct
from kg_construction.interfaces.cli.doctor import run_doctor


class DoctorCliTests(unittest.TestCase):
    def test_run_doctor_validates_core_install_without_services(self) -> None:
        report = run_doctor(skip_services=True)
        by_name = {check.name: check for check in report.checks}

        self.assertIn(report.status, {"ok", "warn"})
        self.assertEqual(by_name["python_version"].status, "ok")
        self.assertEqual(by_name["core_imports"].status, "ok")
        self.assertEqual(by_name["rdf_shacl"].status, "ok")
        self.assertEqual(by_name["config_validity"].status, "ok")
        self.assertEqual(by_name["ollama"].status, "skip")
        self.assertEqual(by_name["neo4j"].status, "skip")
        self.assertEqual(by_name["kuzu"].status, "skip")

    def test_construct_entrypoint_routes_doctor_subcommand_json(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            exit_code = construct.main(["doctor", "--skip-services", "--json"])

        payload = json.loads(stream.getvalue())
        names = {check["name"] for check in payload["checks"]}
        self.assertEqual(exit_code, 0)
        self.assertIn("python_version", names)
        self.assertIn("installed_extras", names)
        self.assertIn("config_validity", names)

    def test_doctor_reports_invalid_extra_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "bad.example.yaml"
            invalid.write_text("[]\n", encoding="utf-8")
            report = run_doctor(
                config_dir=Path(directory) / "missing-configs",
                config_paths=[invalid],
                skip_services=True,
            )

        by_name = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "fail")
        self.assertEqual(by_name["config_validity"].status, "fail")
        self.assertIn("top-level YAML document must be a mapping", by_name["config_validity"].metadata["configs"][0]["error"])


if __name__ == "__main__":
    unittest.main()
