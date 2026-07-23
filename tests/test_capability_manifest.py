"""The capability manifest must stay complete, honest, and doctrine-consistent.

These checks are the Stage-0 exit gate for the manifest: it covers exactly the
live registry, every required capability has an executable implementation owner,
the required/removed split follows the full-release doctrine, and the committed
file matches a fresh regeneration (so it cannot silently drift).
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import yaml

from kg_construction.application.registry import default_registry

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "capability_manifest.yaml"
_GENERATOR = _ROOT / "scripts" / "generate_capability_manifest.py"


def _load_manifest() -> dict:
    return yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))


def _load_generator():
    spec = importlib.util.spec_from_file_location("capability_manifest_generator", _GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ManifestCompletenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _load_manifest()
        self.registry = default_registry()
        self.techniques = {row["name"]: row for row in self.manifest["techniques"]}

    def test_manifest_covers_exactly_the_registry(self) -> None:
        self.assertEqual(set(self.techniques), self.registry.names())
        self.assertEqual(len(self.manifest["techniques"]), len(self.registry.techniques))

    def test_every_disposition_is_valid(self) -> None:
        for row in self.manifest["techniques"]:
            self.assertIn(row["disposition"], {"required", "removed"}, row["name"])

    def test_required_removed_split_follows_doctrine(self) -> None:
        for descriptor in self.registry.techniques:
            row = self.techniques[descriptor.name]
            expected = "removed" if descriptor.status == "stub" else "required"
            self.assertEqual(
                row["disposition"],
                expected,
                f"{descriptor.name}: status={descriptor.status} must be {expected}",
            )

    def test_every_required_capability_has_an_owner(self) -> None:
        for row in self.manifest["techniques"]:
            if row["disposition"] == "required":
                self.assertTrue(
                    row["owner"],
                    f"required capability {row['name']} has no implementation owner",
                )

    def test_removed_set_is_exactly_the_stubs(self) -> None:
        removed = {row["name"] for row in self.manifest["techniques"] if row["disposition"] == "removed"}
        stubs = {d.name for d in self.registry.techniques if d.status == "stub"}
        self.assertEqual(removed, stubs)

    def test_summary_counts_match_rows(self) -> None:
        for section, summary in self.manifest["summary"].items():
            rows = self.manifest[section]
            required = sum(1 for row in rows if row["disposition"] == "required")
            self.assertEqual(summary["total"], len(rows), section)
            self.assertEqual(summary["required"], required, section)
            self.assertEqual(summary["removed"], len(rows) - required, section)

    def test_committed_manifest_matches_regeneration(self) -> None:
        generator = _load_generator()
        fresh = generator.build_manifest()
        committed = {key: value for key, value in self.manifest.items()}
        self.assertEqual(
            fresh,
            committed,
            "capability_manifest.yaml is stale; run scripts/generate_capability_manifest.py",
        )


if __name__ == "__main__":
    unittest.main()
