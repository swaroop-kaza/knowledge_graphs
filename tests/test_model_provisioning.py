"""Provisioning invariants that do not download model artifacts."""

from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import hydrate_models, lock_models, wait_for_services

_ROOT = Path(__file__).resolve().parents[1]


class ModelProvisioningTests(unittest.TestCase):
    def test_raw_ollama_hash_is_normalized_to_algorithm_qualified_digest(self) -> None:
        raw = "b" * 64

        self.assertEqual(
            lock_models._canonical_ollama_digest(raw),
            f"sha256:{raw}",
        )
        self.assertEqual(
            hydrate_models._canonical_ollama_digest(raw),
            f"sha256:{raw}",
        )
        self.assertEqual(
            wait_for_services._canonical_ollama_digest(raw),
            f"sha256:{raw}",
        )

    def test_release_model_lock_cannot_be_profile_partial(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            lock_models.build_parser().parse_args(["--profile", "darwin-arm64"])

    def test_lock_models_resolves_every_profile_and_pulls_mandatory_ollama(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text('{"locked":true}', encoding="utf-8")

            def resolve(
                _repo_id: str,
                _ref: str,
                *,
                expected_license: str,
            ) -> tuple[str, Path, str]:
                return "a" * 40, snapshot, expected_license.lower()

            ollama_record = {
                "name": "qwen3:1.7b-q4_K_M",
                "digest": f"sha256:{'b' * 64}",
                "size": 123,
            }
            with (
                redirect_stdout(io.StringIO()),
                patch.object(lock_models, "_resolve_snapshot", side_effect=resolve),
                patch.object(
                    lock_models,
                    "_ollama_tags",
                    side_effect=[{}, {"qwen3:1.7b-q4_K_M": ollama_record}],
                ),
                patch.object(lock_models, "_pull_ollama") as pull,
            ):
                lock = lock_models.build_lock()

        self.assertEqual(lock["version"], 3)
        self.assertEqual(
            lock["profiles"],
            ["darwin-arm64", "linux-cpu", "linux-cuda"],
        )
        self.assertEqual(len(lock["models"]), 22)
        self.assertEqual(lock["artifact_totals"]["unique_huggingface_snapshots"], 8)
        self.assertTrue(all(len(row["profiles"]) == 1 for row in lock["models"]))
        self.assertGreaterEqual(
            {
                row["name"]
                for row in lock["models"]
                if row["profiles"] == ["darwin-arm64"]
            },
            {"mlx_structured_extractor", "gliner_relex_extractor"},
        )
        self.assertEqual(
            lock["manifest_sha256"],
            hashlib.sha256((_ROOT / "models.manifest.yaml").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            lock["service_models"][0]["digest"],
            f"sha256:{'b' * 64}",
        )
        pull.assert_called_once_with("http://localhost:11434", "qwen3:1.7b-q4_K_M")

    def test_hydration_refuses_existing_ollama_tag_drift_without_pulling(self) -> None:
        lock = {
            "service_models": [
                {
                    "model": "qwen3:1.7b-q4_K_M",
                    "profiles": ["linux-cpu"],
                    "digest": f"sha256:{'c' * 64}",
                }
            ]
        }
        tags = {
            "qwen3:1.7b-q4_K_M": {
                "digest": f"sha256:{'d' * 64}",
            }
        }
        with (
            redirect_stdout(io.StringIO()),
            patch.object(hydrate_models, "_ollama_tags", return_value=tags),
            patch.object(hydrate_models, "_pull_ollama") as pull,
        ):
            passed = hydrate_models._hydrate_service_models(
                lock,
                profile="linux-cpu",
                base_url="http://localhost:11434",
                offline=False,
            )
        self.assertFalse(passed)
        pull.assert_not_called()


if __name__ == "__main__":
    unittest.main()
