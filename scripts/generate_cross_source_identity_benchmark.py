"""Generate the committed cross-source identity/dedup benchmark publication."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import yaml

from kg_construction.domain.evidence import row_evidence
from kg_construction.domain.models import ConstructionBatch, EntityCandidate, stable_id
from kg_construction.evaluation.entity_resolution_metrics import (
    EntityResolutionMetrics,
    predicted_merge_pairs,
)
from kg_construction.evaluation.gold_standards import (
    GoldEntityResolutionFixture,
)
from kg_construction.infrastructure.loaders.csv_loader import CsvLoader
from kg_construction.infrastructure.resolution.entity_resolution import (
    AliasTableResolver,
    NormalizedStringResolver,
)
from kg_construction.infrastructure.resolution.exact import ExactKeyEntityResolver
from kg_construction.research.benchmark_runner import GoldDatasetManifest

BENCHMARK_DIR = Path("benchmarks/cross_source_identity/v1")
MANIFEST_PATH = BENCHMARK_DIR / "benchmark_manifest.yaml"
REPORTS_DIR = Path("reports")
REPORT_MD = REPORTS_DIR / "cross_source_identity_benchmark.md"
REPORT_CSV = REPORTS_DIR / "cross_source_identity_benchmark.csv"
REPORT_JSON = REPORTS_DIR / "cross_source_identity_benchmark.json"

MINIMUM_PAIRWISE_PRECISION = 0.98
MAXIMUM_FALSE_MERGE_RATE = 0.02

ALIASES = {
    "Acme Corporation": ["ACME Corp", "Acme"],
    "Jordan Lee": ["J. Lee"],
    "Globex LLC": ["Globex"],
}


class Resolver(Protocol):
    """Minimal resolver interface used by the benchmark."""

    name: str
    technique: str

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch: ...


class TrustedIdentifierResolver:
    """Reference baseline that treats scoped trusted identifiers as exact keys."""

    name = "trusted_identifier_resolver"
    technique = "trusted_identifier_exact_matching"

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        entities = []
        for entity in batch.entities:
            trusted_identifier = str(entity.properties["trusted_identifier"])
            canonical_id = stable_id(
                "canonical_entity",
                entity.entity_type,
                trusted_identifier.casefold(),
            )
            entities.append(
                entity.model_copy(
                    update={
                        "properties": {
                            **entity.properties,
                            "canonical_entity_id": canonical_id,
                            "canonical_id": canonical_id,
                            "resolution_method": self.technique,
                            "resolution_confidence": 1.0,
                            "review_required": False,
                        }
                    }
                )
            )
        return batch.model_copy(
            update={
                "entities": entities,
                "metadata": {**batch.metadata, "resolver": self.name},
            }
        )


def main() -> int:
    """Regenerate Markdown, CSV, and JSON benchmark outputs."""

    rows, metadata = generate_results()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(rows)
    _write_json(rows, metadata)
    _write_markdown(rows, metadata)
    print(f"cross-source-identity-benchmark-ok: {len(rows)} baselines written")
    return 0


def generate_results() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the committed baselines and return publication-ready data."""

    manifest_payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest = GoldDatasetManifest.model_validate(manifest_payload)
    manifest_gates = dict(manifest.metadata.get("hard_gates", {}))
    minimum_pairwise_precision = float(
        manifest_gates.get("minimum_pairwise_precision", MINIMUM_PAIRWISE_PRECISION)
    )
    maximum_false_merge_rate = float(
        manifest_gates.get("maximum_false_merge_rate", MAXIMUM_FALSE_MERGE_RATE)
    )
    minimum_evidence_coverage = float(
        manifest_gates.get("minimum_evidence_coverage", 1.0)
    )
    require_source_order_invariance = bool(
        manifest_gates.get("require_source_order_invariance", True)
    )
    source_inventory = manifest.source_inventory or list(
        manifest.metadata.get("source_files", [])
    )
    if not source_inventory:
        raise ValueError("Benchmark manifest must declare source_inventory.")
    source_files = [BENCHMARK_DIR / str(path) for path in source_inventory]
    input_hash = _input_inventory_hash(source_files)
    if manifest.input_hash is not None and input_hash != manifest.input_hash:
        raise ValueError(
            "Benchmark input hash does not match manifest: "
            f"expected {manifest.input_hash}, got {input_hash}."
        )
    gold_path = BENCHMARK_DIR / str(manifest.entity_resolution_fixture)
    gold = GoldEntityResolutionFixture.from_json_file(gold_path)
    batch = _load_batch(source_files)
    resolver_factories: list[tuple[str, Callable[[], Resolver]]] = [
        (
            "exact_normalized_type_and_name",
            lambda: ExactKeyEntityResolver(allow_cross_source_name_fallback=True),
        ),
        ("normalized_name_and_legal_suffix", NormalizedStringResolver),
        ("curated_alias_table", lambda: AliasTableResolver(aliases=ALIASES)),
        ("trusted_identifier_exact", TrustedIdentifierResolver),
    ]
    rows: list[dict[str, Any]] = []
    for baseline, factory in resolver_factories:
        resolved = factory().resolve(batch)
        permuted_batches = [
            batch.model_copy(update={"entities": list(reversed(batch.entities))}),
            batch.model_copy(
                update={"entities": sorted(batch.entities, key=lambda item: item.candidate_id)}
            ),
            _load_batch(list(reversed(source_files))),
        ]
        metrics = EntityResolutionMetrics().evaluate(
            resolved,
            gold_pairs=gold.gold_pairs(),
        )
        predicted_pairs = predicted_merge_pairs(resolved)
        canonical_cluster_count = len(
            {
                str(entity.properties.get("canonical_entity_id", entity.candidate_id))
                for entity in resolved.entities
            }
        )
        source_order_invariant = all(
            predicted_pairs == predicted_merge_pairs(factory().resolve(permuted_batch))
            for permuted_batch in permuted_batches
        )
        precision_gate = metrics["pairwise_er_precision"] >= minimum_pairwise_precision
        false_merge_gate = metrics["false_merge_rate"] <= maximum_false_merge_rate
        evidence_coverage = sum(bool(entity.evidence) for entity in resolved.entities) / max(
            1,
            len(resolved.entities),
        )
        evidence_gate = evidence_coverage >= minimum_evidence_coverage
        order_gate = source_order_invariant or not require_source_order_invariance
        passes_all_gates = (
            precision_gate
            and false_merge_gate
            and evidence_gate
            and order_gate
        )
        rows.append(
            {
                "baseline": baseline,
                "resolver": factory().name,
                "technique": factory().technique,
                "record_count": len(batch.entities),
                "gold_cluster_count": len({record.cluster_id for record in gold.records}),
                "predicted_cluster_count": canonical_cluster_count,
                "gold_pair_count": len(gold.gold_pairs()),
                "predicted_pair_count": len(predicted_pairs),
                "pairwise_precision": _round(metrics["pairwise_er_precision"]),
                "pairwise_recall": _round(metrics["pairwise_er_recall"]),
                "pairwise_f1": _round(metrics["pairwise_er_f1"]),
                "false_merge_rate": _round(metrics["false_merge_rate"]),
                "false_split_rate": _round(metrics["false_split_rate"]),
                "evidence_coverage": _round(evidence_coverage),
                "precision_gate": precision_gate,
                "false_merge_gate": false_merge_gate,
                "evidence_gate": evidence_gate,
                "source_order_invariant": source_order_invariant,
                "passes_all_gates": passes_all_gates,
            }
        )

    dataset_files = [MANIFEST_PATH, *source_files, gold_path]
    dataset_hash = _combined_hash(dataset_files)
    config_hash = hashlib.sha256(
        json.dumps(
            {
                "aliases": ALIASES,
                "baselines": [name for name, _factory in resolver_factories],
                "minimum_pairwise_precision": minimum_pairwise_precision,
                "maximum_false_merge_rate": maximum_false_merge_rate,
                "minimum_evidence_coverage": minimum_evidence_coverage,
                "require_source_order_invariance": require_source_order_invariance,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    git_commit, git_tree_state = _git_metadata()
    generator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_tree_hash = _source_tree_hash()
    results_hash = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    publication_date = manifest.publication_date or datetime.now(UTC).date().isoformat()
    metadata = {
        "schema_version": "1.0",
        "benchmark_id": manifest.benchmark_id,
        "benchmark_version": manifest.version,
        "benchmark_status": str(
            manifest.metadata.get("benchmark_status", "gold-candidate")
        ),
        "publication_date": publication_date,
        "generated_at": _reproducible_timestamp(publication_date),
        "generation_time_policy": "SOURCE_DATE_EPOCH_or_manifest_publication_date",
        "dataset_hash": dataset_hash,
        "input_hash": input_hash,
        "manifest_hash": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "gold_hash": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "source_hashes": {
            path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "config_hash": config_hash,
        "generator_hash": generator_hash,
        "source_tree_hash": source_tree_hash,
        "results_hash": results_hash,
        "run_id": hashlib.sha256(
            (
                f"{dataset_hash}:{config_hash}:{generator_hash}:"
                f"{source_tree_hash}:{results_hash}"
            ).encode()
        ).hexdigest()[:16],
        "git_commit": git_commit,
        "git_tree_state": git_tree_state,
        "generation_command": "make cross-source-identity-benchmark",
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "kg_construction": _distribution_version("kg-construction"),
            "pydantic": _distribution_version("pydantic"),
            "pyyaml": _distribution_version("PyYAML"),
        },
        "source_files": [path.as_posix() for path in source_files],
        "gold_fixture": gold_path.as_posix(),
        "gold_support": {
            "records": len(gold.records),
            "clusters": len({record.cluster_id for record in gold.records}),
            "positive_pairs": len(gold.gold_pairs()),
        },
        "hard_gates": {
            "minimum_pairwise_precision": minimum_pairwise_precision,
            "maximum_false_merge_rate": maximum_false_merge_rate,
            "minimum_evidence_coverage": minimum_evidence_coverage,
            "require_source_order_invariance": require_source_order_invariance,
        },
        "limitations": manifest.metadata["limitations"],
    }
    return rows, metadata


def _load_batch(source_files: list[Path]) -> ConstructionBatch:
    sources = []
    entities = []
    for source_path in source_files:
        source = CsvLoader().load(source_path)[0]
        sources.append(source)
        for row_index, row in enumerate(source.records):
            evidence = [row_evidence(source, row_index, row)]
            entities.append(
                EntityCandidate(
                    candidate_id=str(row["record_id"]),
                    entity_type=str(row["entity_type"]),
                    name=str(row["name"]),
                    properties={
                        "source_local_id": row["source_local_id"],
                        "trusted_identifier": row["trusted_identifier"],
                        "source_name": source.source_name,
                    },
                    evidence=evidence,
                    extractor_name="cross_source_benchmark_loader",
                    technique="gold_fixture_record_loading",
                )
            )
    return ConstructionBatch(
        source_records=sources,
        entities=entities,
        metadata={"benchmark_id": "cross_source_identity_dedup"},
    )


def _write_csv(rows: list[dict[str, Any]]) -> None:
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    REPORT_JSON.write_text(
        json.dumps({"metadata": metadata, "results": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_markdown(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    lines = [
        "# Cross-Source Identity and Deduplication Benchmark",
        "",
        "[Reports index](README.md) · [Evaluation guide](../docs/evaluation_metrics.md) · [Benchmark fixture](../benchmarks/cross_source_identity/v1/README.md)",
        "",
        "| Metadata | Value |",
        "| --- | --- |",
        f"| Benchmark | `{metadata['benchmark_id']}` v{metadata['benchmark_version']} |",
        f"| Run ID | `{metadata['run_id']}` |",
        f"| Git commit | `{metadata['git_commit']}` |",
        f"| Git tree state | `{metadata['git_tree_state']}` |",
        f"| Python | `{metadata['environment']['python']}` |",
        "",
        "> **Research only.** This synthetic comparison is non-authoritative and "
        "cannot certify identity quality, graph correctness, or publication eligibility.",
        "",
        "Status: **gold-candidate synthetic regression fixture**. It is published for "
        "review and CI, not presented as a production-scale or domain-release leaderboard.",
        "",
        "The fixture contains aliases, legal suffix variations, exact repeated names, and "
        "a deliberate namesake. A failed gate is reported honestly; rows are never removed "
        "because they score poorly.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "make cross-source-identity-benchmark",
        "```",
        "",
        "### Reproducibility hashes",
        "",
        "| Artifact | SHA-256 |",
        "| --- | --- |",
        f"| Dataset | `{metadata['dataset_hash']}` |",
        f"| Input | `{metadata['input_hash']}` |",
        f"| Gold | `{metadata['gold_hash']}` |",
        f"| Manifest | `{metadata['manifest_hash']}` |",
        f"| Config | `{metadata['config_hash']}` |",
        f"| Generator | `{metadata['generator_hash']}` |",
        f"| Source tree | `{metadata['source_tree_hash']}` |",
        f"| Results | `{metadata['results_hash']}` |",
        "",
        "## Results",
        "",
        "| Baseline | Pairs | Precision | Recall | F1 | False merge | False split | Evidence | Order invariant | Gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['baseline']}` | {row['predicted_pair_count']} | "
            f"{row['pairwise_precision']:.3f} | {row['pairwise_recall']:.3f} | "
            f"{row['pairwise_f1']:.3f} | {row['false_merge_rate']:.3f} | "
            f"{row['false_split_rate']:.3f} | {row['evidence_coverage']:.3f} | "
            f"{row['source_order_invariant']} | "
            f"{'PASS' if row['passes_all_gates'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Hard Gates",
            "",
            "| Gate | Requirement |",
            "| --- | ---: |",
            f"| Pairwise precision | at least `{float(metadata['hard_gates']['minimum_pairwise_precision']):.2f}` |",
            f"| False-merge rate | at most `{float(metadata['hard_gates']['maximum_false_merge_rate']):.2f}` |",
            f"| Evidence coverage | at least `{float(metadata['hard_gates']['minimum_evidence_coverage']):.2f}` |",
            f"| Source-record order invariance | `{metadata['hard_gates']['require_source_order_invariance']}` |",
            "",
            "## Interpretation",
            "",
            "The name-based baselines expose the core multi-source identity risk. Exact names "
            "miss aliases and merge namesakes. Normalization and curated aliases improve recall "
            "but do not make a person name unique. The trusted-identifier reference baseline "
            "passes this fixture because its identifiers are explicitly scoped and authoritative.",
            "",
            "Do not generalize the perfect trusted-identifier score beyond this fixture: real "
            "systems must validate identifier issuer, scope, reuse, and temporal validity.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in metadata["limitations"])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.relative_to(BENCHMARK_DIR).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _input_inventory_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.relative_to(BENCHMARK_DIR).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_metadata() -> tuple[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"
    return commit.stdout.strip() or "unknown", "dirty" if status.stdout.strip() else "clean"


def _source_tree_hash() -> str:
    paths = [Path(__file__), *Path("src/kg_construction").rglob("*.py")]
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "source-tree" if distribution == "kg-construction" else "unknown"


def _reproducible_timestamp(publication_date: str) -> str:
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch:
        try:
            return datetime.fromtimestamp(int(source_date_epoch), tz=UTC).isoformat()
        except ValueError as exc:
            raise ValueError("SOURCE_DATE_EPOCH must be an integer Unix timestamp.") from exc
    return f"{publication_date}T00:00:00+00:00"


def _round(value: float) -> float:
    return round(float(value), 6)


if __name__ == "__main__":
    raise SystemExit(main())
