# Benchmarking and Publication Contract

This document defines when a repository result is a smoke test, a correctness benchmark, or a production recommendation. The distinction prevents coverage proxies and deterministic timing controls from being presented as graph accuracy.

## Benchmark Classes

| Class | Required inputs | What it can establish | Current example |
|---|---|---|---|
| Unit/regression | Tiny expected outputs | Code-path stability | `tests/fixtures/gold_standards/` |
| Smoke leaderboard | Input fixtures and deterministic runtime controls | Executability and stable ranking logic | `reports/benchmark_leaderboard.*` |
| Gold-candidate correctness fixture | Frozen input, separate labels, metric contract, gates, hashes | Metric, gate, and identity-policy correctness on the declared fixture; not generalization | `reports/cross_source_identity_benchmark.*` |
| Publishable gold benchmark | Representative frozen input, independently reviewed gold, leakage controls, gates, hashes | Accuracy on the declared dataset, split, and task | Planned domain packs |
| Scale benchmark | Frozen generator/data, hardware profile, repeated measured runs | Throughput, latency, memory and scaling behavior | Planned KROWN/graph-store runs |
| Domain release benchmark | Representative versioned sources, reviewed gold, leakage-safe split and use-case gates | Fitness for one domain and deployment policy | Planned domain packs |

A deterministic runtime value is useful for regression tests but is not a measured performance result. A coverage estimate is useful for diagnostics but is not precision or recall. A technique must not be described as “best” when it has not been compared with independent gold labels.

## Versioned Gold Manifest

`evaluation.gold_dataset_path` now accepts either a direct JSON gold fixture or a JSON/YAML benchmark manifest. Relative fixture paths in a manifest are resolved from the manifest directory.

```yaml
schema_version: "1.0"
benchmark_id: commerce_product_identity
version: "2026.1"
description: Multi-source product identity test split
domain: commerce
split: test
publication_date: "2026-07-21"
license: CC-BY-4.0
input_hash: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
source_inventory:
  - sources/catalog-a.csv
  - sources/catalog-b.json
extraction_fixture: gold_extraction.json       # optional
entity_resolution_fixture: gold_clusters.json # optional
retrieval_fixture: gold_retrieval.json         # optional
citations:
  - https://example.org/dataset-card
metadata:
  benchmark_status: gold-candidate
  leakage_policy: Entities do not cross train/test splits.
```

At least one fixture path is required in a manifest. Every run records:

- benchmark ID and version
- input, gold and configuration SHA-256 hashes
- Git commit
- Git tree state and a SHA-256 fingerprint of the evaluated Python source tree
- package, Python and platform details
- selected techniques and deterministic-runtime mode
- the exact gold tasks that were evaluated
- task support counts and unavailable judgments

Extraction fixtures populate exact entity/span, relation and triple metrics. Entity-resolution fixtures populate pairwise ER and false-merge/false-split metrics. Retrieval fixtures are only scored when a candidate supplies `retrieval_predictions` in batch metadata; otherwise the report says that retrieval was not evaluated instead of emitting a misleading success value.

The retrieval prediction contract is keyed by gold query ID:

```json
{
  "retrieval_predictions": {
    "query-1": {
      "retrieved_context_ids": ["context-7", "context-2"],
      "path": ["entity-a", "RELATED_TO", "entity-b"],
      "citation_ids": ["evidence-9"],
      "answer": "Expected answer text"
    }
  }
}
```

## Publication Gates

The generic evaluator reports gates independently from its weighted ranking. By default a candidate is publication-eligible only when:

- at least one compatible gold task was evaluated
- dangling edge rate is zero
- semantic duplicate edge rate is zero
- evidence coverage is at least 0.95
- validation failure rate is at most 0.05
- false-merge rate is at most 0.02 when ER gold is part of the benchmark
- entity, relation and triple F1 are each at least 0.80 when that gold task is present
- pairwise ER recall is at least 0.80 when ER gold is present
- retrieval Recall@k is at least 0.70 when retrieval predictions and gold are present

These defaults are configured under `hard_gates` in `configs/evaluation.example.yaml`. A candidate that fails a gate remains in the report; suppressing weak baselines would make the publication misleading.

False merges should normally receive a stricter gate than false splits. A false split is visible and recoverable; a false merge can transfer attributes and relationships between distinct real-world entities.

## Published Artifacts

The benchmark CLI emits the same logical run in three forms:

```bash
PYTHONPATH=src python3 -m kg_construction.interfaces.cli.benchmark \
  --input path/to/input \
  --config path/to/evaluation.yaml \
  --output reports/example.md \
  --json-output reports/example.json \
  --csv-output reports/example.csv
```

- Markdown is the human-readable method, result, gate and limitation report.
- JSON is the complete reproducibility and machine-audit record.
- CSV is the stable comparison table for documentation sites and analysis tools.

The default JSON writer is summary-only: candidate batches and graph payloads are omitted because they can reproduce raw source values, evidence text, file paths, or personal data. Candidate graph export is opt-in through `--candidate-output-dir`, and full JSON embedding requires `--include-sensitive-json-artifacts`. Treat either output as sensitive until it has been reviewed and redacted; hashes are not a substitute for a data-publication review.

Committed reports must include their generation command and must be regenerated in CI or checked against a deterministic generator test. Large generated graphs should remain outside Git; manifests, hashes, summaries and small gold fixtures belong in Git.

Committed snapshot reports should declare a stable `publication_date`. Generators may honor `SOURCE_DATE_EPOCH`; otherwise they use the manifest publication date for the rendered timestamp. A report generated from local changes must say `git_tree_state: dirty` and include a generator/source hash rather than implying that the base commit alone reproduces it.

## Cross-Source Identity Benchmark v1

The committed pack under `benchmarks/cross_source_identity/v1/` contains three independent source files and a separate gold cluster fixture. It includes:

- legal-name variants such as `Acme Corporation`, `ACME Corp`, and `Acme`
- punctuation and alias variation such as `Jordan Lee` and `J. Lee`
- a deliberate `Alex Kim` namesake that a name-only resolver must not merge
- scoped trusted identifiers for the reference baseline

Regenerate its Markdown, CSV and JSON publication with:

```bash
make cross-source-identity-benchmark
```

The report deliberately shows exact-name, normalized-name and curated-alias baselines failing the false-merge/precision gates. This is expected evidence that names are matching signals rather than identity keys. The trusted-identifier reference passes this synthetic fixture, but production use still requires issuer, scope, reuse and temporal-validity checks.

## Public Benchmark Roadmap

No single public dataset tests heterogeneous loading, semantic mapping, extraction, identity, provenance, graph validity, retrieval and operations together. Use a layered suite:

1. [W3C R2RML and Direct Mapping test cases](https://www.w3.org/TR/rdb2rdf-test-cases/) for standards conformance.
2. [KROWN](https://kg-construct.github.io/KROWN/) for reproducible RML materialization, heterogeneous joins and resource measurements.
3. [SemTab](https://www.cs.ox.ac.uk/isg/challenges/sem-tab/) and [WDC Schema Matching Benchmark](https://webdatacommons.org/structureddata/smb/) for table-to-KG annotation and schema alignment.
4. [WDC Products](https://webdatacommons.org/largescaleproductcorpus/wdc-products/index.html) and [WDC Block](https://webdatacommons.org/largescaleproductcorpus/wdc-block/) for real multi-source entity matching and blocking.
5. [OAEI Knowledge Graph track](https://oaei.ontologymatching.org/2025/knowledgegraph/index.html) for schema and instance alignment.
6. Domain packs with frozen identifiers, source snapshots, competency queries, updates, deletions, conflicts and human-reviewed hard negatives.

Public datasets retain their own licenses and distribution rules. Store download manifests and checksums in Git, not restricted or oversized source data.

## Domain Benchmark Acceptance Checklist

A dataset can be marked as a repository benchmark only after it has:

- a stable ID, semantic version, checksum and license
- documented source snapshots and permitted redistribution mode
- a governed schema and independent gold scope
- train/dev/test or time/source splits with a leakage policy
- hard positive and hard negative identity examples
- mappings, normalization rules and model/configuration hashes
- exact metrics, support counts and 95% confidence intervals where sample size permits
- replay, source-order and batch-versus-incremental checks
- explicit correctness and operational gates
- Markdown, CSV and JSON outputs
- a benchmark card describing applicability and limitations

Downstream query or GraphRAG results should be reported as a separate utility layer. They complement identity and fact correctness; they do not replace them.
