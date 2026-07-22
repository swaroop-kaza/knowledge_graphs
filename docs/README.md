# Documentation

This directory is the documentation hub for the Data-to-Knowledge Graph Construction Framework.

Start here:

1. `setup_running_configuration_guide.md`
   - Environment setup.
   - Install commands.
   - Running construction and benchmarking.
   - Configuration files.
   - Validation commands.
   - Generated artifact policy.

2. `repo_implementation_inventory.md`
   - Full implementation inventory.
   - Current capability coverage.
   - Registry totals.
   - Tests and proof points.

3. `packaging_and_notebook_policy.md`
   - Why `pyproject.toml` is the setup source of truth.
   - Why `setup.py`, `setup.cfg`, and `requirements.txt` are not required.
   - Notebook hygiene policy.
   - Registry-to-notebook capability coverage policy.

4. `support_levels.md`
   - Stable, Optional adapter, Experimental, and Planned labels.
   - How public support levels map to registry statuses.
   - When to use optional dependencies and services.
   - How to inspect support levels from Python and reports.

5. `graph_ml_capabilities.md`
   - Graph analytics.
   - PageRank.
   - Communities.
   - Embeddings.
   - Link prediction.
   - Retrieval benchmarks.

6. `production_ingestion.md`
   - Neo4j ingestion.
   - Kuzu ingestion.
   - Retry policy.
   - Dead-letter replay.
   - Live integration profiles.

7. `next_stage_capabilities.md`
   - Standards mapping.
   - Event/observability graph coverage.
   - Graph ML.
   - Production ingestion.
   - Gold benchmark fixtures.

8. `github_open_source_readiness_plan.md`
   - Public GitHub readiness checklist.
   - Security settings.
   - Branch protection.
   - Release readiness.

9. [`benchmarking_and_publication.md`](benchmarking_and_publication.md)
   - Difference between smoke, gold correctness, domain, and scale benchmarks.
   - Versioned gold-manifest and hard-gate contract.
   - Reproducible Markdown, CSV, and JSON publication rules.

10. [`multisource_kg_benchmark_plan.md`](multisource_kg_benchmark_plan.md)
    - Multi-source identity and duplicate-control invariants.
    - Current implementation versus unimplemented production requirements.
    - Evaluation metrics, hard gates, and benchmark acceptance criteria.
    - Standards, public-access domain packs, and phased implementation backlog.

11. [`identity_v2_migration.md`](identity_v2_migration.md)
    - Breaking graph-identity changes in this implementation wave.
    - Rebuild and staged migration guidance for existing graph stores.

Notebook guide:

- `../notebooks/README.md`
- `../notebooks/capability_coverage.yaml`

Current reports:

- `../reports/benchmark_leaderboard.md`
- `../reports/benchmark_leaderboard.csv`
- `../reports/gold_standard_benchmark_story.md`
- `../reports/gold_standard_benchmark_story.csv`
- `../reports/gold_standard_benchmark_story.json`
- [`../reports/cross_source_identity_benchmark.md`](../reports/cross_source_identity_benchmark.md)
- [`../reports/cross_source_identity_benchmark.csv`](../reports/cross_source_identity_benchmark.csv)
- [`../reports/cross_source_identity_benchmark.json`](../reports/cross_source_identity_benchmark.json)
- `../reports/current_repo_capability_summary.md`
- `../reports/current_registry_summary.json`
- `../reports/current_registry_matrix.csv`

Current benchmark packs:

- [`../benchmarks/cross_source_identity/v1/README.md`](../benchmarks/cross_source_identity/v1/README.md)
- [`../benchmarks/cross_source_identity/v1/benchmark_manifest.yaml`](../benchmarks/cross_source_identity/v1/benchmark_manifest.yaml)

## Verification Commands

Run these before publishing changes:

```bash
make config-check
make notebook-check
make quality
make verify
make package-check
make benchmark-leaderboard
make gold-benchmark-story
make cross-source-identity-benchmark
make registry-reports
```

`make quality` includes `config-check` and `notebook-check`.

`make notebook-check` validates both notebook hygiene and coverage for every implemented or adapter technique in `default_registry()`.
