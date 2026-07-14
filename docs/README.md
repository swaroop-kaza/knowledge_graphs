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

Notebook guide:

- `../notebooks/README.md`
- `../notebooks/capability_coverage.yaml`

Current reports:

- `../reports/benchmark_leaderboard.md`
- `../reports/benchmark_leaderboard.csv`
- `../reports/gold_standard_benchmark_story.md`
- `../reports/gold_standard_benchmark_story.csv`
- `../reports/gold_standard_benchmark_story.json`
- `../reports/current_repo_capability_summary.md`
- `../reports/current_registry_summary.json`
- `../reports/current_registry_matrix.csv`

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
```

`make quality` includes `config-check` and `notebook-check`.

`make notebook-check` validates both notebook hygiene and coverage for every implemented or adapter technique in `default_registry()`.
