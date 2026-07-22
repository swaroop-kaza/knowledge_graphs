# Setup, Running, And Configuration Guide

This guide combines the setup, running, configuration, notebook, and verification instructions for the repo.

## 1. Prerequisites

Required:

- Python 3.11 or 3.12.
- `pip`.
- A shell that can run `make`.

Optional:

- Ollama for local LLM extraction.
- Neo4j for live graph database ingestion.
- Kuzu for embedded graph storage.
- Jupyter Lab for notebooks.
- GLiNER and ML extras for model-backed extraction and graph ML experiments.

Support levels:

- Stable techniques should work through the documented API with normal dependencies.
- Optional adapters require the matching optional dependency, model, service, or credentials.
- Experimental techniques are implemented for research and benchmarking, but APIs or metrics may change.
- Planned techniques are registered extension points and should not be treated as callable features.

See `docs/support_levels.md` for the full support-level matrix.

## 2. Create A Local Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Run the 60-second data scientist quickstart with core dependencies only:

```bash
python -m pip install -e .
PYTHONPATH=src python3 examples/quickstart_60_second.py
```

This path covers:

- CSV to KG.
- Text to KG.
- JSON to KG.
- Technique benchmarking.
- RDF export.
- Neo4j Cypher export without a live Neo4j connection.
- NetworkX node-link export without installing the `networkx` optional extra.

Install the full developer environment when you need quality checks and tests:

```bash
python -m pip install -e ".[dev,rdf,graph]"
```

Install notebook support:

```bash
python -m pip install -e ".[dev,rdf,graph,notebook]"
```

Install optional integrations only when needed:

```bash
python -m pip install -e ".[gliner,llm,linking,dedup,ml]"
```

Check the local environment with doctor:

```bash
kg-construction doctor
kg-construction doctor --json
kg-construction doctor --skip-services
```

Doctor checks Python, package metadata, core dependencies, inferred extras, optional imports, RDF/SHACL, notebooks, config validity, and optional Ollama/Neo4j/Kuzu availability. It is read-only and treats missing optional services as skipped unless they are explicitly configured and fail.

View the small committed benchmark leaderboard when deciding which technique to try first:

```bash
sed -n '1,80p' reports/benchmark_leaderboard.md
```

Regenerate it from tiny fixtures:

```bash
make benchmark-leaderboard
```

View the gold-standard benchmark story when checking whether the evaluation examples are credible:

```bash
sed -n '1,120p' reports/gold_standard_benchmark_story.md
```

Regenerate it from human-reviewed gold fixtures:

```bash
make gold-benchmark-story
```

## 3. Setup File Policy

The repo uses `pyproject.toml` as the setup source of truth.

Required setup files:

- `pyproject.toml`
- `LICENSE`
- `README.md`
- `src/kg_construction/py.typed`

Not required:

- `setup.py`
- `setup.cfg`
- `requirements.txt`

Reason:

- `pyproject.toml` defines the build backend, package metadata, dependencies, optional extras, console scripts, pytest config, Ruff config, mypy config, and vulture config.
- Adding separate setup or requirements files would duplicate dependency declarations and create drift.

## 4. Configuration Files

Current configuration examples:

| File | Purpose | Checked by |
| --- | --- | --- |
| `configs/data_to_kg_complete.example.yaml` | End-to-end framework configuration for loaders, mappers, extractors, validation, graph stores, enrichment, evaluation, and production hardening. | `DataToKGConfig` |
| `configs/evaluation.example.yaml` | Technique evaluation, deterministic runtime, scoring weights, penalties, thresholds, and LLM metadata. | `TechniqueEvaluationConfig` |
| `configs/ollama_gliner_label_proposal.example.yaml` | Ollama label proposal plus GLiNER entity extraction workflow. | `HybridGlinerPipelineConfig` |
| `configs/gliner_ollama_relation_typing.example.yaml` | GLiNER entity spans plus Ollama relation typing settings. | `RelationTypingConfig` |
| `configs/graph_ml.example.yaml` | Graph analytics, PageRank, communities, components, embeddings, and link prediction. | `GraphMLWorkflowConfig` |
| `configs/production_ingestion.example.yaml` | Neo4j, Kuzu, retry, dead-letter, and replay settings. | Config-check script |
| `configs/integration_profiles.example.yaml` | Optional live Neo4j/Kuzu integration profiles and graph-store benchmark settings. | Config-check script |

Run:

```bash
make config-check
```

This validates:

- YAML syntax.
- Expected top-level sections.
- Pydantic models where available.
- Registry-backed technique names.
- Numeric bounds for retry, batch, benchmark, graph ML, and model settings.
- Generated-output paths under `data/processed/` or `reports/`.
- Optional-service configs without calling external services.

## 5. Run Construction

From the repo root:

```bash
PYTHONPATH=src python3 -m kg_construction.interfaces.cli.construct \
  --input examples/data_to_kg/sample.txt \
  --source-type text \
  --output data/processed/data_to_kg_text_property_graph.json \
  --rdf-output data/processed/data_to_kg_text_rdf.json \
  --evidence-output data/processed/data_to_kg_text_evidence_graph.json \
  --neo4j-output data/processed/data_to_kg_text_neo4j.cypher
```

Outputs:

- Property graph JSON.
- RDF JSON.
- Evidence graph JSON.
- Neo4j Cypher export.

All outputs belong in `data/processed/` and should not be committed.

## 6. Use The Python API

The stable top-level API is designed for notebooks, scripts, and data-science workflows:

```python
from kg_construction import construct_kg, evaluate_techniques, load_sources, profile_dataset

sources = load_sources("examples/data_to_kg/sample.csv")
profile = profile_dataset(sources)

result = construct_kg("examples/data_to_kg/sample.csv")
property_graph = result.property_graph
rdf_graph = result.rdf_graph
evidence_graph = result.evidence_graph

report = evaluate_techniques(
    "examples/data_to_kg/sample.txt",
    source_type="text",
    techniques="applicable",
)
print(report.best_technique)
```

Public top-level exports:

- `construct_kg`
- `evaluate_techniques`
- `load_sources`
- `profile_dataset`
- `DataToKGConfig`
- `ConstructionWorkflow`
- `ConstructionResult`
- `default_registry`

## 7. Run Technique Evaluation

```bash
PYTHONPATH=src python3 -m kg_construction.interfaces.cli.benchmark \
  --input examples/data_to_kg/sample.txt \
  --source-type text \
  --config configs/evaluation.example.yaml \
  --techniques applicable \
  --output data/processed/data_to_kg_technique_evaluation_report.md \
  --json-output data/processed/data_to_kg_technique_evaluation_report.json \
  --candidate-output-dir data/processed/data_to_kg_candidates
```

The benchmark:

- Profiles the input.
- Selects applicable construction techniques.
- Runs candidate construction paths.
- Validates candidate graphs.
- Computes scoring dimensions.
- Applies deterministic tie-breaking.
- Emits Markdown and JSON reports.

## 8. Run Notebooks

Install notebook extras:

```bash
python -m pip install -e ".[dev,rdf,graph,notebook]"
```

Start Jupyter:

```bash
jupyter lab
```

Suggested order:

1. `notebooks/01_data_loading_and_profiling.ipynb`
2. `notebooks/02_structured_schema_mapping_rdf.ipynb`
3. `notebooks/03_jsonpath_xml_semistructured_mapping.ipynb`
4. `notebooks/04_text_extraction_gliner_ollama_labels.ipynb`
5. `notebooks/05_evaluation_resolution_production_hardening.ipynb`
6. `notebooks/06_graph_ml_capabilities.ipynb`
7. `notebooks/07_standards_mapping_and_virtual_kg.ipynb`
8. `notebooks/08_relation_extraction_and_entity_linking.ipynb`
9. `notebooks/09_enterprise_events_and_observability.ipynb`
10. `notebooks/10_graph_store_ingestion_and_exports.ipynb`
11. `notebooks/11_gold_benchmarks_and_retrieval.ipynb`

Notebook coverage is tracked in `notebooks/capability_coverage.yaml`. Implemented and adapter techniques must appear in this coverage map; stubs are documented as extension points.

Notebook policy:

- Do not commit outputs.
- Do not commit execution counts.
- Do not use absolute local paths.
- Keep generated graphs and reports out of `notebooks/`.

Run:

```bash
make notebook-check
```

## 9. Production Ingestion Commands

Dry-run dead-letter replay:

```bash
PYTHONPATH=src python3 -m kg_construction.interfaces.cli.replay_dead_letters \
  --dead-letter-path data/processed/graph_ingestion_dead_letters.jsonl \
  --store kuzu_ingestion_adapter \
  --dry-run
```

Kuzu replay:

```bash
PYTHONPATH=src python3 -m kg_construction.interfaces.cli.replay_dead_letters \
  --dead-letter-path data/processed/graph_ingestion_dead_letters.jsonl \
  --store kuzu_ingestion_adapter \
  --kuzu-database data/processed/kuzu_graph
```

Neo4j replay uses environment variables:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your-password
export NEO4J_DATABASE=neo4j
```

Secrets must stay in local environment variables or secret managers. Do not commit secrets.

## 10. External Dataset Runs

External public datasets are generated locally and ignored by Git.

Download or create local experiment inputs:

```bash
PYTHONPATH=src python3 scripts/download_experiment_datasets.py
```

Run dataset experiments:

```bash
PYTHONPATH=src python3 scripts/run_experiment_datasets.py
```

Generated locations:

- `data/external/experiment_datasets/`
- `reports/experiment_datasets/`

These are intentionally not committed because they can become large, stale, or license-sensitive.

## 11. Verification Commands

Run focused checks:

```bash
make config-check
make notebook-check
```

Run quality checks:

```bash
make quality
```

Run tests and smoke flows:

```bash
make verify
```

Build the package:

```bash
make package-check
```

`make package-check` removes old `build/` and `dist/`, runs `python -m build`, and validates generated distributions with `python -m twine check dist/*`.

Clean generated local files:

```bash
make clean
```

## 12. What `make quality` Checks

`make quality` runs:

- Ruff over `src`, `tests`, and `scripts`.
- mypy over `src/kg_construction`.
- vulture over `src` and `tests`.
- configuration validation over `configs/*.yaml`.
- notebook hygiene validation over `notebooks/*.ipynb`.
- notebook coverage validation against `kg_construction.application.registry.default_registry()`.

## 13. Public Repo Checklist

Before pushing public:

- Confirm `LICENSE`.
- Run `make quality`.
- Run `make verify`.
- Run `make package-check`.
- Run `make benchmark-leaderboard` when benchmark logic or scoring changes.
- Optionally install from the generated wheel in a fresh environment and run `kg-construction doctor --skip-services`.
- Clean `build/`, `dist/`, caches, and generated outputs.
- Confirm `data/external/experiment_datasets/` is not committed.
- Confirm `data/processed/` contains only `README.md`.
- Confirm notebooks have no outputs.
- Confirm `make notebook-check` passes with registry coverage.
- Add repository URLs to `pyproject.toml` and `CITATION.cff` after the GitHub remote exists.
