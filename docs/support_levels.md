# Technique Support Levels

The registry keeps two kinds of maturity metadata:

- `status`: internal implementation state used by planners, tests, and reports.
- `public_support_level`: user-facing maturity label used by README, docs, notebooks, and reports.

This separation lets the repo stay honest. A technique can have working code and still be marked as optional or experimental when it depends on a model, service, or research workflow.

## Public Labels

| Public support level | Internal signal | Use it when | Be careful when |
| --- | --- | --- | --- |
| Stable | Usually `implemented` | You need a documented path that runs with normal dependencies and local tests. | You need long-term API guarantees beyond the current `0.1.x` package maturity. |
| Optional adapter | `adapter`, or `implemented` plus optional dependency/service tags | You have configured the required package, model, API, graph store, or service. | You expect it to work from a core-only install or without credentials/runtime services. |
| Experimental | `implemented` research-heavy capability | You are comparing techniques, running benchmarks, or exploring graph ML/LLM workflows. | You need stable metrics, stable model behavior, or a frozen public API. |
| Planned | `stub` | You are looking for extension points or roadmap coverage. | You need executable production behavior today. |

## Internal Registry Status

| Registry status | Meaning |
| --- | --- |
| `implemented` | Working code exists and is registered. It may still be Stable, Optional adapter, or Experimental depending on dependency and maturity. |
| `adapter` | A boundary exists for a real implementation or optional workflow, but it needs extra configuration, dependency installation, service access, or project-specific wiring. It maps to Optional adapter. |
| `stub` | A named future capability exists as an explicit extension point. It maps to Planned. |

## Practical Rule

- Start with Stable techniques for notebooks, demos, CI, and first production pilots.
- Use Optional adapters after installing extras such as `gliner`, `llm`, `graph`, `linking`, `dedup`, or `ml`, and after configuring any required services.
- Treat Experimental techniques as benchmarkable research paths. Pin versions and capture run manifests before comparing results.
- Treat Planned techniques as design placeholders, not callable product features.

## Examples

| Technique | Registry status | Public support level | Why |
| --- | --- | --- | --- |
| `csv_loader` | `implemented` | Stable | Runs locally with normal dependencies and test coverage. |
| `regex_entity_extractor` | `implemented` | Stable | Deterministic, lightweight, and core-friendly. |
| `gliner_entity_extractor` | `implemented` | Optional adapter | Real GLiNER inference requires the optional `gliner` extra or an injected model. |
| `ollama_graph_extractor` | `implemented` | Optional adapter | Requires a configured Ollama runtime or injected provider. |
| `rebel_relation_extractor` | `adapter` | Optional adapter | Expects a Hugging Face-style generation pipeline. |
| `node_embedding_enricher` | `implemented` | Experimental | Graph ML and embedding behavior may evolve as benchmarks mature. |
| `graph_retrieval_benchmark` | `implemented` | Experimental | Research/evaluation API and scoring assumptions may change. |
| `chunk_embedding_enricher` | `stub` | Planned | Registered roadmap extension point only. |

## Programmatic Access

```python
from kg_construction import default_registry

registry = default_registry()

for technique in registry.techniques:
    print(
        technique.name,
        technique.status,
        technique.public_support_level,
        technique.public_support_label,
    )
```

## Report Files

Run:

```bash
make registry-reports
```

This refreshes:

- `reports/current_repo_capability_summary.md`
- `reports/current_registry_summary.json`
- `reports/current_registry_matrix.csv`

The CSV and JSON reports include both `status` and `support_level` so users can filter techniques safely.
