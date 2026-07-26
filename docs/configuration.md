# Configuration reference

[Documentation home](README.md) · [Getting started](getting_started.md) · [Evaluation](evaluation_metrics.md) · [Testing](testing_guide.md)

Configuration has three distinct scopes. Do not mix them:

| Scope | Purpose | Authority |
| --- | --- | --- |
| Draft conversion | Select input mapping and inspection behavior | User-configurable; never establishes publishability |
| Certified product runtime | Bind models, services, ontology, gold data, and gates | Repository-owned, immutable release contract |
| Research comparison | Rank or compare candidate techniques | Non-authoritative research only |

## On this page

1. [Draft conversion configuration](#1-draft-conversion-configuration)
2. [Certified product configuration](#2-certified-product-configuration)
3. [Research configuration](#3-research-configuration)
4. [Configuration examples](#4-configuration-examples)
5. [Related guides](#related-guides)

## 1. Draft conversion configuration

`convert(data, source_format, technique, config=...)` accepts a `DataToKGConfig` object or YAML path and always returns draft graphs plus metrics.

```python
from kg_construction import DataToKGConfig, convert

config = DataToKGConfig.from_yaml("configs/convert_configurable.example.yaml")
result = convert("notes.txt", "text", config=config)

print(result.quality_report)
```

Draft settings are useful for inspection, ablation, and schema development. They do not certify provider participation and cannot make a graph release-eligible.

The canonical feature ownership is:

```text
kg_construction.structured      structured mapping settings
kg_construction.unstructured    extraction and evidence settings
kg_construction.evaluation      metric observations
kg_construction.research        comparison weights and ranking
```

### Technique selection

The `technique` argument (or CLI `--technique`) selects a draft converter:

| Source | Product construction technique |
| --- | --- |
| CSV, Excel, SQL | `semantic_table_mapper` |
| JSON, API response | `jsonpath_mapper` |
| JSON-LD | `jsonld_mapper` |
| event payload | `event_graph_mapper` |
| logs | `log_event_mapper` |
| text and document-derived text | `evidence_first_unstructured_pipeline` |

Direct table and dotted-path mappers remain useful baselines or explicitly chosen low-level mappings. They are not allowed to masquerade as semantic product construction when they violate the formation contract.

### Structured declarations

`structured.tables[]` owns the modelling declarations that cannot be safely guessed:

| Field | Meaning |
| --- | --- |
| `table_name` / `record_type` | source table identity and graph class |
| `primary_key` / `alternate_keys` | business identity |
| `identity_scope` | source, governed global, or explicit cross-source resolution |
| `foreign_keys[]` | target table/key, direction, and relationship name |
| `datatypes` | source field → graph datatype |
| `column_mappings` | semantic role and node/edge/property materialization |
| `controlled_vocabularies[]` | concept scheme, values, aliases, and open/closed policy |
| `table_kind` | entity, event, association, or pure edge |

Declarations are authoritative. Profiling may propose missing semantics, but its confidence and evidence stay visible.

### Draft extraction settings

`extraction` can select deterministic inspection behavior or the immutable model-backed bundle. A draft model run is still a draft: it cannot prove that the complete platform provider set executed.

Important fields include:

| Field | Meaning |
| --- | --- |
| `runtime_profile` | draft execution profile |
| `entity_labels` | governed labels requested from recognizers |

There are deliberately no per-provider switches in this section. With `model_backed`, the immutable model lock selects every platform-applicable GLiNER, GLiNER RelEx, Transformers, spaCy, Qwen3, embedding, and cross-encoder provider; a missing provider fails instead of changing the profile. Certification independently proves the complete bundle from `models.lock.yaml`.

## 2. Certified product configuration

The certified profile is represented by [`configs/data_to_kg_complete.example.yaml`](../configs/data_to_kg_complete.example.yaml) and bound to repository-owned policy. Its capability entries are assertions, not feature switches.

The trusted inputs include:

- `uv.lock`;
- `models.manifest.yaml` and `models.lock.yaml`;
- `certification/services.lock.yaml`;
- `src/kg_construction/resources/service-runtime.env`;
- `src/kg_construction/resources/publication_contracts_v1.yaml`;
- the package-owned governed ontology and held-out gold;
- the exact source and deterministic post-transformation snapshot;
- the evaluation-suite bytes; and
- the graph-builder contract version.

The caller cannot disable a required provider, relax a repository-owned threshold, replace held-out gold, or self-authorize a new suite.

### Platform applicability

| Profile | Mandatory model execution |
| --- | --- |
| `darwin-arm64` | Torch MPS, GLiNER, GLiNER RelEx, Transformers NER/Qwen3, spaCy, MLX Qwen3, Ollama Qwen3, embedding and cross-encoder identity |
| `linux-cpu` | Torch CPU plus the same provider surface except Darwin-only MLX |
| `linux-cuda` | CUDA plus the same provider surface except Darwin-only MLX |

All profiles also require Ollama, Neo4j, Memgraph, Ladybug, RDF, and declared live identity/reconciliation endpoints. MLX is inapplicable on Linux; it is not counted as skipped.

### Service configuration

`src/kg_construction/resources/service-runtime.env` is the single local service contract used by setup, doctor, construction workloads, and certification. Local overrides must match that contract; conflicting or empty values fail closed.

`.env.example` documents the variable names for local tooling. It does not define a second model or credential policy.

### Evaluation thresholds

Draft threshold settings label returned observations. They never weaken the repository-owned publication policy.

Certified thresholds come from the versioned quality and publication contracts. Every observation records:

- metric and purpose;
- actual value and support;
- operator and threshold;
- whether it executed;
- fixture and graph hashes; and
- pass/fail plus failure reason.

Missing required support or execution is failure, not a passing zero.

## 3. Research configuration

[`configs/evaluation.example.yaml`](../configs/evaluation.example.yaml) configures technique comparison, ranking weights, penalties, and research report behavior. It belongs to `kg_construction.research`.

Weighted scores can compare candidates, but they:

- do not decide publishability;
- do not replace formation or task-correctness gates;
- do not prove live provider participation; and
- must not be described as the product’s final schema score.

Synthetic cross-source identity comparisons and deterministic leaderboard runtime values are also research artifacts. Core entity-resolution metrics remain in `kg_construction.evaluation`.

## 4. Configuration examples

| File | Scope |
| --- | --- |
| `configs/convert_configurable.example.yaml` | draft conversion and inspection |
| `configs/data_to_kg_complete.example.yaml` | complete product contract example |
| `configs/graph_evaluation_suite.sample_csv.yaml` | structured formation and intended-use judgments |
| `configs/graph_evaluation_suite.sample_text.yaml` | unstructured task and intended-use judgments |
| `configs/production_ingestion.example.yaml` | graph-store ingestion settings |
| `configs/evaluation.example.yaml` | research comparison only |

## Related guides

- [Getting started](getting_started.md) explains provisioning and the complete hard pass.
- [Structured construction](techniques_structured_to_kg.md) explains semantic table declarations and materialization.
- [Unstructured construction](techniques_unstructured_to_kg.md) explains the evidence-first model-backed pipeline.
- [Evaluation metrics](evaluation_metrics.md) defines observations, thresholds, and publication policy.
