# Documentation map

The documentation follows the same boundary as the product: two construction pillars, one evaluation pillar, and separate supporting runtime/research material.

## Choose a path

| Goal | Start with | Continue with |
| --- | --- | --- |
| Understand the repository | [Architecture](architecture.md) | [Knowledge-graph modelling](knowledge_graph_modelling.md) |
| Build a first graph | [Getting started](getting_started.md) | [Configuration](configuration.md) |
| Model structured data | [Structured construction](techniques_structured_to_kg.md) | [Stage-by-stage implementation](stage_by_stage_implementation.md) |
| Model unstructured data | [Unstructured construction](techniques_unstructured_to_kg.md) | [Evaluation metrics](evaluation_metrics.md) |
| Verify or release a change | [Testing guide](testing_guide.md) | [Certification contract](../certification/README.md) |

## Start here

- [Architecture](architecture.md) — the real package boundaries, data flow, graph projections, and the distinction between draft conversion and certification.
- [Knowledge-graph modelling](knowledge_graph_modelling.md) — the identity, relationship, evidence, ontology, and evaluation invariants.
- [Stage-by-stage implementation](stage_by_stage_implementation.md) — the ordered implementation and release gates.

## Pillar 1: structured construction

- [Structured data → knowledge graph](techniques_structured_to_kg.md) — semantic roles, materialization, identity scope, categorical concept reuse, foreign keys, associations/events, and standards mappings.

Canonical namespace: `kg_construction.structured`.

## Pillar 2: unstructured construction

- [Unstructured data → knowledge graph](techniques_unstructured_to_kg.md) — immutable evidence, mandatory provider ensembles, coreference, semantic assertions, qualifiers, identity resolution, ontology alignment, and SHACL.

`Claim` is the implementation name for an evidence-grounded semantic assertion. It does not introduce a business domain.

Canonical namespace: `kg_construction.unstructured`.

## Pillar 3: evaluation

- [Evaluation metrics](evaluation_metrics.md) — formation, task-correctness, and intended-use metrics plus their conjunctive publication policy.
- [Testing guide](testing_guide.md) — focused tests, product smoke, research smoke, runtime checks, and the zero-skip hard pass.

Cross-source identity data is an entity-resolution judgment fixture. Relevance data is an intended-use judgment for vector or GraphRAG workloads. Neither defines the construction schema.

Canonical namespace: `kg_construction.evaluation`.

Evaluation implementations are physically owned by that namespace. The former `kg_construction.infrastructure.evaluation.*` modules preserve named legacy imports only. Their exports are explicit and identity-preserving; there is no wildcard façade from infrastructure to the canonical package.

## Runtime and certification

- [Getting started](getting_started.md) — provisioning the mandatory platform-specific model and service profile.
- [Configuration](configuration.md) — draft settings versus immutable certified settings.
- [Certification contract](../certification/README.md) — locked inputs, per-capability evidence, and the zero-skip/zero-fallback decision.

Every provider applicable to the selected certification profile is mandatory. MLX is required on Apple Silicon and inapplicable—not skipped—on Linux.

## Research boundary

Simple benchmark runs, weighted technique rankings, leaderboards, synthetic cross-source comparisons, Graph ML experiments, and mocked provider demonstrations are non-authoritative research. They may help compare approaches, but they never establish publishability or replace product evaluation.

Canonical namespace: `kg_construction.research`.

When adding documentation, link it from this page and identify whether it describes product construction, product evaluation, certification, or research.
