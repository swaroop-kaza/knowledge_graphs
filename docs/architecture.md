# Repository architecture

`kg_construction` is a **hexagonal (ports-and-adapters)** framework that turns structured and unstructured data into evidence-grounded knowledge graphs.

- **The graph shape is decided inside** — identities, relationships, and provenance come from an inner domain contract, not from any tool.
- **Every external technology is a replaceable adapter** — a loader, an NER model, Neo4j, RDF, or a metric sits behind a small protocol.
- **Storage is a projection, never the source of truth** — see the pillars below.

This document is the map. The technique write-ups drill into each layer:

- [Structured data → KG techniques](techniques_structured_to_kg.md)
- [Unstructured data → KG techniques](techniques_unstructured_to_kg.md)
- [Evaluation metrics, weighting, and thresholds](evaluation_metrics.md)
- [Testing every technique](testing_guide.md)
- [Modelling doctrine and stage gates](knowledge_graph_modelling.md)

---

## 1. The three pillars

- **Three pillars, nothing else** — every module serves structured→graph, unstructured→graph, or evaluation.
- **Storage is downstream** — Neo4j, RDF, Memgraph, Ladybug, and vector indexes are projections of a correctly formed model.
- **A store cannot repair bad formation** — wrong identities or relationships stay wrong no matter where they are written.

```mermaid
flowchart LR
    subgraph P1["Pillar 1 — Structured → Graph"]
        S1[CSV / Excel / SQL / JSON / XML]
        S1 --> SM[Semantic profiling<br/>+ resolved plan<br/>+ source catalog]
    end
    subgraph P2["Pillar 2 — Unstructured → Graph"]
        U1[Text / HTML / PDF / email / chat]
        U1 --> UM[Evidence-first<br/>11-stage pipeline]
    end
    SM --> G[(Property graph<br/>RDF graph<br/>Evidence graph)]
    UM --> G
    subgraph P3["Pillar 3 — Evaluation & Gates"]
        G --> F[Formation metrics]
        G --> T[Task metrics]
        G --> IU[Intended-use metrics]
        F --> GATE{Hard gates}
        T --> GATE
        IU --> GATE
    end
    GATE -->|pass| PUB[Published graph + certificate]
    GATE -->|fail| Q[Quarantine / reject]
```

| Pillar | Question it answers | Canonical modules |
| --- | --- | --- |
| 1. Structured → graph | Are entities identified once and related correctly, with no duplicated literals? | `application/semantic_profiler.py`, `application/resolved_plan.py`, `application/source_catalog.py`, `application/semantic_table_mapper.py` |
| 2. Unstructured → graph | Is every promoted claim backed by an exact span and complete provenance? | `application/unstructured_pipeline.py`, `domain/claim.py`, `infrastructure/extractors/model_backed/` |
| 3. Evaluation & gates | Is the graph intrinsically well-formed *and* fit for its declared use? | `infrastructure/evaluation/`, `application/scoring.py`, `certification/` |

---

## 2. Hexagonal layers

The package is split into four rings with **dependencies pointing inward only**:

- **Domain** knows nothing about Neo4j or GLiNER.
- **Infrastructure adapters** depend on the ports, not the reverse.
- **Application** orchestrates the ports; it never imports a concrete adapter directly.

```mermaid
flowchart TB
    subgraph interfaces["interfaces/ — CLI entry points"]
        CLI[construct · benchmark · doctor · replay_dead_letters]
    end
    subgraph application["application/ — orchestration"]
        WF[ConstructionWorkflow]
        REG[PluginRegistry]
        PROF[Dataset & semantic profilers]
        SCORE[ScoringEngine / recommendation]
        API[public_api: construct_kg / evaluate_techniques]
    end
    subgraph ports["ports/ — Protocol interfaces"]
        P1[SourceLoader]
        P2[SourceMapper]
        P3[EntityExtractor / RelationExtractor]
        P4[EntityResolver]
        P5[GraphValidator]
        P6[GraphStore]
        P7[ConstructionEvaluator]
    end
    subgraph domain["domain/ — Pydantic contracts"]
        D1[SourceRecord · EvidenceRef]
        D2[EntityCandidate · RelationCandidate · GraphFact]
        D3[Claim · Mention · Provenance]
        D4[PropertyGraph · RDFGraph · GraphSchema]
        D5[ModelingPlan · OntologySchema]
    end
    subgraph infrastructure["infrastructure/ — adapters"]
        I1[loaders]
        I2[mapping]
        I3[extractors]
        I4[resolution]
        I5[ontology / SHACL]
        I6[graph_builders]
        I7[graph_stores]
        I8[evaluation]
        I9[production]
    end
    CLI --> application
    application --> ports
    infrastructure -. implements .-> ports
    application --> domain
    infrastructure --> domain
```

| Layer | Directory | Responsibility | Depends on |
| --- | --- | --- | --- |
| Domain | `src/kg_construction/domain/` | Immutable Pydantic models and modelling contracts; no I/O | nothing |
| Ports | `src/kg_construction/ports/` | `Protocol` interfaces (loader, mapper, extractor, resolver, validator, store, evaluator, enricher) | domain |
| Application | `src/kg_construction/application/` | Workflow orchestration, registry, profiling, scoring, public API | domain, ports |
| Infrastructure | `src/kg_construction/infrastructure/` | Concrete adapters for every external technology | domain, ports |
| Interfaces | `src/kg_construction/interfaces/cli/` | `construct`, `benchmark`, `doctor`, `replay_dead_letters` console scripts | application |

---

## 3. Core data flow

The same domain vocabulary flows through both pillars:

- **One vocabulary, two pillars** — structured and text inputs converge on the same objects.
- **Everything carries evidence** — candidates hold an `EvidenceRef` before they are ever promoted.
- **Three graph views out** — property, RDF, and evidence graphs from one `ConstructionBatch`.

```mermaid
flowchart LR
    A[Source data] --> B[SourceLoader]
    B --> C[SourceRecord]
    C --> D{Structured<br/>or text?}
    D -->|structured| E[SemanticTableMapper<br/>via ResolvedModelingPlan]
    D -->|text| F[UnstructuredPipeline<br/>mentions → claims]
    E --> G[ConstructionBatch<br/>entities · relations · facts]
    F --> G
    G --> H[EntityResolver<br/>canonicalise / link]
    H --> I[GraphValidator<br/>evidence · schema · SHACL]
    I --> J[Graph builders]
    J --> K[(PropertyGraph)]
    J --> L[(RDFGraph)]
    J --> M[(EvidenceGraph)]
    K & L & M --> N[Evaluators + gates]
    N --> O[Reports · rankings · certificate]
```

Key domain objects (`domain/models.py`, `domain/claim.py`, `domain/graph_objects.py`, `domain/modeling.py`):

| Object | Role |
| --- | --- |
| `SourceRecord` | One immutable loaded record with a content-derived identity |
| `EvidenceRef` | Exact span/selector + source id + extractor metadata that backs any candidate |
| `EntityCandidate` / `RelationCandidate` / `GraphFact` | Extraction observations before promotion |
| `Claim` / `Mention` / `Provenance` / `TextSpan` | The reified, evidence-bound unstructured contract |
| `ConstructionBatch` | The bundle handed between workflow stages |
| `ModelingPlan` → `ResolvedModelingPlan` | Proposed then resolved column semantics for structured data |
| `OntologySchema` | The single source projected into 6 schema artifacts |
| `PropertyGraph` / `RDFGraph` / `EvidenceGraph` | The three materialized graph views |

---

## 4. The technique registry

- **Technique-centered** — many construction techniques are registered side by side, then ranked and recommended.
- **Two registry views** (`application/registry.py`) — `research_registry()` (the full inventory) and `default_registry()` (the mandatory release surface).

```mermaid
flowchart TB
    RR[research_registry<br/>complete inventory] --> filt{status == implemented<br/>AND category != enrichment<br/>AND has adapter}
    filt -->|yes| DR[default_registry<br/>mandatory release surface]
    filt -->|no| OUT[research inventory + roadmap]
```

| Registry view | Count (live) | Meaning |
| --- | ---: | --- |
| `research_registry()` total | 165 | Every implemented, adapter, and roadmap entry — comparative research surface |
| — implemented | 123 | Working technique with a class |
| — adapter | 15 | Wraps an external tool; kept for comparison |
| — stub | 27 | Roadmap extension point, not executable |
| `default_registry()` | 105 | Mandatory, certifiable release capabilities only |

Public maturity labels (via `TechniqueDescriptor.public_support_label`):

| Maturity label | Count | Meaning |
| --- | ---: | --- |
| Mandatory | 96 | Part of the executable release contract; no skip or fallback |
| Research inventory | 42 | Implemented/adapted outside the release contract; kept for comparison |
| Roadmap | 27 | Non-executable extension point, outside the release registry |

Registered categories (research view):

| Category | Techniques | Category | Techniques |
| --- | ---: | --- | ---: |
| `extractor` | 27 | `validation` | 9 |
| `mapper` | 25 | `schema` | 8 |
| `enrichment` | 24 | `hybrid` | 8 |
| `loader` | 20 | `evaluation` | 6 |
| `resolution` | 16 | `graph_builder` | 3 |
| `graph_store` | 9 | `llm` | 1 |
| `production` | 9 | | |

> Counts drift as the code evolves — reproduce them with `make registry-reports` or the `Counter` snippet in [testing_guide.md](testing_guide.md).

---

## 5. The mandatory "hard pass" (certification)

The release contract is **zero-skip**:

- **Credit requires execution** — a capability counts only when `declared = configured = instantiated = executed = evaluated = passed`.
- **Importability earns nothing** — being present in the codebase is not a pass.
- **No silent fallback** — a skipped provider or a downgraded device fails the run.

```mermaid
flowchart LR
    L[release-locks<br/>uv.lock · models.lock · services.lock] --> S[make setup<br/>hydrate models + services]
    S --> D[doctor --strict<br/>ok/fail only]
    D --> C[run_all_approaches.sh]
    C --> OFF[Disconnect package/model/container registries]
    OFF --> CERT[Per-capability certifier<br/>fresh nonce + fixture]
    CERT --> V{semantic metrics<br/>+ named assertions?}
    V -->|yes, no fallback| PASS[Certificate]
    V -->|skip/fallback| FAIL[Hard fail]
```

| Artifact | Produced by | Purpose |
| --- | --- | --- |
| `uv.lock` | `uv lock` | Every Python distribution pinned |
| `models.lock.yaml` | `scripts/lock_models.py` | Model commit + per-file SHA-256 + device + license |
| `certification/services.lock.yaml` | `scripts/lock_service_digests.py` | Container images by immutable digest |
| `capability_manifest.yaml` | `scripts/generate_capability_manifest.py` | Registry ↔ owner bijection, doctrine split |
| `certification/execution_plan.yaml` | maintained | Per-capability semantic certifier contract |

Model/platform profiles live in `models.manifest.yaml`:

- **Three profiles** — `darwin-arm64` (MPS + MLX), `linux-cpu`, `linux-cuda`.
- **Every profile mandates the same models** — GLiNER, Transformers, spaCy, sentence-transformers, and Qwen3 (Transformers + Ollama; MLX added on Apple Silicon).
- **Every profile mandates the same services** — Ollama, Neo4j, Memgraph, and embedded Ladybug.

---

## 6. Where to find things

```text
src/kg_construction/
  domain/           models, claim, graph_objects, modeling, normalization, schema, evidence
  ports/            loaders, mappers, extractors, resolvers, validators, graph_store, evaluators, enrichers
  application/      workflow, registry, semantic_profiler, resolved_plan, source_catalog,
                    semantic_table_mapper, plan_resolution, scoring, recommendation_engine,
                    benchmark_runner, public_api, unstructured_pipeline, modelling_correctness
  infrastructure/
    loaders/        csv, json, text, excel, sql, api, xml, pdf, email, chat, html, log, ...
    mapping/        direct_table, semi_structured (JSONPath), standards (R2RML/RML/CSVW/JSON-LD/OBDA)
    extractors/     regex, gazetteer, relations, ollama_graph, gliner; model_backed/ (ensembles)
    resolution/     exact, canonicalization, entity_resolution, linking, mandatory_ensemble
    ontology/       schema_proposal, structured_ontology, shacl, ontology_alignment, unstructured_governance
    graph_builders/ property_graph, rdf_graph, evidence_graph
    graph_stores/   neo4j/vector, memgraph, ladybug, networkx, rdf exporters + ingestion
    evaluation/     metrics, accuracy, coverage, validation, entity_resolution, retrieval,
                    schema_quality, intended_use_metrics, quality_gates, claim_evaluations, graph_bound_suite
    production/     logging, manifests, watermarks, retry, dead_letter, quarantine, review, pii
  interfaces/cli/   construct, benchmark, doctor, replay_dead_letters
  certification/    runner, execution, capabilities, profiles, preflight
```
