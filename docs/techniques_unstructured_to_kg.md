# Unstructured data → knowledge graph: techniques explained

Technical reference for **Pillar 2** — converting text (and HTML, PDF, email, chat, scientific papers) into entities, claims, and events.

- **Every promoted fact quotes an exact source span.**
- **Every promoted fact carries complete provenance.**

The governing rule of this pillar:

- **Keep four identity layers separate** (below).
- **Never force an uncertain mention** into an existing entity.

```mermaid
flowchart LR
    D[Source record] --> M[Mention<br/>one textual occurrence]
    M --> O[Observed / coreference identity<br/>document-local cluster]
    O --> CE[CanonicalEntity<br/>resolver / linker identity]
    CE --> X[External identity<br/>Wikidata / DBpedia QID]
```

Contents:

1. [The evidence-first pipeline](#1-the-evidence-first-pipeline)
2. [The claim contract](#2-the-claim-contract-why-facts-are-reified)
3. [Stage-by-stage techniques](#3-stage-by-stage-techniques)
4. [The mention ensemble](#4-the-mention-ensemble-technique-detail)
5. [The claim / relation / event ensemble](#5-the-claim--relation--event-ensemble)
6. [Attribute extraction](#6-attribute-extraction-negation-modality-temporal-quantity-attribution)
7. [Identity resolution and linking](#7-identity-resolution-and-external-linking)
8. [Ontology alignment and the promotion gate](#8-ontology-alignment-and-the-shaclprovenance-promotion-gate)
9. [Runtime profiles: deterministic vs model-backed](#9-runtime-profiles-deterministic-vs-model-backed)
10. [Legacy / comparison extractors](#10-legacy--comparison-extractors)

---

## 1. The evidence-first pipeline

**File:** `application/unstructured_pipeline.py` · **Registry:** `evidence_first_unstructured_pipeline`, `mandatory_composed_unstructured_runtime`

- **Eleven ordered stages** — `UnstructuredPipeline.run()` runs them in sequence.
- **Observations stay immutable** — nothing is rewritten in place.
- **Promotion is gated** — a claim promotes only after its offsets, document version, provider provenance, confidence, and ontology predicate all pass.

```mermaid
flowchart TD
    U0[U0 Document version<br/>DocumentRef: digest + version] --> U1[U1 Segmentation<br/>paragraphs/sentences + exact offsets]
    U1 --> U2[U2 Mention ensemble<br/>GLiNER + Transformers NER + spaCy + rules]
    U2 --> U3[U3 Coreference<br/>document-local clusters]
    U3 --> U4[U4 Claims/relations/events<br/>patterns + GLiNER RelEx + Qwen3]
    U4 --> U4b[U4b Attributes<br/>negation/modality/temporal/quantity/attribution]
    U4b --> U5a[Identity resolution<br/>exact + embedding + cross-encoder + linking]
    U5a --> U5b[Ontology alignment<br/>predicate on governed schema?]
    U5b --> U5c[U5 Promotion gate<br/>exact evidence + provenance + SHACL]
    U5c -->|pass| U6[U6 Layered materialization<br/>doc · mention · identity · claim · event]
    U5c -->|fail| Q[Quarantine]
```

Each stage has a hard exit gate (detailed in [knowledge_graph_modelling.md](knowledge_graph_modelling.md), Pillar 2). The two that matter most:

- **Segments must reproduce the source exactly** — `document_text[start:end]` (start inclusive, end exclusive).
- **Coverage must be perfect** — exact-evidence and provenance coverage both `= 1.0` in the release fixture.

---

## 2. The claim contract (why facts are reified)

**File:** `domain/claim.py`

- **Facts are not flattened prematurely** — no bare `(subject, predicate, object)` edge up front.
- **A `Claim` is immutable and reified** — it preserves the qualifiers a plain edge would lose.

```mermaid
classDiagram
    class Claim {
        +subject
        +predicate
        +object
        +kind (relation/fact/intent/event)
        +event_roles
        +negated
        +modality (asserted/possible/probable/hypothetical)
        +temporal
        +quantity
        +attribution
        +confidence
        +review_state
        +has_exact_evidence()
        +has_complete_provenance()
    }
    class Evidence {
        +span (TextSpan)
        +prefix_suffix
        +document_id
        +document_version
        +kind
    }
    class Provenance {
        +extractor
        +provider
        +model_version
        +configuration_id
        +confidence
    }
    Claim "1" --> "*" Evidence
    Claim "1" --> "1" Provenance
    Claim "1" --> "*" EventRole
```

`has_complete_provenance()` is the promotion gate. A claim is promotable **only when all** of the following hold:

- **document digest matches**;
- **every evidence and semantic span is exact**;
- **extractor / provider / model-revision / configuration / confidence all exist**;
- **confidence meets policy**; and
- **the predicate is on the governed ontology**.

Standards alignment:

- **Spans** — `TextSpan` uses end-exclusive offsets aligned to the [W3C Web Annotation selector model](https://www.w3.org/TR/selectors-states/).
- **Provenance** — follows [PROV-O](https://www.w3.org/TR/prov-o/).

---

## 3. Stage-by-stage techniques

| Stage | Deterministic default | Model-backed (certified) | Technique |
| --- | --- | --- | --- |
| U0 doc version | `DocumentRef.from_source` | same | Content digest + version id; a changed doc = a new version |
| U1 segmentation | `RuleSegmenter` | same | Split into sentences/paragraphs retaining absolute offsets + quote/prefix/suffix |
| U2 mentions | `RuleMentionDetector` | `MandatoryMentionEnsemble` | Union of specialist recognizers with provider-agreement ratios |
| U3 coreference | `ConservativeCoreferenceResolver` | same/statistical | Safe exact clustering; ambiguous stays NIL |
| U4 claims | `PatternClaimExtractor` | `MandatoryClaimEnsemble` | Subject-predicate-object + event roles with exact cue spans |
| U4b attributes | `RuleAttributeExtractor` | same | Negation / modality / temporal / quantity / attribution cues |
| U5 identity | `KeepUnresolvedResolver` / `ExplicitNilObservedResolver` | `ProviderIdentityResolver` + `MandatoryCanonicalIdentityEnsemble` | Merge only safe cases; keep NIL otherwise |
| U5 ontology/SHACL | `GovernedClaimShaclGate` | same | Predicate must align to governed schema; RDF must conform |
| U6 materialization | builders | builders | Emit doc/mention/identity/claim/event as distinct resources |

---

## 4. The mention ensemble (technique detail)

**Files:** `infrastructure/extractors/model_backed/mentions.py`, `infrastructure/extractors/{regex,gazetteer}.py`

- **All providers run** — every applicable profile runs each one; the union is kept with provider-agreement ratios.
- **Disagreement is information** — not a reason to erase a supported mention.
- **Failure is aggregated** — a mandatory provider failure fails the stage *after* the others have been attempted.

```mermaid
flowchart LR
    T[Segment text] --> A[Deterministic patterns<br/>date/email/version/ID regex + gazetteer]
    T --> B[GLiNER zero-shot NER<br/>gliner_small-v2.5]
    T --> C[Transformers token-classification<br/>dslim/bert-base-NER]
    T --> D[spaCy statistical NER<br/>en_core_web_sm]
    T --> E[Declared domain labels<br/>from config]
    A & B & C & D & E --> U[Offset-validated union<br/>+ ProviderAgreement ratios]
```

| Provider | Model / basis | Why it is a *separate* mandatory provider |
| --- | --- | --- |
| `RuleMentionDetector` | regex + gazetteer | Exact structured entities (dates, emails, IDs, versions) |
| `GlinerMentionDetector` | `gliner-community/gliner_small-v2.5` | Open-label, parallel zero-shot NER ([NAACL 2024](https://aclanthology.org/2024.naacl-long.300/)) — not an LLM synonym |
| `TransformersMentionDetector` | `dslim/bert-base-NER` | Supervised token classification |
| `SpacyMentionDetector` | `spacy/en_core_web_sm` | Statistical NER, CPU-only |

- **Exact-span evaluation is separate from label/name evaluation** — by design.
- **Matching key** — `(document version, start, end, type, occurrence)`, so a boundary error cannot hide behind entity F1.
- **NIL is reported explicitly** — motivated by the [EMNLP 2023 in-depth entity-linking evaluation](https://aclanthology.org/2023.emnlp-main.411/).

---

## 5. The claim / relation / event ensemble

**Files:** `infrastructure/extractors/model_backed/{claims,relations}.py`

Three independent extraction paths run so the contract never depends on one model family or one decoding path:

```mermaid
flowchart LR
    S[Segment + mentions] --> P[Deterministic patterns<br/>PatternClaimExtractor]
    S --> G[GLiNER RelEx<br/>gliner-relex-large-v1.0]
    S --> Q[Structured Qwen3 generation]
    Q --> Q1[Transformers Qwen3-1.7B<br/>all profiles]
    Q --> Q2[MLX Qwen3-1.7B-4bit<br/>Apple Silicon]
    Q --> Q3[Ollama qwen3:1.7b<br/>all profiles]
    P & G & Q1 & Q2 & Q3 --> C[Claims with exact cue spans]
```

- **Rich response schema** — `StructuredClaimResponse` covers subject, predicate, object, event roles, evidence, negation, modality, attribution, time, quantity, and confidence.
- **Values and cues require exact spans.**
- **Shape rules** — relation/fact/intent claims require an object; events require typed roles.

| Path | Basis | Role |
| --- | --- | --- |
| `PatternClaimExtractor` | `_RELATION_PATTERNS` regex (affected, uses, depends_on, located_in, part_of, reports_to, contains…) | Deterministic, offline-testable floor |
| `GlinerRelExClaimExtractor` | `knowledgator/gliner-relex-large-v1.0` | Joint entity+relation extraction ([GLiNER-Relex](https://arxiv.org/abs/2605.10108)) |
| `TransformersQwenClaimExtractor` / `MlxQwenClaimExtractor` / `OllamaQwenClaimExtractor` | Qwen3-1.7B | Structured-schema LLM generation across three independent backends |

- **Evaluation is document-level, not sentence-level** — relations cross sentence boundaries ([DocRED](https://aclanthology.org/P19-1074/)).

> **License note:** the earlier REBEL relation extractor was removed from the product registry (its `cc-by-nc-sa` license is non-commercial). GLiNER RelEx (Apache-2.0) replaces it. The `TransformerRelationExtractor` class remains for research but is de-registered/de-certified.

---

## 6. Attribute extraction (negation, modality, temporal, quantity, attribution)

**File:** `application/unstructured_pipeline.py` (`RuleAttributeExtractor`)

Each claim is annotated with qualifier cues, each requiring an exact span:

| Attribute | Cue pattern (`re`) | Example → value |
| --- | --- | --- |
| negation | `no, not, never, without, failed to, didn't, n't` | "did not deploy" → `negated=true` |
| modality | possible: `may/might/could/possibly` · probable: `likely/probably` · hypothetical: `if/assuming/suppose` | "might affect" → `Modality.possible` |
| temporal | ISO dates, `in/during/before/after/since/until <period>`, `yesterday/today/tomorrow` | "since 2024" → temporal scope |
| quantity | `\d+ (%|kg|km|hours|days|items|units…)` | "40 hours" → quantity |
| attribution | `according to / reported by / stated by <Source>` | "according to Alice" → attribution |

- **Why qualifiers matter** — they make a **reified claim** more faithful than a bare edge: "X did *not* cause Y, *reportedly*, in *2023*" survives intact.

---

## 7. Identity resolution and external linking

**Files:** `infrastructure/extractors/model_backed/identity.py`, `infrastructure/resolution/`

The certified `MandatoryCanonicalIdentityEnsemble` votes across providers and keeps NIL explicit rather than forcing a merge:

```mermaid
flowchart LR
    M[Mention clusters] --> EX[ExactClusterIdentityProvider<br/>safe exact matches]
    M --> EM[LockedEmbeddingIdentityProvider<br/>all-MiniLM-L6-v2]
    M --> CR[LockedCrossEncoderIdentityProvider<br/>stsb-roberta-base]
    EX & EM & CR --> V[IdentityVote ensemble]
    V --> L{external link?}
    L --> W[WikidataLinker]
    L --> DB[DBpediaLinker]
    L --> RC[OpenRefine-compatible reconciliation<br/>Wikimedia Cloud API]
    V --> NIL[NilIdentityPolicy: unresolved stays observed]
```

| Layer | Adapters | Technique |
| --- | --- | --- |
| deterministic | `ExactKeyEntityResolver`, `NormalizedStringResolver`, `AliasTableResolver` | Exact/normalized/alias key matching |
| fuzzy | `RapidFuzzResolver` | Edit-distance similarity |
| ML / embedding | `EmbeddingResolver`, `CrossEncoderResolver`, `Splink`, `Dedupe` | Vector similarity, probabilistic linkage, active learning |
| external | `WikidataLinker`, `DBpediaLinker`, `OpenRefineReconciler` | Link to a public KB via the [Reconciliation Service API](https://reconciliation-api.github.io/specs/latest/) |

- **Ambiguity stays NIL** — unresolved cases can be quarantined, never forced to the nearest name.
- **ER quality is measured** — pairwise/cluster F1, false-merge rate, false-split rate, NIL accuracy, and record-universe coverage (see [evaluation_metrics.md](evaluation_metrics.md)).

---

## 8. Ontology alignment and the SHACL/provenance promotion gate

**Files:** `infrastructure/ontology/unstructured_governance.py`, `infrastructure/ontology/ontology_alignment.py`, `infrastructure/ontology/shacl.py`

```mermaid
flowchart TD
    C[Annotated + identity-bound claim] --> A{predicate on<br/>governed ontology?}
    A -->|no| OFF[off_ontology → quarantine]
    A -->|yes| E{exact evidence<br/>+ complete provenance?}
    E -->|no| Q[quarantine]
    E -->|yes| S{SHACL conforms?}
    S -->|no| Q
    S -->|yes| PROMOTE[promoted claim]
```

- **SHACL runs on the generated RDF** — via `GovernedClaimShaclGate`.
- **Four ways to be quarantined** — off-ontology predicate, inexact span, missing provenance, or a SHACL violation; none are silently written.
- **Schema is declared in config** — e.g. `configs/governed_unstructured_ontology.yaml`.

---

## 9. Runtime profiles: deterministic vs model-backed

`RuntimeProfile` selects which providers run:

| Profile | Providers | Use |
| --- | --- | --- |
| `DETERMINISTIC` (default) | Rule segmenter/mention/coref/claim/attribute + `KeepUnresolvedResolver` | Runs fully offline, no models — the testable contract floor |
| `MODEL_BACKED` (certified) | GLiNER, Transformers NER, spaCy, GLiNER RelEx, Qwen3 (Transformers + Ollama + MLX on Apple Silicon), embedding/cross-encoder identity, external linking | The release/hard-pass path |

- **Both profiles satisfy the same protocols** — `Segmenter`, `MentionDetector`, `CoreferenceResolver`, `ClaimExtractor`, `AttributeExtractor`, `IdentityResolver`, `EntityLinkProvider`.
- **Deterministic proves the contract offline.**
- **Model-backed fulfils it for real documents.**

---

## 10. Legacy / comparison extractors

Kept in `research_registry()` for side-by-side comparison, not part of the mandatory release surface:

| Technique | Basis | Status |
| --- | --- | --- |
| `openie_relation_extractor` | Open Information Extraction triples | implemented |
| `dependency_pattern_relation_extractor` | Dependency-parse patterns | implemented |
| `ollama_graph_extractor` | Single-shot LLM graph extraction | implemented (optional adapter) |
| `gliner_ollama_relation_typing_extractor` | GLiNER spans → Ollama relation typing | implemented (hybrid) |
| `transformer_relation_extractor` | Supervised transformer RE | adapter (de-certified) |
| `spacy_entity_ruler_extractor`, `coreference_adapter`, `pos_tagging_adapter` | spaCy components | stub/roadmap |

See runnable commands in the [README run guide](../README.md#running-every-technique) and how to test each in the [testing guide](testing_guide.md).
