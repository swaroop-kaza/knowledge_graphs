"""Conjunctive contracts for the mandatory unstructured product runtime."""

from __future__ import annotations

import numpy as np

from kg_construction.application.mandatory_unstructured_runtime import (
    HeldOutClaimCase,
    HeldOutClaimDataset,
    MandatoryUnstructuredRuntime,
)
from kg_construction.application.unstructured_pipeline import (
    ExplicitNilObservedResolver,
    PatternClaimExtractor,
    RuleMentionDetector,
    UnstructuredPipeline,
)
from kg_construction.domain.claim import (
    Evidence,
    EvidenceKind,
    Mention,
    Provenance,
    ResolutionStatus,
    TextSpan,
)
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    RelationCandidate,
    SourceRecord,
    stable_id,
)
from kg_construction.infrastructure.evaluation.claim_evaluations import (
    HardCoverageThresholds,
)
from kg_construction.infrastructure.extractors.model_backed.claims import (
    LocalOllamaQwenBackend,
)
from kg_construction.infrastructure.extractors.model_backed.contracts import (
    HydratedServiceModel,
)
from kg_construction.infrastructure.extractors.model_backed.factory import (
    MandatoryProviderBundle,
)
from kg_construction.infrastructure.extractors.model_backed.identity import (
    ExactClusterIdentityProvider,
    MandatoryCanonicalIdentityEnsemble,
    _cosine,
)
from kg_construction.infrastructure.graph_builders.rdf_graph import RDFGraphBuilder
from kg_construction.infrastructure.ontology.unstructured_governance import (
    GovernedResourceShaclGate,
    GovernedUnstructuredSchema,
    final_rdf_graph_sha256,
)
from kg_construction.infrastructure.resolution.mandatory_ensemble import (
    ExternalIdentityPromotionPolicy,
    GovernedExternalIdentityPromoter,
    MandatoryScopedResolutionEnsemble,
)


class _MentionProvider(RuleMentionDetector):
    provider_ids = ("rule_mention",)


class _ClaimProvider(PatternClaimExtractor):
    provider_ids = ("rule_claim",)


class _RecordingResolver:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, ...]] = []

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        self.calls.append(tuple(entity.name for entity in batch.entities))
        return batch


def test_identity_cosine_accepts_live_numpy_embeddings() -> None:
    assert _cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0


def _entity(
    name: str,
    entity_type: str,
    *,
    identity_layer: str | None = None,
    mention_ids: list[str] | None = None,
) -> EntityCandidate:
    source = SourceRecord.from_text(name, source_name=f"memory://{name}")
    evidence = EvidenceRef.build(
        source,
        location={"kind": "span", "start": 0, "end": len(name)},
        text=name,
    )
    properties: dict[str, object] = {}
    if identity_layer is not None:
        properties["identity_layer"] = identity_layer
    if mention_ids is not None:
        properties["mention_ids"] = mention_ids
    return EntityCandidate.build(
        entity_type=entity_type,
        name=name,
        properties=properties,
        evidence=[evidence],
        confidence=1.0,
        extractor_name="test",
        technique="test",
        schema_version="v1",
    )


def test_governed_alignment_and_real_shacl_run_before_promotion() -> None:
    schema = GovernedUnstructuredSchema.from_package_resource()
    source = SourceRecord.from_text(
        "Alan Turing worked at Bletchley Park.",
        source_name="memory://governed.txt",
    )
    result = UnstructuredPipeline(
        governed_schema=schema,
        identity_resolver=ExplicitNilObservedResolver(),
    ).run(source)

    assert result.promoted
    assert {claim.predicate for claim in result.promoted} == {"works_at"}
    assert result.shacl_executed is True
    assert result.shacl_conforms is True
    assert result.claim_shacl_results
    assert all(item.executed and item.conforms for item in result.claim_shacl_results)
    assert result.governed_schema_sha256 == schema.source_sha256


def test_unmapped_provider_type_is_quarantined_before_coreference() -> None:
    class UnmappedMentionProvider:
        provider_ids = ("unmapped_provider",)

        def detect(self, document, text, segments):
            del segments
            span = TextSpan(start=0, end=len(text), text=text)
            return (
                Mention(
                    document_version=document.version,
                    evidence=Evidence.build(
                        document,
                        span=span,
                        kind=EvidenceKind.MENTION,
                    ),
                    surface=text,
                    entity_type="ProviderInventedType",
                    provenance=Provenance(
                        extractor="test_unmapped",
                        provider="test",
                        model_version="v1",
                        configuration_id="unmapped-v1",
                        confidence=1.0,
                    ),
                ),
            )

    class RecordingCoreference:
        def __init__(self) -> None:
            self.received: tuple[Mention, ...] | None = None

        def cluster(self, mentions):
            self.received = mentions
            return mentions

    coreference = RecordingCoreference()
    source = SourceRecord.from_text(
        "Unmapped",
        source_name="memory://unmapped-type.txt",
    )
    result = UnstructuredPipeline(
        mention_detector=UnmappedMentionProvider(),  # type: ignore[arg-type]
        coreference=coreference,
        identity_resolver=ExplicitNilObservedResolver(),
        governed_schema=GovernedUnstructuredSchema.from_package_resource(),
    ).run(source)

    assert coreference.received == ()
    assert result.mentions == ()
    assert len(result.quarantined_mentions) == 1
    assert result.quarantined_mentions[0].entity_type == "ProviderInventedType"
    assert result.off_ontology_entity_types == ("ProviderInventedType",)
    assert result.entity_type_alignment_results[0]["status"] == "quarantined"


def test_final_resource_shacl_detects_post_external_promotion_type_mutation() -> None:
    schema = GovernedUnstructuredSchema.from_package_resource()
    source = SourceRecord.from_text(
        "Alan Turing worked at Bletchley Park.",
        source_name="memory://post-promotion-mutation.txt",
    )
    batch = UnstructuredPipeline(
        governed_schema=schema,
        identity_resolver=ExplicitNilObservedResolver(),
    ).extract(source)
    observed = next(
        entity
        for entity in batch.entities
        if entity.properties.get("identity_layer") == "observed"
        and entity.properties.get("semantic_entity_type") == "Person"
    )
    linked = observed.model_copy(
        update={
            "properties": {
                **observed.properties,
                "external_links": [
                    {
                        "provider": "wikidata",
                        "id": "Q7251",
                        "label": "Alan Turing",
                        "confidence": 1.0,
                        "review_required": False,
                    },
                    {
                        "provider": "openrefine",
                        "id": "Q7251",
                        "label": "Alan Turing",
                        "confidence": 0.99,
                        "review_required": False,
                    },
                ],
            }
        }
    )
    pre_promotion = batch.model_copy(
        update={
            "entities": [
                linked if entity.candidate_id == observed.candidate_id else entity
                for entity in batch.entities
            ]
        }
    )
    promoted = GovernedExternalIdentityPromoter(
        governed_entity_types=schema.entity_types,
        policy=ExternalIdentityPromotionPolicy(schema_id=schema.schema_id),
    ).promote(pre_promotion)
    gate = GovernedResourceShaclGate(schema)
    _, valid = gate.validate(promoted)
    assert valid.executed and valid.conforms

    canonical = next(
        entity
        for entity in promoted.entities
        if entity.entity_type == "CanonicalEntity"
    )
    mutated = promoted.model_copy(
        update={
            "entities": [
                entity.model_copy(
                    update={
                        "properties": {
                            **entity.properties,
                            "semantic_entity_type": "ProviderInventedType",
                        }
                    }
                )
                if entity.candidate_id == canonical.candidate_id
                else entity
                for entity in promoted.entities
            ]
        }
    )
    _, rejected = gate.validate(mutated)

    assert rejected.executed is True
    assert rejected.conforms is False
    assert rejected.rdf_graph_sha256 != valid.rdf_graph_sha256
    assert "semantic_entity_type" in rejected.results_text


def test_nil_identity_ensemble_retains_observed_identity_and_votes() -> None:
    source = SourceRecord.from_text(
        "Alan Turing worked at Bletchley Park. Turing designed Bombe.",
        source_name="memory://nil.txt",
    )
    pipeline = UnstructuredPipeline(
        identity_resolver=MandatoryCanonicalIdentityEnsemble(
            (ExactClusterIdentityProvider(),),
            schema_id="governed-unstructured-core",
        )
    )
    result = pipeline.run(source)

    assert not result.canonical_entities
    assert result.identity_nil_policy == "retain_observed"
    assert result.identity_nil_cluster_ids
    assert result.identity_decisions
    assert "identity_exact_nfkc" in result.provider_participation_ids
    assert all(mention.canonical_entity_id is None for mention in result.mentions)
    assert all(
        mention.resolution_status
        in {ResolutionStatus.UNRESOLVED, ResolutionStatus.COREFERENCE_ONLY}
        for mention in result.mentions
    )


def test_scoped_resolution_queries_one_representation_per_cluster() -> None:
    canonical = _entity(
        "Alan Turing",
        "Person",
        identity_layer="canonical",
        mention_ids=["m1", "m2"],
    )
    duplicate_observed = _entity(
        "Turing",
        "Person",
        identity_layer="observed",
        mention_ids=["m1", "m2"],
    )
    nil_observed = _entity(
        "Bletchley Park",
        "Place",
        identity_layer="observed",
        mention_ids=["m3"],
    )
    technical = [
        _entity("document", "Document"),
        _entity("mention", "Mention"),
        _entity("claim", "Claim"),
        _entity("event", "Event"),
    ]
    resolvers = tuple(
        _RecordingResolver(name)
        for name in (
            "embedding_resolver",
            "cross_encoder_resolver",
            "wikidata_linker",
            "dbpedia_linker",
            "openrefine_reconciler",
        )
    )
    ensemble = MandatoryScopedResolutionEnsemble(
        resolvers,
        governed_entity_types=("Person", "Place"),
    )
    batch = ConstructionBatch(
        entities=[canonical, duplicate_observed, nil_observed, *technical]
    )

    resolved = ensemble.resolve(batch)

    for resolver in resolvers:
        assert resolver.calls == [("Alan Turing", "Bletchley Park")]
    assert [row["provider_id"] for row in ensemble.last_participation] == [
        *(resolver.name for resolver in resolvers),
        "governed_external_identity_promoter",
    ]
    assert all(row["eligible_entity_count"] == 2 for row in ensemble.last_participation)
    before_technical = {
        item.candidate_id: item for item in technical
    }
    after_technical = {
        item.candidate_id: item
        for item in resolved.entities
        if item.candidate_id in before_technical
    }
    assert after_technical == before_technical


def _with_external_links(
    entity: EntityCandidate,
    *links: tuple[str, str, float, bool],
) -> EntityCandidate:
    return entity.model_copy(
        update={
            "properties": {
                **entity.properties,
                "external_links": [
                    {
                        "provider": provider,
                        "id": identifier,
                        "label": entity.name,
                        "confidence": confidence,
                        "review_required": review_required,
                    }
                    for provider, identifier, confidence, review_required in links
                ],
            }
        }
    )


def _external_promoter() -> GovernedExternalIdentityPromoter:
    return GovernedExternalIdentityPromoter(
        governed_entity_types=("Person",),
        policy=ExternalIdentityPromotionPolicy(schema_id="governed-test"),
    )


def test_external_identity_consensus_creates_canonical_and_links_occurrences() -> None:
    observed = _with_external_links(
        _entity(
            "Alan Turing",
            "Person",
            identity_layer="observed",
            mention_ids=["m1"],
        ),
        ("wikidata", "Q7251", 1.0, False),
        ("openrefine", "Q7251", 0.99, False),
    )
    mention = _entity("Alan Turing", "Mention").model_copy(
        update={
            "properties": {
                "domain_mention_id": "m1",
                "resolution_status": "unresolved",
            }
        }
    )
    denotes = RelationCandidate.build(
        source=mention.candidate_id,
        relation_type="DENOTES_OBSERVED_ENTITY",
        target=observed.candidate_id,
        evidence=mention.evidence,
        extractor_name="test",
        technique="test",
    )

    promoted = _external_promoter().promote(
        ConstructionBatch(entities=[observed, mention], relations=[denotes])
    )

    canonical_id = stable_id("canonical_entity", "wikidata", "Q7251")
    canonical = [
        entity
        for entity in promoted.entities
        if entity.entity_type == "CanonicalEntity"
    ]
    assert len(canonical) == 1
    assert canonical[0].properties["canonical_entity_id"] == canonical_id
    assert canonical[0].properties["external_id"] == "Q7251"
    promoted_observed = next(
        entity for entity in promoted.entities if entity.candidate_id == observed.candidate_id
    )
    assert "canonical_entity_id" not in promoted_observed.properties
    assert promoted_observed.properties["linked_canonical_entity_id"] == canonical_id
    promoted_mention = next(
        entity for entity in promoted.entities if entity.candidate_id == mention.candidate_id
    )
    assert promoted_mention.properties["linked_canonical_entity_id"] == canonical_id
    assert any(
        relation.source == mention.candidate_id
        and relation.relation_type == "DENOTES_OBSERVED_ENTITY"
        and relation.target == observed.candidate_id
        for relation in promoted.relations
    )
    assert any(
        relation.source == mention.candidate_id
        and relation.relation_type == "REFERS_TO_CANONICAL_ENTITY"
        and relation.target == canonical_id
        for relation in promoted.relations
    )
    decision = promoted.metadata["external_identity_promotion"]["decisions"][0]
    assert decision["status"] == "accepted"
    assert set(decision["provider_ids"]) == {"wikidata", "openrefine"}


def test_external_identity_conflict_is_reviewed_and_never_promoted() -> None:
    observed = _with_external_links(
        _entity(
            "Ambiguous Person",
            "Person",
            identity_layer="observed",
            mention_ids=["m1"],
        ),
        ("wikidata", "Q1", 0.99, False),
        ("openrefine", "Q2", 0.99, False),
    )

    promoted = _external_promoter().promote(
        ConstructionBatch(entities=[observed])
    )

    assert not any(
        entity.entity_type == "CanonicalEntity" for entity in promoted.entities
    )
    decision = promoted.metadata["external_identity_promotion"]["decisions"][0]
    assert decision["status"] == "conflict"
    assert decision["review_required"] is True
    assert promoted.metadata["external_identity_review_queue"]
    candidate = promoted.entities[0]
    assert candidate.properties["external_identity_status"] == "conflict"
    assert candidate.properties["review_required"] is True
    assert "linked_canonical_entity_id" not in candidate.properties


def test_single_linker_candidate_remains_explicit_nil_observed_identity() -> None:
    observed = _with_external_links(
        _entity(
            "Unconfirmed Person",
            "Person",
            identity_layer="observed",
            mention_ids=["m1"],
        ),
        ("wikidata", "Q42", 1.0, False),
    )

    promoted = _external_promoter().promote(
        ConstructionBatch(entities=[observed])
    )

    assert not any(
        entity.entity_type == "CanonicalEntity" for entity in promoted.entities
    )
    decision = promoted.metadata["external_identity_promotion"]["decisions"][0]
    assert decision["status"] == "nil"
    candidate = promoted.entities[0]
    assert candidate.properties["identity_layer"] == "observed"
    assert candidate.properties["external_identity_nil"] is True
    assert "linked_canonical_entity_id" not in candidate.properties


def test_ollama_retry_policy_repairs_invalid_structured_output() -> None:
    class Transport:
        def __init__(self) -> None:
            self.responses = iter(
                [
                    {"message": {"content": "not-json"}},
                    {"message": {"content": '{"claims": []}'}},
                ]
            )
            self.chat_calls = 0
            self.payloads = []

        def list_models(self, base_url: str):
            del base_url
            return {"qwen3:test": "a" * 64}

        def chat(self, base_url: str, payload):
            del base_url
            self.chat_calls += 1
            self.payloads.append(payload)
            return next(self.responses)

    transport = Transport()
    model = HydratedServiceModel(
        name="ollama_structured_extractor",
        service="ollama",
        model="qwen3:test",
        digest=f"sha256:{'a' * 64}",
    )
    backend = LocalOllamaQwenBackend(
        model,
        transport=transport,
        retry_on_invalid_output=True,
        max_retries=2,
    )

    output = backend.generate_json(
        prompt="extract",
        json_schema={"type": "object"},
    )

    assert output.claims == ()
    assert transport.chat_calls == 2
    assert backend.last_attempt_count == 2
    assert all(payload["think"] is False for payload in transport.payloads)
    assert all(payload["keep_alive"] == "10m" for payload in transport.payloads)
    assert all(payload["options"]["num_predict"] == 2048 for payload in transport.payloads)
    assert "REPAIR REQUIRED" in transport.payloads[1]["messages"][-1]["content"]


def test_runtime_executes_same_pipeline_on_versioned_gold_and_retains_report() -> None:
    schema = GovernedUnstructuredSchema.from_package_resource()
    identity = MandatoryCanonicalIdentityEnsemble(
        (ExactClusterIdentityProvider(),),
        schema_id=schema.schema_id,
    )
    resolvers = tuple(
        _RecordingResolver(name)
        for name in (
            "embedding_resolver",
            "cross_encoder_resolver",
            "wikidata_linker",
            "dbpedia_linker",
            "openrefine_reconciler",
        )
    )
    resolution = MandatoryScopedResolutionEnsemble(
        resolvers,
        governed_entity_types=schema.entity_types,
    )
    mention_provider = _MentionProvider()
    claim_provider = _ClaimProvider()
    bundle = MandatoryProviderBundle(
        profile="test-profile",
        mention_detector=mention_provider,  # type: ignore[arg-type]
        claim_extractor=claim_provider,  # type: ignore[arg-type]
        identity_resolver=identity,
        resolution_ensemble=resolution,
        mandatory_provider_names=("rule_mention", "rule_claim"),
        mandatory_runtime_participation_ids=(
            "rule_mention",
            "rule_claim",
            "identity_exact_nfkc",
            *resolution.provider_ids,
        ),
        locked_artifact_paths=(),
    )
    pipeline = UnstructuredPipeline(
        mention_detector=mention_provider,
        claim_extractor=claim_provider,
        identity_resolver=identity,
        governed_schema=schema,
    )
    text = "Alan Turing worked at Bletchley Park."
    gold = HeldOutClaimDataset(
        dataset_id="test-gold",
        version="v1",
        description="test",
        thresholds=HardCoverageThresholds(),
        cases=(
            HeldOutClaimCase.model_validate(
                {
                    "case_id": "case-1",
                    "source_name": "held-out://case-1",
                    "text": text,
                    "gold_entities": [
                        {"name": "Alan Turing", "type": "Person"},
                        {"name": "Bletchley Park", "type": "Place"},
                    ],
                    "gold_mentions": [
                        {"start": 0, "end": 11, "text": "Alan Turing", "type": "Person"},
                        {
                            "start": 22,
                            "end": 36,
                            "text": "Bletchley Park",
                            "type": "Place",
                        },
                    ],
                    "gold_claims": [
                        {
                            "subject": "Alan Turing",
                            "predicate": "works_at",
                            "object": "Bletchley Park",
                            "negated": False,
                            "modality": "asserted",
                            "temporal": None,
                            "attribution": None,
                        }
                    ],
                    "gold_events": [],
                    "gold_intents": [],
                    "gold_facts": [],
                }
            ),
        ),
        sha256="f" * 64,
    )
    runtime = MandatoryUnstructuredRuntime(
        pipeline=pipeline,
        providers=bundle,
        governed_schema=schema,
        held_out_gold=gold,
        model_lock_sha256="e" * 64,
    )
    source = SourceRecord.from_text(text, source_name="memory://publication.txt")

    batch = runtime.extract(source)

    assert runtime.pipeline_results
    report = runtime.claim_evaluation_report
    assert report is not None and report.passed
    assert report.dataset_version == "v1"
    assert report.model_lock_sha256 == "e" * 64
    assert report.cases[0].provider_participation_passed
    assert report.cases[0].prediction_sha256
    metadata = batch.metadata["mandatory_unstructured_runtime"]
    assert metadata["held_out_claim_evaluation_passed"] is True
    assert set(metadata["participating_provider_ids"]) == set(
        bundle.mandatory_runtime_participation_ids
    )
    assert metadata["final_resource_shacl_executed"] is True
    assert metadata["final_resource_shacl_conforms"] is True
    assert metadata["final_resource_schema_sha256"] == schema.source_sha256
    publication_rdf = RDFGraphBuilder().build(batch)
    assert (
        metadata["final_rdf_graph_sha256"]
        == final_rdf_graph_sha256(publication_rdf)
        == batch.metadata["final_resource_shacl"]["rdf_graph_sha256"]
    )
