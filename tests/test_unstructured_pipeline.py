"""Evidence-first contracts and hard evaluations for unstructured construction."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from kg_construction.construction.workflow import DraftConstructionWorkflow
from kg_construction.domain.claim import (
    Claim,
    ClaimKind,
    DocumentRef,
    Evidence,
    EvidenceKind,
    Mention,
    Modality,
    Provenance,
    ResolutionStatus,
    ReviewState,
    TextSpan,
)
from kg_construction.domain.models import SourceRecord
from kg_construction.evaluation.claim_evaluations import (
    ClaimEvaluationSuite,
    HardCoverageThresholds,
)
from kg_construction.infrastructure.ontology.unstructured_governance import (
    GovernedUnstructuredSchema,
)
from kg_construction.unstructured.pipeline import (
    EntityLink,
    PatternClaimExtractor,
    ProviderContractError,
    ProviderIdentityResolver,
    RuleMentionDetector,
    RuntimeProfile,
    UnstructuredPipeline,
    UnstructuredPipelineConfig,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "gold_standards" / "unstructured_claims.json"


def _gold() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _run() -> tuple:
    gold = _gold()
    source = SourceRecord.from_text(gold["text"], source_name="held_out.txt")
    return UnstructuredPipeline().run(source), gold


class ImmutableObservationTests(unittest.TestCase):
    def test_document_evidence_mention_and_claim_are_distinct_immutable_resources(self) -> None:
        result, _ = _run()
        mention = result.mentions[0]
        claim = result.promoted[0]

        self.assertNotEqual(result.document.document_id, mention.mention_id)
        self.assertNotEqual(mention.mention_id, mention.evidence.evidence_id)
        self.assertNotEqual(claim.claim_id, claim.evidence[0].evidence_id)
        self.assertEqual(mention.document_version, result.document.version)
        self.assertEqual(claim.document.version, result.document.version)

        with self.assertRaises(ValidationError):
            result.document.version = "mutated"  # type: ignore[misc]
        with self.assertRaises(ValidationError):
            mention.surface = "mutated"  # type: ignore[misc]
        with self.assertRaises(ValidationError):
            claim.predicate = "mutated"  # type: ignore[misc]

    def test_changed_source_content_creates_a_new_snapshot(self) -> None:
        first = SourceRecord.from_text("Ada built a machine.", source_name="same.txt")
        second = SourceRecord.from_text("Ada built the machine.", source_name="same.txt")
        first_ref = DocumentRef.from_source(first)
        second_ref = DocumentRef.from_source(second)

        self.assertEqual(first_ref.document_id, second_ref.document_id)
        self.assertNotEqual(first_ref.version, second_ref.version)
        self.assertNotEqual(first_ref.content_digest, second_ref.content_digest)
        self.assertFalse(first_ref.matches_text(second.content or ""))

    def test_qualifier_changes_are_part_of_claim_identity(self) -> None:
        result, _ = _run()
        claim = result.promoted[0]
        changed = claim.model_copy(update={"negated": not claim.negated})
        self.assertNotEqual(claim.claim_id, changed.claim_id)


class EvidenceAndProvenanceTests(unittest.TestCase):
    def test_every_promoted_claim_has_exact_evidence_and_complete_provenance(self) -> None:
        result, gold = _run()
        self.assertTrue(result.promoted)
        self.assertEqual(result.exact_evidence_coverage, 1.0)
        self.assertEqual(result.provenance_coverage, 1.0)
        for claim in result.promoted:
            self.assertTrue(claim.has_exact_evidence(gold["text"]))
            self.assertTrue(claim.has_complete_provenance(gold["text"]))
            self.assertEqual(claim.review_state, ReviewState.PROMOTED)
            self.assertTrue(claim.document.version)
            self.assertTrue(claim.provenance.model_version)

    def test_all_segment_mention_and_claim_offsets_quote_the_exact_document(self) -> None:
        result, gold = _run()
        evidence = list(result.segments)
        evidence.extend(mention.evidence for mention in result.mentions)
        evidence.extend(item for claim in result.claims for item in claim.evidence)
        for item in evidence:
            self.assertEqual(
                gold["text"][item.span.start : item.span.end],
                item.span.text,
            )
            self.assertEqual(item.document_version, result.document.version)

    def test_inexact_mention_provider_output_is_rejected(self) -> None:
        class BadMentionProvider:
            def detect(self, document, text, segments):
                del text, segments
                evidence = Evidence.build(
                    document,
                    span=TextSpan(start=0, end=4, text="Nope"),
                    kind=EvidenceKind.MENTION,
                )
                return (
                    Mention(
                        document_version=document.version,
                        evidence=evidence,
                        surface="Nope",
                        entity_type="Entity",
                        provenance=Provenance(
                            extractor="bad_detector",
                            provider="test",
                            model_version="1",
                            configuration_id="test",
                            confidence=1.0,
                        ),
                    ),
                )

        source = SourceRecord.from_text("Alan Turing.", source_name="bad.txt")
        with self.assertRaises(ProviderContractError):
            UnstructuredPipeline(mention_detector=BadMentionProvider()).run(source)

    def test_inexact_claim_evidence_is_quarantined(self) -> None:
        class BadClaimProvider:
            def extract(self, document, text, segments, mentions):
                del text, segments, mentions
                evidence = Evidence.build(
                    document,
                    span=TextSpan(start=0, end=4, text="Nope"),
                    kind=EvidenceKind.SENTENCE,
                )
                return (
                    Claim(
                        document=document,
                        kind=ClaimKind.RELATION,
                        subject="Alan Turing",
                        predicate="knows",
                        object="Ada Lovelace",
                        evidence=(evidence,),
                        provenance=Provenance(
                            extractor="bad_claim_provider",
                            provider="test",
                            model_version="1",
                            configuration_id="test",
                            confidence=1.0,
                        ),
                    ),
                )

        source = SourceRecord.from_text(
            "Alan Turing knows Ada Lovelace.", source_name="bad-claim.txt"
        )
        result = UnstructuredPipeline(claim_extractor=BadClaimProvider()).run(source)
        self.assertFalse(result.promoted)
        self.assertEqual(len(result.quarantined), 1)
        self.assertEqual(result.quarantined[0].review_state, ReviewState.QUARANTINED)


class IdentityResolutionDoctrineTests(unittest.TestCase):
    def test_coreference_is_not_canonical_entity_linking(self) -> None:
        result, _ = _run()
        alan = next(mention for mention in result.mentions if mention.surface == "Alan Turing")
        turing = next(mention for mention in result.mentions if mention.surface == "Turing")

        self.assertEqual(alan.coreference_cluster_id, turing.coreference_cluster_id)
        self.assertEqual(alan.coreference_anchor, "Alan Turing")
        self.assertEqual(turing.coreference_anchor, "Alan Turing")
        self.assertEqual(alan.resolution_status, ResolutionStatus.COREFERENCE_ONLY)
        self.assertIsNone(alan.canonical_entity_id)
        self.assertIsNone(turing.canonical_entity_id)
        self.assertFalse(result.canonical_entities)

    def test_uncertain_mentions_are_never_forced_into_entities(self) -> None:
        result, _ = _run()
        self.assertTrue(result.mentions)
        self.assertTrue(all(mention.unresolved for mention in result.mentions))
        self.assertTrue(all(mention.entity_id is None for mention in result.mentions))

    def test_claim_arguments_use_resolved_coreference_anchors(self) -> None:
        class SurfaceClaimProvider:
            def extract(self, document, text, segments, mentions):
                del text
                turing = next(
                    mention for mention in mentions if mention.surface == "Turing"
                )
                bombe = next(
                    mention for mention in mentions if mention.surface == "Bombe"
                )
                evidence = next(
                    segment
                    for segment in segments
                    if "Turing designed the Bombe" in segment.span.text
                )
                return (
                    Claim(
                        document=document,
                        kind=ClaimKind.RELATION,
                        subject=turing.surface,
                        predicate="designed",
                        object=bombe.surface,
                        subject_mention_id=turing.mention_id,
                        object_mention_id=bombe.mention_id,
                        evidence=(evidence,),
                        provenance=Provenance(
                            extractor="surface_claim_provider",
                            provider="test",
                            model_version="1",
                            configuration_id="coreference-anchor",
                            confidence=1.0,
                        ),
                    ),
                )

        gold = _gold()
        source = SourceRecord.from_text(
            gold["text"],
            source_name="coreference-claim.txt",
        )
        result = UnstructuredPipeline(
            claim_extractor=SurfaceClaimProvider(),
        ).run(source)

        self.assertEqual(len(result.promoted), 1)
        self.assertEqual(result.promoted[0].subject, "Alan Turing")
        self.assertEqual(result.promoted[0].object, "Bombe")

    def test_partial_tokens_never_merge(self) -> None:
        source = SourceRecord.from_text(
            "Ann met Anna. York contacted Yorkshire.", source_name="partial.txt"
        )
        result = UnstructuredPipeline().run(source)
        by_surface = {mention.surface: mention for mention in result.mentions}

        self.assertNotEqual(
            by_surface["Ann"].coreference_cluster_id,
            by_surface["Anna"].coreference_cluster_id,
        )
        self.assertNotEqual(
            by_surface["York"].coreference_cluster_id,
            by_surface["Yorkshire"].coreference_cluster_id,
        )

    def test_ambiguous_surname_is_not_attached_to_either_person(self) -> None:
        source = SourceRecord.from_text(
            "John Smith met Jane Smith. Smith replied.", source_name="ambiguous.txt"
        )
        result = UnstructuredPipeline().run(source)
        by_surface = {mention.surface: mention for mention in result.mentions}

        self.assertNotEqual(
            by_surface["Smith"].coreference_cluster_id,
            by_surface["John Smith"].coreference_cluster_id,
        )
        self.assertNotEqual(
            by_surface["Smith"].coreference_cluster_id,
            by_surface["Jane Smith"].coreference_cluster_id,
        )

    def test_injected_link_provider_creates_canonical_identity_and_binds_claims(self) -> None:
        class TuringLinkProvider:
            def link(self, canonical_name, entity_type, mentions):
                del entity_type, mentions
                if canonical_name != "Alan Turing":
                    return None
                return EntityLink(
                    external_system="wikidata",
                    external_id="Q7251",
                    canonical_name="Alan Turing",
                    entity_type="Person",
                    confidence=0.99,
                )

        gold = _gold()
        source = SourceRecord.from_text(gold["text"], source_name="held_out.txt")
        result = UnstructuredPipeline(
            identity_resolver=ProviderIdentityResolver(TuringLinkProvider())
        ).run(source)

        self.assertEqual(len(result.canonical_entities), 1)
        entity = result.canonical_entities[0]
        linked_mentions = [
            mention for mention in result.mentions if mention.canonical_entity_id == entity.entity_id
        ]
        self.assertEqual({mention.surface for mention in linked_mentions}, {"Alan Turing", "Turing"})
        self.assertTrue(
            all(
                claim.subject_entity_id == entity.entity_id
                for claim in result.promoted
                if claim.subject == "Alan Turing"
            )
        )


class SemanticPreservationTests(unittest.TestCase):
    def test_provider_duplicate_claims_merge_evidence_and_provenance_once(
        self,
    ) -> None:
        class DuplicateEvidenceProvider:
            def extract(inner_self, document, text, segments, mentions):
                del inner_self
                base = next(
                    claim
                    for claim in PatternClaimExtractor().extract(
                        document,
                        text,
                        segments,
                        mentions,
                    )
                    if claim.kind == ClaimKind.RELATION
                    and claim.predicate == "works_with"
                )
                passage = Evidence.build(
                    document,
                    span=base.evidence[0].span,
                    kind=EvidenceKind.PASSAGE,
                )
                model_duplicate = base.model_copy(
                    update={
                        "evidence": (passage,),
                        "provenance": Provenance(
                            extractor="model_claim_provider",
                            provider="model_provider",
                            model_version="model-v1",
                            configuration_id="grounded-passages-v1",
                            confidence=0.9,
                        ),
                    }
                )
                return base, model_duplicate

        source = SourceRecord.from_text(
            "Alan Turing worked with Charles Babbage.",
            source_name="semantic-claim-duplicates.txt",
        )
        pipeline = UnstructuredPipeline(
            claim_extractor=DuplicateEvidenceProvider(),
            governed_schema=GovernedUnstructuredSchema.from_package_resource(),
        )
        batch = pipeline.extract(source)
        result = pipeline.pipeline_results[-1]

        self.assertEqual(len(result.promoted), 1)
        self.assertFalse(result.quarantined)
        merged = result.promoted[0]
        self.assertEqual(merged.predicate, "works_with")
        self.assertEqual(
            {item.kind for item in merged.evidence},
            {EvidenceKind.SENTENCE, EvidenceKind.PASSAGE},
        )
        self.assertEqual(
            set(merged.provenance.provider.split("+")),
            {"local_rules", "model_provider"},
        )
        self.assertEqual(len(result.semantic_claim_merge_results), 1)
        merge = result.semantic_claim_merge_results[0]
        self.assertEqual(merge["merged_claim_count"], 2)
        self.assertEqual(merge["output_claim_id"], merged.claim_id)
        self.assertEqual(
            sum(
                relation.relation_type == "WORKS_WITH"
                and relation.properties.get("claim_kind") == "relation"
                for relation in batch.relations
            ),
            1,
        )

    def test_ineligible_duplicate_cannot_poison_publishable_claim(
        self,
    ) -> None:
        class MixedConfidenceProvider:
            def extract(inner_self, document, text, segments, mentions):
                del inner_self
                base = next(
                    claim
                    for claim in PatternClaimExtractor().extract(
                        document,
                        text,
                        segments,
                        mentions,
                    )
                    if claim.kind == ClaimKind.RELATION
                    and claim.predicate == "works_with"
                )
                below_threshold = base.model_copy(
                    update={
                        "provenance": Provenance(
                            extractor="below_threshold_provider",
                            provider="low_confidence_model",
                            model_version="model-v1",
                            configuration_id="below-publication-threshold",
                            confidence=0.1,
                        )
                    }
                )
                return base, below_threshold

        source = SourceRecord.from_text(
            "Alan Turing worked with Charles Babbage.",
            source_name="semantic-claim-confidence.txt",
        )
        pipeline = UnstructuredPipeline(
            config=UnstructuredPipelineConfig(
                minimum_claim_confidence=0.5,
            ),
            claim_extractor=MixedConfidenceProvider(),
        )
        pipeline.extract(source)
        result = pipeline.pipeline_results[-1]

        self.assertEqual(len(result.promoted), 1)
        self.assertEqual(
            result.promoted[0].provenance.provider,
            "local_rules",
        )
        self.assertEqual(len(result.quarantined), 1)
        self.assertEqual(
            result.quarantined[0].provenance.provider,
            "low_confidence_model",
        )
        self.assertFalse(result.semantic_claim_merge_results)

    def test_semantic_claim_consolidation_retains_distinct_occurrences(
        self,
    ) -> None:
        source = SourceRecord.from_text(
            "Ada worked with Bob. Ada worked with Bob.",
            source_name="repeated-claim-occurrences.txt",
        )
        pipeline = UnstructuredPipeline()
        pipeline.extract(source)
        result = pipeline.pipeline_results[-1]
        repeated = [
            claim
            for claim in result.promoted
            if claim.kind == ClaimKind.RELATION
            and claim.predicate == "works_with"
        ]

        self.assertEqual(len(repeated), 2)
        self.assertNotEqual(
            repeated[0].subject_mention_id,
            repeated[1].subject_mention_id,
        )
        self.assertFalse(result.semantic_claim_merge_results)

    def test_semantic_claim_consolidation_retains_qualifier_differences(
        self,
    ) -> None:
        class QualifiedDuplicateProvider:
            def extract(inner_self, document, text, segments, mentions):
                del inner_self
                asserted = next(
                    claim
                    for claim in PatternClaimExtractor().extract(
                        document,
                        text,
                        segments,
                        mentions,
                    )
                    if claim.kind == ClaimKind.RELATION
                    and claim.predicate == "works_with"
                )
                possible = asserted.model_copy(
                    update={
                        "modality": Modality.POSSIBLE,
                        "provenance": Provenance(
                            extractor="possible_claim_provider",
                            provider="possible_model",
                            model_version="model-v1",
                            configuration_id="qualifier-preservation-v1",
                            confidence=0.9,
                        ),
                    }
                )
                return asserted, possible

        class PreserveProviderAttributes:
            def annotate(self, text, claims):
                del text
                return claims

        source = SourceRecord.from_text(
            "Ada worked with Bob.",
            source_name="semantic-claim-qualifiers.txt",
        )
        pipeline = UnstructuredPipeline(
            claim_extractor=QualifiedDuplicateProvider(),
            attribute_extractor=PreserveProviderAttributes(),
        )
        pipeline.extract(source)
        result = pipeline.pipeline_results[-1]

        self.assertEqual(
            {claim.modality for claim in result.promoted},
            {Modality.ASSERTED, Modality.POSSIBLE},
        )
        self.assertFalse(result.semantic_claim_merge_results)

    def test_relations_events_intents_and_facts_are_all_extracted(self) -> None:
        result, _ = _run()
        kinds = {claim.kind for claim in result.promoted}
        self.assertEqual(
            kinds,
            {
                ClaimKind.RELATION,
                ClaimKind.EVENT,
                ClaimKind.INTENT,
                ClaimKind.FACT,
            },
        )

        created_event = next(
            claim
            for claim in result.promoted
            if claim.kind == ClaimKind.EVENT and claim.predicate == "created"
        )
        self.assertEqual(
            {(role.role, role.value) for role in created_event.event_roles},
            {("agent", "Ada Lovelace"), ("theme", "Analytical Engine")},
        )

    def test_negation_time_and_attribution_are_preserved_as_values(self) -> None:
        result, _ = _run()
        works_with = next(
            claim for claim in result.promoted if claim.predicate == "works_with"
        )
        works_at = next(
            claim for claim in result.promoted if claim.predicate == "works_at"
        )
        created = next(
            claim
            for claim in result.promoted
            if claim.kind == ClaimKind.RELATION and claim.predicate == "created"
        )

        self.assertTrue(works_with.negated)
        self.assertEqual(works_at.temporal, "during the war")
        self.assertEqual(created.temporal, "in 1843")
        self.assertEqual(created.attribution, "Menabrea")

    def test_modality_and_quantity_are_not_discarded(self) -> None:
        source = SourceRecord.from_text(
            "Acme Corp reportedly acquired Beta Labs in 2027 for 20 units.",
            source_name="qualifiers.txt",
        )
        result = UnstructuredPipeline().run(source)
        acquired = next(
            claim
            for claim in result.promoted
            if claim.kind == ClaimKind.RELATION and claim.predicate == "acquired"
        )
        self.assertEqual(acquired.modality, Modality.POSSIBLE)
        self.assertEqual(acquired.temporal, "in 2027")
        self.assertEqual(acquired.quantity, "20 units")

    def test_common_operational_relation_is_not_lost(self) -> None:
        source = SourceRecord.from_text(
            "INCIDENT-1042 affected Recommendation API.",
            source_name="operational.txt",
        )
        result = UnstructuredPipeline().run(source)
        affected = next(claim for claim in result.promoted if claim.predicate == "affected")
        self.assertEqual(affected.subject, "INCIDENT-1042")
        self.assertEqual(affected.object, "Recommendation API")
        self.assertIsNone(affected.temporal)
        self.assertTrue(affected.has_exact_evidence(source.content or ""))


class ProviderAndPromotionGateTests(unittest.TestCase):
    def test_model_backed_profile_requires_real_provider_ports(self) -> None:
        config = UnstructuredPipelineConfig(profile=RuntimeProfile.MODEL_BACKED)
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            UnstructuredPipeline(config=config)

        # The contract is provider-agnostic: production GLiNER/transformer
        # adapters and deterministic test doubles enter through the same ports.
        pipeline = UnstructuredPipeline(
            config=config,
            mention_detector=RuleMentionDetector(),
            claim_extractor=PatternClaimExtractor(),
        )
        self.assertEqual(pipeline.config.profile, RuntimeProfile.MODEL_BACKED)

    def test_off_ontology_predicate_is_quarantined_not_promoted(self) -> None:
        gold = _gold()
        source = SourceRecord.from_text(gold["text"], source_name="held_out.txt")
        result = UnstructuredPipeline(ontology_predicates={"works_at"}).run(source)
        self.assertTrue(all(claim.predicate == "works_at" for claim in result.promoted))
        self.assertTrue(any(claim.predicate != "works_at" for claim in result.quarantined))
        self.assertTrue(
            all(
                claim.review_state == ReviewState.QUARANTINED
                for claim in result.quarantined
            )
        )

    def test_wrong_kind_does_not_quarantine_valid_same_predicate_claim(
        self,
    ) -> None:
        class MixedKindClaimProvider:
            def extract(self, document, text, segments, mentions):
                claims = PatternClaimExtractor().extract(
                    document,
                    text,
                    segments,
                    mentions,
                )
                event = next(
                    claim for claim in claims if claim.kind == ClaimKind.EVENT
                )
                relation = next(
                    claim for claim in claims if claim.kind == ClaimKind.RELATION
                )
                invalid_fact = event.model_copy(
                    update={
                        "kind": ClaimKind.FACT,
                        "event_roles": (),
                    }
                )
                return relation, event, invalid_fact

        source = SourceRecord.from_text(
            "Ada Lovelace created the Analytical Engine in 1843.",
            source_name="same-predicate-wrong-kind.txt",
        )
        result = UnstructuredPipeline(
            claim_extractor=MixedKindClaimProvider(),
            governed_schema=GovernedUnstructuredSchema.from_package_resource(),
        ).run(source)

        self.assertEqual(
            [(claim.kind, claim.predicate) for claim in result.promoted],
            [
                (ClaimKind.RELATION, "created"),
                (ClaimKind.EVENT, "created"),
            ],
        )
        self.assertEqual(
            [(claim.kind, claim.predicate) for claim in result.quarantined],
            [(ClaimKind.FACT, "created")],
        )
        self.assertTrue(result.shacl_conforms)
        self.assertEqual(
            [
                (
                    item["claim_kind"],
                    item["provider_predicate"],
                    item["status"],
                )
                for item in result.claim_ontology_alignment_results
            ],
            [
                ("relation", "created", "aligned"),
                ("event", "created", "aligned"),
                ("fact", "created", "quarantined"),
            ],
        )


class ClaimArgumentResolutionTests(unittest.TestCase):
    @staticmethod
    def _provenance() -> Provenance:
        return Provenance(
            extractor="argument_test_provider",
            provider="test",
            model_version="1",
            configuration_id="exact-anchor-tests",
            confidence=1.0,
        )

    def test_duplicate_surface_arguments_without_occurrence_ids_quarantine_entire_claim(
        self,
    ) -> None:
        class AmbiguousProvider:
            def extract(inner_self, document, text, segments, mentions):
                del inner_self, mentions
                return (
                    Claim(
                        document=document,
                        kind=ClaimKind.RELATION,
                        subject="Ada",
                        predicate="met",
                        object="Ada",
                        evidence=(segments[0],),
                        provenance=self._provenance(),
                    ),
                )

        source = SourceRecord.from_text("Ada met Ada.", source_name="duplicates.txt")
        pipeline = UnstructuredPipeline(claim_extractor=AmbiguousProvider())
        batch = pipeline.extract(source)
        result = pipeline.pipeline_results[-1]

        self.assertFalse(result.promoted)
        self.assertEqual(len(result.quarantined), 1)
        # The quarantined claim is materialized in the graph flagged, not dropped:
        # conversion always returns every extracted claim with its status.
        quarantined_claim_nodes = [
            entity
            for entity in batch.entities
            if entity.entity_type == "Claim"
            and entity.properties.get("promotion_status") == "quarantined"
        ]
        self.assertEqual(len(quarantined_claim_nodes), 1)
        statuses = {
            item["status"]
            for item in result.claim_argument_resolution_results[0]["arguments"]
        }
        self.assertEqual(statuses, {"ambiguous_evidence_anchor"})

    def test_stable_occurrence_ids_disambiguate_duplicate_surface_arguments(
        self,
    ) -> None:
        class AnchoredProvider:
            def extract(inner_self, document, text, segments, mentions):
                del inner_self, text
                ada_mentions = sorted(
                    (mention for mention in mentions if mention.surface == "Ada"),
                    key=lambda mention: mention.span.start,
                )
                return (
                    Claim(
                        document=document,
                        kind=ClaimKind.RELATION,
                        subject="Ada",
                        predicate="met",
                        object="Ada",
                        subject_mention_id=ada_mentions[0].mention_id,
                        object_mention_id=ada_mentions[1].mention_id,
                        evidence=(segments[0],),
                        provenance=self._provenance(),
                    ),
                )

        source = SourceRecord.from_text("Ada met Ada.", source_name="anchored.txt")
        result = UnstructuredPipeline(claim_extractor=AnchoredProvider()).run(source)

        self.assertEqual(len(result.promoted), 1)
        promoted = result.promoted[0]
        self.assertNotEqual(
            promoted.subject_mention_id,
            promoted.object_mention_id,
        )
        statuses = {
            item["status"]
            for item in result.claim_argument_resolution_results[0]["arguments"]
        }
        self.assertEqual(statuses, {"provided_id"})

    def test_exact_evidence_span_can_recover_one_repeated_quote_occurrence(
        self,
    ) -> None:
        class FirstOccurrenceProvider:
            def extract(inner_self, document, text, segments, mentions):
                del inner_self, text, mentions
                return (
                    Claim(
                        document=document,
                        kind=ClaimKind.RELATION,
                        subject="Ada",
                        predicate="met",
                        object="Bob",
                        evidence=(segments[0],),
                        provenance=self._provenance(),
                    ),
                )

        source = SourceRecord.from_text(
            "Ada met Bob. Ada met Bob.",
            source_name="repeated-quote.txt",
        )
        result = UnstructuredPipeline(
            claim_extractor=FirstOccurrenceProvider()
        ).run(source)

        self.assertEqual(len(result.promoted), 1)
        self.assertIsNotNone(result.promoted[0].subject_mention_id)
        self.assertIsNotNone(result.promoted[0].object_mention_id)
        statuses = {
            item["status"]
            for item in result.claim_argument_resolution_results[0]["arguments"]
        }
        self.assertEqual(statuses, {"unique_evidence_anchor"})

    def test_repeated_evidence_and_unknown_ids_never_fall_back_first_wins(
        self,
    ) -> None:
        class AdversarialProvider:
            def extract(inner_self, document, text, segments, mentions):
                del inner_self, text, mentions
                return (
                    Claim(
                        document=document,
                        kind=ClaimKind.INTENT,
                        subject="Ada",
                        predicate="meet",
                        object="Bob",
                        subject_mention_id="mention_unknown",
                        evidence=segments,
                        provenance=self._provenance(),
                    ),
                )

        source = SourceRecord.from_text(
            "Ada met Bob. Ada met Bob.",
            source_name="unknown-anchor.txt",
        )
        result = UnstructuredPipeline(claim_extractor=AdversarialProvider()).run(
            source
        )

        self.assertFalse(result.promoted)
        self.assertEqual(len(result.quarantined), 1)
        quarantined = result.quarantined[0]
        self.assertEqual(quarantined.kind, ClaimKind.INTENT)
        self.assertEqual(quarantined.provenance, self._provenance())
        self.assertEqual(
            tuple(item.evidence_id for item in quarantined.evidence),
            tuple(item.evidence_id for item in result.segments),
        )
        statuses = {
            item["argument"]: item["status"]
            for item in result.claim_argument_resolution_results[0]["arguments"]
        }
        self.assertEqual(statuses["subject"], "unknown_mention_id")
        self.assertEqual(statuses["object"], "ambiguous_evidence_anchor")


class ConstructionWorkflowContractTests(unittest.TestCase):
    def test_extract_materializes_all_resource_layers_with_evidence(self) -> None:
        gold = _gold()
        source = SourceRecord.from_text(gold["text"], source_name="held_out.txt")
        batch = UnstructuredPipeline().extract(source)
        entity_types = {entity.entity_type for entity in batch.entities}

        self.assertTrue({"Document", "Mention", "Claim", "Event"} <= entity_types)
        self.assertTrue(any(entity.properties.get("identity_layer") == "observed" for entity in batch.entities))
        self.assertEqual(len(batch.facts), 1)
        self.assertTrue(
            {"WORKS_AT", "WORKS_WITH", "DESIGNED", "CREATED", "PUBLISH"}
            <= {relation.relation_type for relation in batch.relations}
        )
        self.assertTrue(all(entity.evidence for entity in batch.entities))
        self.assertTrue(all(relation.evidence for relation in batch.relations))
        self.assertTrue(all(fact.evidence for fact in batch.facts))

        for evidence in (
            item
            for graph_object in [*batch.entities, *batch.relations, *batch.facts]
            for item in graph_object.evidence
        ):
            start = int(evidence.location["start"])
            end = int(evidence.location["end"])
            self.assertEqual(gold["text"][start:end], evidence.text)

    def test_workflow_can_use_pipeline_directly_and_passes_validation(self) -> None:
        gold = _gold()
        source = SourceRecord.from_text(gold["text"], source_name="held_out.txt")
        result = DraftConstructionWorkflow(
            extractors=[UnstructuredPipeline()]
        ).run([source])

        self.assertTrue(result.validation_report.valid)
        self.assertTrue(result.batch.entities)
        self.assertTrue(result.batch.relations)
        self.assertTrue(result.batch.facts)
        self.assertTrue(result.property_graph.nodes)
        self.assertTrue(result.property_graph.edges)
        self.assertTrue(result.rdf_graph.quads)

    def test_linked_identity_materializes_as_canonical_entity_resource(self) -> None:
        class TuringLinkProvider:
            def link(self, canonical_name, entity_type, mentions):
                del entity_type, mentions
                if canonical_name != "Alan Turing":
                    return None
                return EntityLink(
                    external_system="wikidata",
                    external_id="Q7251",
                    canonical_name="Alan Turing",
                    entity_type="Person",
                    confidence=0.99,
                )

        gold = _gold()
        source = SourceRecord.from_text(gold["text"], source_name="held_out.txt")
        batch = UnstructuredPipeline(
            identity_resolver=ProviderIdentityResolver(TuringLinkProvider())
        ).extract(source)
        canonical = [
            entity for entity in batch.entities if entity.entity_type == "CanonicalEntity"
        ]

        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0].properties["external_id"], "Q7251")
        self.assertTrue(
            any(
                relation.relation_type == "REFERS_TO_CANONICAL_ENTITY"
                and relation.target == canonical[0].properties["canonical_entity_id"]
                for relation in batch.relations
            )
        )


class EvaluationSuiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result, self.gold = _run()
        self.suite = ClaimEvaluationSuite()
        self.scores = self.suite.evaluate(self.result, self.gold)

    def test_every_hard_evaluation_executes(self) -> None:
        self.assertEqual(set(self.scores), self.suite.required_metrics)
        self.assertTrue(all(score.executed for score in self.scores.values()))

    def test_fixture_has_perfect_exact_semantic_and_traceability_scores(self) -> None:
        for metric in self.suite.required_metrics:
            self.assertEqual(self.scores[metric].f1, 1.0, metric)
        self.assertEqual(self.scores["entity_span"].gold_count, 13)
        self.assertEqual(self.scores["relation"].gold_count, 4)
        self.assertEqual(self.scores["event_role"].gold_count, 2)
        self.assertEqual(self.scores["intent"].gold_count, 1)
        self.assertEqual(self.scores["fact"].gold_count, 1)
        self.assertEqual(self.scores["negation"].aligned_count, 4)
        self.assertEqual(self.scores["quantity"].gold_count, 4)

    def test_default_hard_gate_passes_only_with_full_coverage_and_no_false_claims(self) -> None:
        report = self.suite.hard_gate(
            self.result,
            self.gold,
            HardCoverageThresholds(),
        )
        self.assertTrue(report.passed, report.failures)
        self.assertEqual(report.claim_coverage, 1.0)
        self.assertEqual(report.unsupported_claim_rate, 0.0)
        self.assertFalse(report.failures)

    def test_hard_gate_fails_when_gold_semantics_are_not_recovered(self) -> None:
        altered = copy.deepcopy(self.gold)
        altered["gold_claims"][0]["object"] = "A Different Place"
        report = self.suite.hard_gate(self.result, altered)

        self.assertFalse(report.passed)
        self.assertLess(report.scores["relation"].f1, 1.0)
        self.assertGreater(report.unsupported_claim_rate, 0.0)
        self.assertTrue(any("relation F1" in failure for failure in report.failures))


if __name__ == "__main__":
    unittest.main()
