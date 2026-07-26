"""Unstructured entity identification, canonicalisation, and coverage."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from kg_construction.domain.evidence import span_evidence
from kg_construction.domain.modeling import OntologyClass, OntologyPredicate, OntologySchema
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    RelationCandidate,
    SourceRecord,
)
from kg_construction.evaluation.extraction_coverage import CoverageEvaluator
from kg_construction.infrastructure.ontology.ontology_alignment import OntologyAligner
from kg_construction.infrastructure.resolution.canonicalization import (
    LinkResult,
    UnstructuredCanonicalizer,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "gold_standards" / "unstructured_coverage.json"


def _gold() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _source() -> SourceRecord:
    return SourceRecord.from_text(_gold()["text"], source_name="doc.txt")


def _entities(source: SourceRecord, mentions: list[tuple[str, str]]) -> list[EntityCandidate]:
    entities = []
    for offset, (name, entity_type) in enumerate(mentions):
        entities.append(
            EntityCandidate.build(
                entity_type=entity_type,
                name=name,
                evidence=[span_evidence(source, start=offset, end=offset + len(name), text=name)],
                extractor_name="test_extractor",
                technique="test",
            )
        )
    return entities


class _WikidataStub:
    """A stub linker that links only Turing mentions."""

    def link(self, name: str, entity_type: str) -> LinkResult | None:
        if "turing" in name.lower():
            return LinkResult(canonical_id="Q7251", canonical_name="Alan Turing", knowledge_base="wikidata", confidence=0.95)
        return None


class _KeywordJudge:
    """A stub LLM-as-judge: an intent is covered if its subject appears in context."""

    def judge(self, intent: str, context: str) -> bool:
        subject = intent.split(" ", 2)[0].lower()  # first token, e.g. "alan"/"ada"
        return subject in context.lower()


class CanonicalizationTests(unittest.TestCase):
    def test_coreference_merges_partial_mentions_and_defaults_to_nil(self) -> None:
        source = _source()
        batch = ConstructionBatch(
            source_records=[source],
            entities=_entities(source, [("Alan Turing", "Person"), ("Turing", "Person"), ("Ada Lovelace", "Person")]),
        )
        resolved = UnstructuredCanonicalizer().resolve(batch)
        canonical_ids = {entity.properties["canonical_entity_id"] for entity in resolved.entities}
        self.assertEqual(len(canonical_ids), 2)  # Turing folds into Alan Turing
        self.assertTrue(all(entity.properties["link_status"] == "nil" for entity in resolved.entities))

    def test_linker_links_and_lowers_nil_rate(self) -> None:
        source = _source()
        batch = ConstructionBatch(
            source_records=[source],
            entities=_entities(source, [("Alan Turing", "Person"), ("Bletchley Park", "Organization")]),
        )
        resolved = UnstructuredCanonicalizer(_WikidataStub()).resolve(batch)
        statuses = {entity.name: entity.properties["link_status"] for entity in resolved.entities}
        self.assertEqual(statuses["Alan Turing"], "linked")
        self.assertEqual(statuses["Bletchley Park"], "nil")
        turing = next(e for e in resolved.entities if e.name == "Alan Turing")
        self.assertEqual(turing.properties["knowledge_base"], "wikidata")


class CoverageAcceptanceTests(unittest.TestCase):
    """'intent not missed' becomes reported numbers."""

    def setUp(self) -> None:
        self.gold = _gold()
        self.source = _source()

    def _covered_batch(self) -> ConstructionBatch:
        # Extraction captured 3 of 5 gold entities (missed Analytical Engine, Bombe).
        entities = _entities(
            self.source,
            [("Alan Turing", "Person"), ("Turing", "Person"), ("Bletchley Park", "Organization"), ("Ada Lovelace", "Person")],
        )
        resolved = UnstructuredCanonicalizer().resolve(ConstructionBatch(source_records=[self.source], entities=entities))
        turing_id = next(e.properties["canonical_entity_id"] for e in resolved.entities if "turing" in e.name.lower())
        park_id = next(e.properties["canonical_entity_id"] for e in resolved.entities if e.name == "Bletchley Park")
        relation = RelationCandidate.build(
            source=turing_id, relation_type="works_at", target=park_id,
            evidence=[span_evidence(self.source, start=0, end=11, text="Alan Turing")],
            extractor_name="test", technique="test",
        )
        return resolved.model_copy(update={"relations": [relation]})

    def test_entity_coverage_is_a_reported_number(self) -> None:
        report = CoverageEvaluator().evaluate(self._covered_batch(), gold_entities=self.gold["gold_entities"])
        self.assertEqual(report.gold_entity_count, 5)
        self.assertEqual(report.matched_entity_count, 3)
        self.assertAlmostEqual(report.entity_coverage, 3 / 5)

    def test_claim_coverage_is_a_reported_number(self) -> None:
        report = CoverageEvaluator().evaluate(
            self._covered_batch(),
            gold_entities=self.gold["gold_entities"],
            gold_claims=self.gold["gold_claims"],
        )
        # 1 of 2 gold claims captured (works_at; not wrote).
        self.assertEqual(report.gold_claim_count, 2)
        self.assertEqual(report.matched_claim_count, 1)
        self.assertAlmostEqual(report.claim_coverage, 0.5)

    def test_nil_rate_is_reported(self) -> None:
        report = CoverageEvaluator().evaluate(self._covered_batch(), gold_entities=self.gold["gold_entities"])
        self.assertEqual(report.nil_rate, 1.0)  # no linker configured

    def test_intent_coverage_llm_with_judge(self) -> None:
        report = CoverageEvaluator().evaluate(
            self._covered_batch(),
            gold_entities=self.gold["gold_entities"],
            gold_intents=self.gold["gold_intents"],
            judge=_KeywordJudge(),
        )
        # Both intents' subjects (Alan, Ada) appear in the graph context.
        self.assertEqual(report.gold_intent_count, 2)
        self.assertIsNotNone(report.intent_coverage_llm)
        self.assertAlmostEqual(report.intent_coverage_llm or 0.0, 1.0)

    def test_report_is_serialisable(self) -> None:
        report = CoverageEvaluator().evaluate(self._covered_batch(), gold_entities=self.gold["gold_entities"])
        payload = report.model_dump(mode="json")
        self.assertIn("entity_coverage", payload)
        self.assertIn("claim_coverage", payload)


class OntologyAlignmentTests(unittest.TestCase):
    def test_conformance_and_off_ontology_flagging(self) -> None:
        source = _source()
        entities = _entities(source, [("Alan Turing", "Person"), ("Bletchley Park", "Organization")])
        relation = RelationCandidate.build(
            source=entities[0].candidate_id, relation_type="works_at", target=entities[1].candidate_id,
            evidence=[span_evidence(source, start=0, end=11, text="Alan Turing")],
            extractor_name="t", technique="t",
        )
        batch = ConstructionBatch(source_records=[source], entities=entities, relations=[relation])
        ontology = OntologySchema(
            classes=[OntologyClass(name="Person", role="record")],
            predicates=[OntologyPredicate(name="works_at", domain="Person", range="Organization", kind="object")],
        )
        report = OntologyAligner(ontology).align(batch)
        self.assertAlmostEqual(report.in_ontology_entity_rate, 0.5)  # Person aligned, Organization not
        self.assertEqual(report.off_ontology_entity_types, ["Organization"])
        self.assertAlmostEqual(report.in_ontology_relation_rate, 1.0)  # works_at aligned
        self.assertEqual(report.off_ontology_relation_types, [])


if __name__ == "__main__":
    unittest.main()
