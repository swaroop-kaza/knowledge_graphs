from __future__ import annotations

import json
import unittest
import urllib.parse
from typing import Any

from kg_construction.application.registry import default_registry
from kg_construction.domain.evidence import span_evidence
from kg_construction.domain.models import ConstructionBatch, EntityCandidate, SourceRecord
from kg_construction.infrastructure.resolution import (
    CrossEncoderResolver,
    CrossEncoderResolverConfig,
    DBpediaLinker,
    EmbeddingResolver,
    EmbeddingResolverConfig,
    OpenRefineReconciler,
    OpenRefineReconcilerConfig,
    WikidataLinker,
)


def batch_with_entities(*names: str, entity_type: str = "System") -> ConstructionBatch:
    text = " | ".join(names)
    source = SourceRecord.from_text(text, source_name="memory://entity-linking")
    entities: list[EntityCandidate] = []
    cursor = 0
    for name in names:
        start = text.find(name, cursor)
        cursor = start + len(name)
        entities.append(
            EntityCandidate.build(
                entity_type=entity_type,
                name=name,
                properties={},
                evidence=[span_evidence(source, start=start, end=start + len(name), text=name)],
                extractor_name="test",
                technique="fixture",
            )
        )
    return ConstructionBatch(source_records=[source], entities=entities)


class FakeEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            normalized = text.casefold()
            if "recommendation" in normalized:
                vectors.append([1.0, 0.0, 0.0])
            elif "recommender" in normalized:
                vectors.append([0.96, 0.28, 0.0])
            elif "feature store" in normalized:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class FakeCrossEncoderModel:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for left, right in pairs:
            joined = f"{left} {right}".casefold()
            scores.append(0.96 if "recommendation api" in joined and "recommender api" in joined else 0.2)
        return scores


class EntityLinkingTests(unittest.TestCase):
    def test_wikidata_linker_attaches_external_link_candidates(self) -> None:
        def transport(url: str, **kwargs: Any) -> dict[str, object]:
            self.assertIn("wikidata.org", url)
            self.assertEqual(kwargs["params"]["action"], "wbsearchentities")
            return {
                "search": [
                    {
                        "id": "Q123",
                        "label": "Recommendation API",
                        "description": "software service",
                        "concepturi": "https://www.wikidata.org/entity/Q123",
                    }
                ]
            }

        batch = batch_with_entities("Recommendation API")
        resolved = WikidataLinker(transport=transport).resolve(batch)
        entity = resolved.entities[0]

        self.assertEqual(entity.properties["wikidata_id"], "Q123")
        self.assertEqual(entity.properties["wikidata_label"], "Recommendation API")
        self.assertFalse(entity.properties["wikidata_review_required"])
        self.assertEqual(resolved.metadata["external_linking_audit"][0]["provider"], "wikidata")

    def test_dbpedia_linker_attaches_lookup_results(self) -> None:
        def transport(url: str, **kwargs: Any) -> dict[str, object]:
            self.assertIn("dbpedia", url)
            self.assertEqual(kwargs["params"]["query"], "Feature Store")
            return {
                "docs": [
                    {
                        "resource": ["http://dbpedia.org/resource/Feature_store"],
                        "label": ["Feature Store"],
                        "comment": ["Feature stores manage ML features."],
                    }
                ]
            }

        batch = batch_with_entities("Feature Store")
        resolved = DBpediaLinker(transport=transport).resolve(batch)
        entity = resolved.entities[0]

        self.assertEqual(entity.properties["dbpedia_id"], "http://dbpedia.org/resource/Feature_store")
        self.assertEqual(entity.properties["dbpedia_label"], "Feature Store")
        self.assertEqual(resolved.metadata["external_linking_audit"][0]["provider"], "dbpedia")

    def test_openrefine_reconciler_links_and_merges_same_external_match(self) -> None:
        def transport(url: str, **kwargs: Any) -> dict[str, object]:
            self.assertEqual(url, "http://localhost:3333/reconcile")
            body = kwargs["body"].decode("utf-8")
            queries = json.loads(urllib.parse.parse_qs(body)["queries"][0])
            self.assertEqual(len(queries), 2)
            return {
                "q0": {
                    "result": [
                        {
                            "id": "wd:Q37156",
                            "name": "International Business Machines",
                            "score": 98,
                            "match": True,
                        }
                    ]
                },
                "q1": {
                    "result": [
                        {
                            "id": "wd:Q37156",
                            "name": "International Business Machines",
                            "score": 97,
                            "match": True,
                        }
                    ]
                },
            }

        batch = batch_with_entities("IBM", "International Business Machines", entity_type="Organization")
        resolved = OpenRefineReconciler(
            OpenRefineReconcilerConfig(endpoint_url="http://localhost:3333/reconcile"),
            transport=transport,
        ).resolve(batch)

        canonical_ids = {entity.properties["canonical_entity_id"] for entity in resolved.entities}
        self.assertEqual(len(canonical_ids), 1)
        self.assertEqual(resolved.entities[0].properties["openrefine_id"], "wd:Q37156")
        self.assertEqual(resolved.metadata["entity_resolution_audit"][0]["resolver"], "openrefine_reconciler")

    def test_embedding_resolver_merges_semantically_similar_entities(self) -> None:
        batch = batch_with_entities("Recommendation API", "Recommender API", "Feature Store")
        resolved = EmbeddingResolver(
            EmbeddingResolverConfig(threshold=0.95, review_threshold=0.99),
            model=FakeEmbeddingModel(),
        ).resolve(batch)

        recommendation_ids = {
            entity.properties.get("canonical_entity_id")
            for entity in resolved.entities
            if "API" in entity.name
        }
        feature_store = next(entity for entity in resolved.entities if entity.name == "Feature Store")
        self.assertEqual(len(recommendation_ids), 1)
        self.assertTrue(all(entity.properties.get("review_required") for entity in resolved.entities if "API" in entity.name))
        self.assertNotIn("canonical_entity_id", feature_store.properties)

    def test_cross_encoder_resolver_merges_high_scoring_pairs(self) -> None:
        batch = batch_with_entities("Recommendation API", "Recommender API", "Feature Store")
        resolved = CrossEncoderResolver(
            CrossEncoderResolverConfig(threshold=0.9, review_threshold=0.98),
            model=FakeCrossEncoderModel(),
        ).resolve(batch)

        recommendation_ids = {
            entity.properties.get("canonical_entity_id")
            for entity in resolved.entities
            if "API" in entity.name
        }
        self.assertEqual(len(recommendation_ids), 1)
        self.assertEqual(resolved.metadata["entity_resolution_audit"][0]["method"], "cross_encoder_pair_scoring")

    def test_registry_marks_entity_linking_upgrades_as_implemented(self) -> None:
        registry = default_registry()
        for name in {
            "wikidata_linker",
            "dbpedia_linker",
            "openrefine_reconciler",
            "embedding_resolver",
            "cross_encoder_resolver",
        }:
            self.assertEqual(registry.get(name).status, "implemented")


if __name__ == "__main__":
    unittest.main()
