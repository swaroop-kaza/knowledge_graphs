from __future__ import annotations

import json
import unittest

from kg_construction.certification.registry import default_registry, research_registry
from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.extractors.relations import (
    DependencyPatternRelationExtractor,
    GlinerOllamaRelationTypingExtractor,
    OpenIERelationExtractor,
    RebelRelationExtractor,
    RelationTypingConfig,
)
from kg_construction.infrastructure.llm.ollama import OllamaConfig, OllamaProvider


class FakeGlinerModel:
    def predict_entities(
        self,
        text: str,
        labels: list[str],
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
    ) -> list[dict[str, object]]:
        _ = threshold, flat_ner
        spans = {
            "Incident": "INCIDENT-1042",
            "System": "Recommendation API",
            "Date": "2026-07-09",
        }
        entities: list[dict[str, object]] = []
        for label, value in spans.items():
            if label not in labels:
                continue
            start = text.find(value)
            if start >= 0:
                entities.append(
                    {
                        "text": value,
                        "label": label,
                        "start": start,
                        "end": start + len(value),
                        "score": 0.92,
                    }
                )
        return entities


class FakeRebelPipeline:
    def __call__(self, text: str, **kwargs: object) -> list[dict[str, str]]:
        _ = text, kwargs
        return [
            {
                "generated_text": (
                    "<triplet> Ada Lovelace <subj> Analytical Engine <obj> created"
                )
            }
        ]


class RelationExtractionTests(unittest.TestCase):
    def test_openie_extractor_emits_evidence_backed_relation_candidates(self) -> None:
        source = SourceRecord.from_text(
            "Ada Lovelace works at Analytical Engine Lab.",
            source_name="memory://openie",
        )

        batch = OpenIERelationExtractor().extract(source)

        self.assertEqual(len(batch.relations), 1)
        relation = batch.relations[0]
        self.assertEqual(relation.relation_type, "WORKS_AT")
        self.assertEqual(relation.properties["source_name"], "Ada Lovelace")
        self.assertEqual(relation.properties["target_name"], "Analytical Engine Lab")
        self.assertEqual(relation.evidence[0].text, "Ada Lovelace works at Analytical Engine Lab")
        self.assertEqual(len(batch.entities), 2)

    def test_dependency_pattern_extractor_emits_configured_relation_types(self) -> None:
        source = SourceRecord.from_text(
            "Grace Hopper reports to Ada Lovelace.",
            source_name="memory://dependency-pattern",
        )

        batch = DependencyPatternRelationExtractor().extract(source)

        self.assertEqual(len(batch.relations), 1)
        relation = batch.relations[0]
        self.assertEqual(relation.relation_type, "REPORTS_TO")
        self.assertEqual(relation.properties["dependency_pattern"], "reports_to")
        self.assertEqual(relation.properties["source_name"], "Grace Hopper")
        self.assertEqual(relation.properties["target_name"], "Ada Lovelace")

    def test_gliner_entities_plus_ollama_relation_typing(self) -> None:
        text = "INCIDENT-1042 affected Recommendation API on 2026-07-09."
        source = SourceRecord.from_text(text, source_name="memory://gliner-ollama-relations")

        def transport(body: dict[str, object]) -> dict[str, object]:
            prompt = str(body["messages"][0]["content"])  # type: ignore[index]
            self.assertIn("already extracted entity spans", prompt)
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "relations": [
                                {
                                    "source": "INCIDENT-1042",
                                    "type": "AFFECTED",
                                    "target": "Recommendation API",
                                    "properties": {"severity": "customer-facing"},
                                    "evidence_text": "INCIDENT-1042 affected Recommendation API",
                                    "confidence": 0.94,
                                }
                            ]
                        }
                    )
                }
            }

        provider = OllamaProvider(OllamaConfig(), transport=transport)
        config = RelationTypingConfig(
            entity_labels=["Incident", "System", "Date"],
            allowed_relationship_types=["AFFECTED", "RELATED_TO"],
        )
        batch = GlinerOllamaRelationTypingExtractor(
            provider,
            config=config,
            gliner_model=FakeGlinerModel(),
        ).extract(source)

        self.assertEqual([entity.name for entity in batch.entities], ["INCIDENT-1042", "Recommendation API", "2026-07-09"])
        self.assertEqual(len(batch.relations), 1)
        relation = batch.relations[0]
        self.assertEqual(relation.relation_type, "AFFECTED")
        self.assertEqual(relation.properties["source_name"], "INCIDENT-1042")
        self.assertEqual(relation.properties["target_name"], "Recommendation API")
        self.assertEqual(relation.properties["prompt_version"], "ollama_relation_typing_v1")
        self.assertEqual(relation.evidence[0].text, "INCIDENT-1042 affected Recommendation API")

    def test_relation_typing_config_loads_from_yaml(self) -> None:
        config = RelationTypingConfig.from_yaml("configs/gliner_ollama_relation_typing.example.yaml")

        self.assertIn("Incident", config.entity_labels)
        self.assertIn("AFFECTED", config.allowed_relationship_types)
        self.assertEqual(config.schema_version, "v1")

    def test_rebel_adapter_parses_generated_triples(self) -> None:
        source = SourceRecord.from_text(
            "Ada Lovelace created Analytical Engine.",
            source_name="memory://rebel",
        )

        batch = RebelRelationExtractor(pipeline=FakeRebelPipeline()).extract(source)

        self.assertEqual(len(batch.relations), 1)
        relation = batch.relations[0]
        self.assertEqual(relation.relation_type, "CREATED")
        self.assertEqual(relation.properties["source_name"], "Ada Lovelace")
        self.assertEqual(relation.properties["target_name"], "Analytical Engine")
        self.assertTrue(relation.evidence)

    def test_registry_marks_relation_extractors_as_implemented_or_adapter(self) -> None:
        registry = default_registry()

        self.assertEqual(registry.get("openie_relation_extractor").status, "implemented")
        self.assertEqual(registry.get("dependency_pattern_relation_extractor").status, "implemented")
        self.assertEqual(registry.get("gliner_ollama_relation_typing_extractor").status, "implemented")
        self.assertEqual(registry.get("evidence_first_unstructured_pipeline").status, "implemented")
        self.assertNotIn("transformer_relation_extractor", registry.names())
        self.assertEqual(
            research_registry().get("transformer_relation_extractor").status,
            "adapter",
        )
        # REBEL is removed from the product registry: its common model is non-commercial.
        with self.assertRaises(KeyError):
            registry.get("rebel_relation_extractor")


if __name__ == "__main__":
    unittest.main()
