from __future__ import annotations

import unittest

import kg_construction.application.registry as legacy_registry
import kg_construction.certification.registry as certification_registry
from kg_construction.certification import (
    SUPPORT_LEVEL_DEFINITIONS,
    default_registry,
    research_registry,
)


class RegistrySupportLevelTests(unittest.TestCase):
    def test_application_registry_is_an_identity_preserving_compatibility_shim(
        self,
    ) -> None:
        self.assertEqual(legacy_registry.__all__, certification_registry.__all__)
        for name in certification_registry.__all__:
            self.assertIs(
                getattr(legacy_registry, name),
                getattr(certification_registry, name),
                name,
            )

    def test_release_registry_contains_only_mandatory_techniques(self) -> None:
        registry = default_registry()
        levels = {technique.public_support_level for technique in registry.techniques}

        self.assertEqual(set(SUPPORT_LEVEL_DEFINITIONS), {"mandatory", "research", "roadmap"})
        self.assertEqual(levels, {"mandatory"})
        self.assertTrue(all(technique.status == "implemented" for technique in registry.techniques))
        self.assertTrue(all(technique.adapter is not None for technique in registry.techniques))
        self.assertTrue(all(technique.public_support_label for technique in registry.techniques))

    def test_research_inventory_keeps_non_release_work_visible(self) -> None:
        registry = research_registry()

        self.assertEqual(registry.get("csv_loader").status, "implemented")
        self.assertEqual(registry.get("csv_loader").public_support_level, "mandatory")
        self.assertEqual(registry.get("transformer_relation_extractor").status, "adapter")
        self.assertEqual(
            registry.get("transformer_relation_extractor").public_support_level,
            "research",
        )
        self.assertEqual(registry.get("chunk_embedding_enricher").status, "stub")
        self.assertEqual(registry.get("chunk_embedding_enricher").public_support_level, "roadmap")

    def test_downstream_graph_enrichment_is_outside_the_release_registry(self) -> None:
        release = default_registry()
        research = research_registry()

        self.assertNotIn("graph_retrieval_benchmark", release.names())
        self.assertIn("graph_retrieval_benchmark", research.names())
        # Service/model-backed construction is mandatory in the release contract.
        self.assertEqual(release.get("ollama_graph_extractor").public_support_level, "mandatory")
        self.assertEqual(release.get("neo4j_ingestion_adapter").public_support_level, "mandatory")


if __name__ == "__main__":
    unittest.main()
