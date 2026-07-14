from __future__ import annotations

import unittest

from kg_construction import SUPPORT_LEVEL_DEFINITIONS, default_registry


class RegistrySupportLevelTests(unittest.TestCase):
    def test_public_support_levels_are_available_for_all_techniques(self) -> None:
        registry = default_registry()
        levels = {technique.public_support_level for technique in registry.techniques}

        self.assertEqual(
            set(SUPPORT_LEVEL_DEFINITIONS),
            {"stable", "optional_adapter", "experimental", "planned"},
        )
        self.assertTrue({"stable", "optional_adapter", "experimental", "planned"} <= levels)
        self.assertTrue(all(technique.public_support_label for technique in registry.techniques))

    def test_internal_status_maps_to_public_support_level(self) -> None:
        registry = default_registry()

        self.assertEqual(registry.get("csv_loader").status, "implemented")
        self.assertEqual(registry.get("csv_loader").public_support_level, "stable")
        self.assertEqual(registry.get("rebel_relation_extractor").status, "adapter")
        self.assertEqual(registry.get("rebel_relation_extractor").public_support_level, "optional_adapter")
        self.assertEqual(registry.get("chunk_embedding_enricher").status, "stub")
        self.assertEqual(registry.get("chunk_embedding_enricher").public_support_level, "planned")

    def test_research_and_service_backed_features_are_not_labeled_core_stable(self) -> None:
        registry = default_registry()

        self.assertEqual(registry.get("graph_retrieval_benchmark").status, "implemented")
        self.assertEqual(registry.get("graph_retrieval_benchmark").public_support_level, "experimental")
        self.assertEqual(registry.get("ollama_graph_extractor").status, "implemented")
        self.assertEqual(registry.get("ollama_graph_extractor").public_support_level, "optional_adapter")
        self.assertEqual(registry.get("neo4j_ingestion_adapter").status, "implemented")
        self.assertEqual(registry.get("neo4j_ingestion_adapter").public_support_level, "optional_adapter")


if __name__ == "__main__":
    unittest.main()
