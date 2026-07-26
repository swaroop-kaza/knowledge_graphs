from __future__ import annotations

import unittest
from pathlib import Path

from kg_construction.certification.registry import default_registry
from kg_construction.construction.profiling import DatasetProfiler
from kg_construction.infrastructure.loaders import (
    APIResponseLoader,
    ChatTranscriptLoader,
    EmailLoader,
    ExcelLoader,
    HTMLLoader,
    LogLoader,
    MarkdownLoader,
    PDFLoader,
    ReportLoader,
    ScientificPaperLoader,
    SQLDatabaseLoader,
    WebPageLoader,
    XMLLoader,
)
from kg_construction.infrastructure.mapping import HTMLMetadataMapper, LogEventMapper

FIXTURES = Path("tests/fixtures/enterprise_loaders")


def assert_source_evidence(testcase: unittest.TestCase, metadata: dict[str, object]) -> None:
    evidence = metadata.get("source_evidence")
    testcase.assertIsInstance(evidence, list)
    testcase.assertTrue(evidence)
    first = evidence[0] if isinstance(evidence, list) else {}
    testcase.assertIn("evidence_id", first)
    testcase.assertIn("location", first)


class EnterpriseLoaderTests(unittest.TestCase):
    def test_excel_loader_emits_sheet_records_with_stable_ids_and_evidence(self) -> None:
        path = FIXTURES / "service_inventory.xlsx"
        first = ExcelLoader().load(path)
        second = ExcelLoader().load(path)
        services = next(source for source in first if source.metadata["sheet_name"] == "Services")

        self.assertEqual([source.source_id for source in first], [source.source_id for source in second])
        self.assertEqual(len(first), 2)
        self.assertEqual(services.source_type, "excel")
        self.assertEqual(services.metadata["row_count"], 2)
        self.assertEqual(services.records[0]["name"], "Recommendation API")
        assert_source_evidence(self, services.metadata)

    def test_sql_loader_emits_table_records_with_schema_metadata_and_evidence(self) -> None:
        path = FIXTURES / "enterprise.sqlite"
        sources = SQLDatabaseLoader().load(path)
        services = next(source for source in sources if source.metadata["table"] == "services")
        incidents = next(source for source in sources if source.metadata["table"] == "incidents")

        self.assertEqual(services.source_type, "sql")
        self.assertEqual(services.metadata["primary_key"], ["service_id"])
        self.assertEqual(services.records[0]["name"], "Recommendation API")
        self.assertEqual(incidents.metadata["foreign_keys"][0]["referred_table"], "services")
        self.assertEqual(
            [source.source_id for source in sources],
            [source.source_id for source in SQLDatabaseLoader().load(path)],
        )
        assert_source_evidence(self, services.metadata)

    def test_api_response_loader_emits_rows_from_data_array_with_evidence(self) -> None:
        source = APIResponseLoader().load(FIXTURES / "api_response.json")[0]

        self.assertEqual(source.source_type, "api_response")
        self.assertEqual(source.metadata["row_count"], 2)
        self.assertEqual(source.records[0]["service_id"], "svc-rec-api")
        self.assertEqual(source.metadata["json_payload"]["request_id"], "req-2026-07-09-1042")
        assert_source_evidence(self, source.metadata)

    def test_xml_loader_flattens_repeated_elements_with_path_evidence(self) -> None:
        source = XMLLoader().load(FIXTURES / "services.xml")[0]

        self.assertEqual(source.source_type, "xml")
        self.assertEqual(source.metadata["root_tag"], "services")
        self.assertEqual(source.metadata["row_count"], 2)
        self.assertEqual(source.records[0]["service.@id"], "svc-rec-api")
        self.assertEqual(source.records[0]["service.name"], "Recommendation API")
        self.assertIn("/services/service[1]", source.metadata["row_paths"])
        assert_source_evidence(self, source.metadata)

    def test_markdown_loader_preserves_headings_and_text_evidence(self) -> None:
        source = MarkdownLoader().load(FIXTURES / "incident_review.md")[0]

        self.assertEqual(source.source_type, "markdown")
        self.assertIn("INCIDENT-1042", source.content or "")
        self.assertEqual(source.metadata["heading_count"], 3)
        self.assertEqual(source.metadata["headings"][0]["title"], "Incident Review")
        assert_source_evidence(self, source.metadata)

    def test_report_loader_preserves_report_sections_and_text_evidence(self) -> None:
        source = ReportLoader().load(FIXTURES / "ops_report.report")[0]

        self.assertEqual(source.source_type, "report")
        self.assertIn("Recommendation API", source.content or "")
        self.assertGreaterEqual(source.metadata["section_count"], 2)
        self.assertEqual(source.metadata["sections"][0]["title"], "Operational Risk Report")
        assert_source_evidence(self, source.metadata)

    def test_pdf_loader_extracts_text_and_page_evidence(self) -> None:
        source = PDFLoader().load(FIXTURES / "incident_brief.pdf")[0]

        self.assertEqual(source.source_type, "pdf")
        self.assertIn("INCIDENT-1042", source.content or "")
        self.assertEqual(source.metadata["page_count"], 1)
        self.assertEqual(source.metadata["pages"][0]["page_number"], 1)
        assert_source_evidence(self, source.metadata)

    def test_email_loader_extracts_headers_body_and_evidence(self) -> None:
        source = EmailLoader().load(FIXTURES / "incident_email.eml")[0]

        self.assertEqual(source.source_type, "email")
        self.assertIn("Recommendation API", source.content or "")
        self.assertEqual(source.metadata["headers"]["subject"], "INCIDENT-1042 Recommendation API follow-up")
        self.assertEqual(source.metadata["headers"]["message_id"], "<incident-1042@example.com>")
        assert_source_evidence(self, source.metadata)

    def test_chat_transcript_loader_emits_turn_records_with_evidence(self) -> None:
        source = ChatTranscriptLoader().load(FIXTURES / "support_chat.chat")[0]

        self.assertEqual(source.source_type, "chat_transcript")
        self.assertEqual(source.metadata["message_count"], 3)
        self.assertEqual(source.records[0]["speaker"], "Asha")
        self.assertEqual(source.records[1]["message"], "INCIDENT-1042 is linked to the Feature Store timeout.")
        self.assertEqual(source.metadata["speakers"], ["Asha", "Vikram"])
        assert_source_evidence(self, source.metadata)

    def test_html_loader_extracts_metadata_links_jsonld_and_text(self) -> None:
        source = HTMLLoader().load(FIXTURES / "recommendation_runbook.html")[0]

        self.assertEqual(source.source_type, "html")
        self.assertIn("INCIDENT-1042 affected Recommendation API", source.content or "")
        self.assertEqual(source.metadata["title"], "Recommendation API Runbook")
        self.assertEqual(source.metadata["description"], "Operational runbook for Recommendation API incidents.")
        self.assertEqual(source.metadata["canonical_url"], "https://example.com/runbooks/recommendation-api")
        html_metadata = source.metadata["html_metadata"]
        self.assertEqual(len(html_metadata["meta_tags"]), 3)
        self.assertEqual(html_metadata["json_ld"][0]["@type"], "TechArticle")
        assert_source_evidence(self, source.metadata)

    def test_webpage_loader_uses_webpage_source_type_for_local_html_fixture(self) -> None:
        source = WebPageLoader().load(FIXTURES / "recommendation_runbook.html")[0]

        self.assertEqual(source.source_type, "webpage")
        self.assertEqual(source.metadata["loader"], "webpage_loader")
        self.assertEqual(source.metadata["title"], "Recommendation API Runbook")
        assert_source_evidence(self, source.metadata)

    def test_log_loader_parses_json_apache_key_value_and_generic_logs(self) -> None:
        source = LogLoader().load(FIXTURES / "service_events.log")[0]

        self.assertEqual(source.source_type, "logs")
        self.assertEqual(source.metadata["row_count"], 4)
        self.assertEqual(source.records[0]["parser"], "apache_combined")
        self.assertEqual(source.records[0]["status"], "503")
        self.assertEqual(source.records[1]["parser"], "json")
        self.assertEqual(source.records[1]["incident_id"], "INCIDENT-1042")
        self.assertEqual(source.records[2]["parser"], "key_value")
        self.assertEqual(source.records[3]["parser"], "generic")
        assert_source_evidence(self, source.metadata)

    def test_scientific_paper_loader_extracts_jats_metadata_sections_and_citations(self) -> None:
        source = ScientificPaperLoader().load(FIXTURES / "kg_construction_article.jats")[0]

        self.assertEqual(source.source_type, "scientific_paper")
        self.assertEqual(source.metadata["paper_format"], "jats")
        self.assertEqual(source.metadata["title"], "Evidence-backed Data-to-Knowledge Graph Construction")
        self.assertEqual(source.metadata["doi"], "10.1234/kg.2026.001")
        self.assertEqual(source.metadata["authors"], ["Asha Rao", "Swaroop Kaza"])
        self.assertEqual(len(source.metadata["sections"]), 2)
        self.assertEqual(source.metadata["citations"][0]["doi"], "10.1145/3447772")
        self.assertIn("Enterprise data contains hidden graph-like structure", source.content or "")
        assert_source_evidence(self, source.metadata)

    def test_html_metadata_mapper_builds_document_metadata_link_and_jsonld_graph(self) -> None:
        source = HTMLLoader().load(FIXTURES / "recommendation_runbook.html")[0]
        batch = HTMLMetadataMapper().map_source(source)

        self.assertEqual(batch.metadata["meta_tag_count"], 3)
        self.assertTrue(any(entity.entity_type == "HTMLDocument" for entity in batch.entities))
        self.assertTrue(any(entity.entity_type == "HTMLMetaTag" for entity in batch.entities))
        self.assertTrue(any(entity.entity_type == "SchemaOrgEntity" for entity in batch.entities))
        self.assertTrue(any(relation.relation_type == "HAS_STRUCTURED_DATA" for relation in batch.relations))
        self.assertTrue(all(relation.evidence for relation in batch.relations))

    def test_log_event_mapper_builds_event_graph_candidates(self) -> None:
        source = LogLoader().load(FIXTURES / "service_events.log")[0]
        batch = LogEventMapper().map_source(source)

        self.assertEqual(batch.metadata["event_count"], 4)
        self.assertEqual(len([entity for entity in batch.entities if entity.entity_type == "LogEvent"]), 4)
        self.assertTrue(any(entity.name == "Recommendation API" for entity in batch.entities))
        self.assertTrue(any(entity.name == "/recommendations" for entity in batch.entities))
        self.assertTrue(any(relation.relation_type == "REQUESTED_ENDPOINT" for relation in batch.relations))
        self.assertTrue(all(entity.evidence for entity in batch.entities))

    def test_profiler_and_registry_know_enterprise_loaders_are_implemented(self) -> None:
        profile = DatasetProfiler().profile_path(FIXTURES / "service_inventory.xlsx")
        registry = default_registry()

        self.assertEqual(profile.file_format, "xlsx")
        self.assertEqual(profile.data_type, "structured")
        for name in {
            "api_response_loader",
            "chat_transcript_loader",
            "email_loader",
            "excel_loader",
            "html_loader",
            "log_loader",
            "markdown_loader",
            "pdf_loader",
            "report_loader",
            "scientific_paper_loader",
            "sql_database_loader",
            "webpage_loader",
            "xml_loader",
        }:
            self.assertEqual(registry.get(name).status, "implemented")
        for name in {"html_metadata_mapper", "log_event_mapper"}:
            self.assertEqual(registry.get(name).status, "implemented")


if __name__ == "__main__":
    unittest.main()
