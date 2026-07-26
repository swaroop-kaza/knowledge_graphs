#!/usr/bin/env python3
"""Run implemented Data-to-KG techniques across downloaded experiment datasets."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kg_construction.construction.profiling import DatasetProfiler  # noqa: E402
from kg_construction.construction.workflow import (  # noqa: E402
    ConstructionResult,
    ConstructionWorkflow,
)
from kg_construction.domain.models import ConstructionBatch, SourceRecord  # noqa: E402
from kg_construction.domain.schema import (  # noqa: E402
    GraphSchema,
    NodeTypeDefinition,
    RelationshipTypeDefinition,
)
from kg_construction.infrastructure.extractors.gazetteer import (
    GazetteerEntityExtractor,  # noqa: E402
)
from kg_construction.infrastructure.extractors.gliner import (  # noqa: E402
    GlinerConfig,
    GlinerEntityExtractor,
)
from kg_construction.infrastructure.extractors.ollama_graph import (
    OllamaGraphExtractor,  # noqa: E402
)
from kg_construction.infrastructure.extractors.regex import RegexEntityExtractor  # noqa: E402
from kg_construction.infrastructure.llm.label_proposal import (  # noqa: E402
    LabelProposalSettings,
    OllamaLabelProposalExtractor,
)
from kg_construction.infrastructure.loaders import (  # noqa: E402
    EmailLoader,
    XMLLoader,
)
from kg_construction.infrastructure.loaders.csv_loader import CsvLoader  # noqa: E402
from kg_construction.infrastructure.loaders.enterprise import SQLDatabaseLoader  # noqa: E402
from kg_construction.infrastructure.loaders.json_loader import JsonLoader  # noqa: E402
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured  # noqa: E402
from kg_construction.infrastructure.loaders.text_loader import TextLoader  # noqa: E402
from kg_construction.infrastructure.mapping.direct_table_mapper import (
    DirectTableMapper,  # noqa: E402
)
from kg_construction.infrastructure.mapping.semi_structured import (  # noqa: E402
    DottedPathJsonMapper,
    JsonPathMapper,
)
from kg_construction.infrastructure.ontology.schema_proposal import (  # noqa: E402
    SchemaProposalWorkflow,
)
from kg_construction.infrastructure.ontology.shacl import ShaclValidator  # noqa: E402
from kg_construction.infrastructure.ontology.validation import (
    EvidenceRequiredValidator,  # noqa: E402
)
from kg_construction.infrastructure.resolution.entity_resolution import (  # noqa: E402
    AliasTableResolver,
    DedupeActiveLearningResolver,
    NormalizedStringResolver,
    RapidFuzzResolver,
    SplinkProbabilisticResolver,
)
from kg_construction.research.benchmark_runner import (  # noqa: E402
    EvaluationSettings,
    TechniqueEvaluationConfig,
    TechniqueEvaluationRunner,
)

DATA_DIR = ROOT / "data" / "external" / "experiment_datasets"
REPORT_DIR = ROOT / "reports" / "experiment_datasets"
GRAPH_DIR = REPORT_DIR / "graphs"
SCHEMA_DIR = REPORT_DIR / "schemas"
BENCHMARK_DIR = REPORT_DIR / "benchmarks"

DEFAULT_GAZETTEER = {
    "Organization": [
        "Apple",
        "Enron",
        "NYC",
        "SEC",
        "PubMed",
        "DBLP",
        "Hugging Face",
        "Recommendation API",
        "NameNode",
        "DataNode",
    ],
    "System": ["HDFS", "API", "Recommendation API", "NameNode", "DataNode"],
    "Dataset": ["Chinook", "WikiANN", "Loghub", "PubMed", "DBLP"],
    "Incident": ["INCIDENT-1042"],
}

ALIAS_TABLE = {
    "Recommendation API": ["recommendation api", "Recommendations API", "Recs API"],
    "New York City": ["NYC", "N.Y.C."],
    "HDFS": ["hdfs", "Hadoop Distributed File System"],
}

DATASET_SOURCE_REFERENCES = [
    [
        "chinook_sqlite",
        "Chinook relational sample database",
        "https://github.com/lerocha/chinook-database",
        "Downloaded SQLite file from the public Chinook repository.",
    ],
    [
        "w3c_rdb2rdf_d001",
        "W3C RDB2RDF direct mapping fixture",
        "https://www.w3.org/TR/rdb2rdf-test-cases/",
        "Downloaded D001 SQL/manifest/Turtle fixture and generated a local SQLite database.",
    ],
    [
        "nyc_311_service_requests",
        "NYC 311 Service Requests from 2020 to present",
        "https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9",
        "Downloaded a bounded CSV sample through the Socrata API.",
    ],
    [
        "sec_apple_companyfacts",
        "SEC EDGAR companyfacts API",
        "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
        "Downloaded Apple companyfacts JSON and generated a bounded concept/fact sample.",
    ],
    [
        "pubmed_baseline",
        "PubMed baseline XML",
        "https://pubmed.ncbi.nlm.nih.gov/download/",
        "Stream-sampled articles from the first PubMed 2026 baseline XML gzip file.",
    ],
    [
        "dblp_publication_api",
        "DBLP publication API",
        "https://dblp.org/faq/How+can+I+download+the+whole+dblp+dataset",
        "Downloaded a DBLP API JSON search sample; full XML dump is intentionally not mirrored.",
    ],
    [
        "loghub_hdfs",
        "Loghub HDFS logs",
        "https://github.com/logpai/loghub",
        "Downloaded HDFS 2k raw logs and structured CSV.",
    ],
    [
        "wikiann_en_ner",
        "WikiANN English NER",
        "https://huggingface.co/datasets/unimelb-nlp/wikiann",
        "Downloaded English parquet and generated bounded text/JSONL samples with gold spans.",
    ],
    [
        "enron_email_subset",
        "Enron email corpus context",
        "https://www.cs.cmu.edu/~enron/",
        "Full CMU archive is not mirrored; a bounded Enron-derived text subset is used for loader/extractor tests.",
    ],
]


@dataclass(frozen=True)
class DatasetRun:
    dataset_id: str
    display_name: str
    path: Path
    loader: str
    benchmark_source_type: str | None
    notes: str
    enabled: bool = True


DATASET_RUNS = [
    DatasetRun(
        "chinook_sqlite",
        "Chinook relational SQLite",
        DATA_DIR / "chinook" / "Chinook_Sqlite.sqlite",
        "sqlite",
        None,
        "Relational source for SQL loading, schema proposal, and direct table mapping.",
    ),
    DatasetRun(
        "w3c_rdb2rdf_d001",
        "W3C RDB2RDF D001 SQLite fixture",
        DATA_DIR / "w3c_rdb2rdf" / "D001-1table1column1row" / "D001.sqlite",
        "sqlite",
        None,
        "Standards fixture for direct relational-to-RDF mapping checks.",
    ),
    DatasetRun(
        "nyc_311_service_requests",
        "NYC 311 service requests CSV",
        DATA_DIR / "nyc_311" / "nyc_311_service_requests_sample.csv",
        "csv",
        "csv",
        "Tabular operational data with IDs, dates, categories, and status fields.",
    ),
    DatasetRun(
        "sec_apple_companyfacts",
        "SEC Apple companyfacts JSON sample",
        DATA_DIR / "sec" / "apple_companyfacts_sample.json",
        "json",
        "json",
        "Nested API-style XBRL JSON for JSONPath and array-to-edge mapping.",
    ),
    DatasetRun(
        "pubmed_baseline",
        "PubMed XML article sample",
        DATA_DIR / "pubmed" / "pubmed26n0001_sample.xml",
        "xml",
        "xml",
        "Scientific-paper metadata XML for XML loading and schema proposal.",
    ),
    DatasetRun(
        "dblp_publication_api",
        "DBLP publication API JSON sample",
        DATA_DIR / "dblp" / "dblp_knowledge_graph_publications.json",
        "json",
        "json",
        "Semi-structured publication-search API payload.",
    ),
    DatasetRun(
        "loghub_hdfs_raw",
        "Loghub HDFS raw logs",
        DATA_DIR / "loghub" / "HDFS_2k.log",
        "text",
        "txt",
        "Raw system logs for regex extraction and text benchmark paths.",
    ),
    DatasetRun(
        "loghub_hdfs_structured",
        "Loghub HDFS structured CSV",
        DATA_DIR / "loghub" / "HDFS_2k.log_structured.csv",
        "csv",
        "csv",
        "Parsed log events for tabular profiling and direct graph construction.",
    ),
    DatasetRun(
        "wikiann_en_ner",
        "WikiANN English NER sample text",
        DATA_DIR / "wikiann" / "wikiann_en_sample.txt",
        "text",
        "txt",
        "NER benchmark text for GLiNER adapter and label-proposal workflow.",
    ),
    DatasetRun(
        "enron_email_subset_text",
        "Enron-derived bounded email text corpus",
        DATA_DIR / "enron" / "enron_sample_corpus.txt",
        "text",
        "txt",
        "Email text corpus for text extraction, regex, and gazetteer checks.",
    ),
    DatasetRun(
        "enron_email_subset_eml",
        "Enron-derived wrapped EML sample",
        DATA_DIR / "enron" / "enron_wrapped_sample.eml",
        "email",
        "eml",
        "RFC 5322 wrapped sample for EmailLoader coverage.",
    ),
]


def model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [model_dump(item) for item in value]
    if isinstance(value, dict):
        return {str(key): model_dump(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model_dump(payload), indent=2, sort_keys=True), encoding="utf-8")


def load_sources(dataset: DatasetRun, *, sql_limit: int) -> list[SourceRecord]:
    if dataset.loader == "sqlite":
        return SQLDatabaseLoader(limit=sql_limit).load(dataset.path)
    if dataset.loader == "csv":
        return CsvLoader().load(dataset.path)
    if dataset.loader == "json":
        return JsonLoader().load(dataset.path)
    if dataset.loader == "xml":
        return XMLLoader().load(dataset.path)
    if dataset.loader == "email":
        return EmailLoader().load(dataset.path)
    return TextLoader().load(dataset.path)


def source_summary(sources: list[SourceRecord]) -> dict[str, Any]:
    return {
        "source_count": len(sources),
        "record_count": sum(len(source.records) for source in sources),
        "text_character_count": sum(len(source.content or "") for source in sources),
        "source_types": sorted({source.source_type for source in sources}),
        "source_names": [source.source_name for source in sources[:5]],
    }


def run_workflow(
    *,
    dataset_id: str,
    technique_name: str,
    sources: list[SourceRecord],
    mappers: list[Any] | None = None,
    extractors: list[Any] | None = None,
) -> tuple[ConstructionResult | None, dict[str, Any]]:
    started = time.perf_counter()
    try:
        workflow = ConstructionWorkflow(
            mappers=mappers or [],
            extractors=extractors or [],
            resolvers=[
                NormalizedStringResolver(),
                AliasTableResolver(ALIAS_TABLE),
            ],
            validators=[EvidenceRequiredValidator()],
            pipeline_name=f"{dataset_id}_{technique_name}",
        )
        result = workflow.run(sources)
        elapsed = time.perf_counter() - started
        summary = summarize_result(technique_name, result, elapsed)
        write_graph_artifacts(dataset_id, technique_name, result)
        shacl_summary = validate_with_generated_shacl(dataset_id, technique_name, result)
        summary["shacl"] = shacl_summary
        return result, summary
    except Exception as exc:
        return None, {
            "technique": technique_name,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }


def summarize_result(technique_name: str, result: ConstructionResult, elapsed: float) -> dict[str, Any]:
    batch = result.batch
    object_count = len(batch.entities) + len(batch.relations) + len(batch.facts)
    evidence_backed = sum(1 for entity in batch.entities if entity.evidence)
    evidence_backed += sum(1 for relation in batch.relations if relation.evidence)
    evidence_backed += sum(1 for fact in batch.facts if fact.evidence)
    labels = Counter(entity.entity_type for entity in batch.entities)
    relation_types = Counter(relation.relation_type for relation in batch.relations)
    merge_audit = batch.metadata.get("entity_resolution_audit", [])
    return {
        "technique": technique_name,
        "status": "ok",
        "entity_count": len(batch.entities),
        "relation_count": len(batch.relations),
        "fact_count": len(batch.facts),
        "property_graph_nodes": len(result.property_graph.nodes),
        "property_graph_edges": len(result.property_graph.edges),
        "rdf_triples": len(result.rdf_graph.triples),
        "rdf_quads": len(result.rdf_graph.quads),
        "validation_findings": len(result.validation_report.findings),
        "evidence_coverage": round(evidence_backed / max(1, object_count), 6),
        "unsupported_extraction_rate": round(1 - (evidence_backed / max(1, object_count)), 6),
        "top_entity_types": labels.most_common(8),
        "top_relation_types": relation_types.most_common(8),
        "entity_resolution_clusters": len(merge_audit) if isinstance(merge_audit, list) else 0,
        "metrics": result.metrics,
        "elapsed_seconds": round(elapsed, 6),
    }


def write_graph_artifacts(dataset_id: str, technique_name: str, result: ConstructionResult) -> None:
    directory = GRAPH_DIR / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = technique_name.replace("/", "_")
    (directory / f"{safe_name}.property_graph.json").write_text(
        result.property_graph.to_json(),
        encoding="utf-8",
    )
    write_json(directory / f"{safe_name}.rdf.json", result.rdf_graph.to_dict())
    (directory / f"{safe_name}.rdf.ttl").write_text(result.rdf_graph.serialize_turtle(), encoding="utf-8")
    (directory / f"{safe_name}.rdf.nq").write_text(result.rdf_graph.serialize_nquads(), encoding="utf-8")
    (directory / f"{safe_name}.evidence_graph.json").write_text(
        result.evidence_graph.to_json(),
        encoding="utf-8",
    )


def graph_schema_from_batch(batch: ConstructionBatch, *, schema_id: str) -> GraphSchema:
    node_types = [
        NodeTypeDefinition(name=name, status="promoted", confidence=1.0)
        for name in sorted({entity.entity_type for entity in batch.entities})
    ]
    relationship_types = [
        RelationshipTypeDefinition(name=name, status="promoted", confidence=1.0)
        for name in sorted({relation.relation_type for relation in batch.relations})
    ]
    return GraphSchema(
        schema_id=schema_id,
        version="experiment",
        status="promoted",
        node_types=node_types,
        relationship_types=relationship_types,
        metadata={"generated_from": "constructed_candidate_batch"},
    )


def validate_with_generated_shacl(
    dataset_id: str,
    technique_name: str,
    result: ConstructionResult,
) -> dict[str, Any]:
    if not result.rdf_graph.triples:
        return {"status": "skipped", "reason": "No RDF triples were produced."}
    schema = graph_schema_from_batch(
        result.batch,
        schema_id=f"{dataset_id}_{technique_name}_runtime_schema",
    )
    try:
        validation = ShaclValidator().validate_rdf_graph(result.rdf_graph, graph_schema=schema)
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    schema_dir = SCHEMA_DIR / dataset_id
    schema_dir.mkdir(parents=True, exist_ok=True)
    safe_name = technique_name.replace("/", "_")
    (schema_dir / f"{safe_name}.runtime_schema.shacl.ttl").write_text(
        validation.shacl_turtle,
        encoding="utf-8",
    )
    (schema_dir / f"{safe_name}.pyshacl_results.ttl").write_text(
        validation.results_graph_turtle,
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "conforms": validation.conforms,
        "results_text_excerpt": validation.results_text[:500],
    }


def propose_schema(dataset_id: str, sources: list[SourceRecord]) -> dict[str, Any]:
    if not any(source.records for source in sources):
        return {"status": "skipped", "reason": "No structured records were available."}
    try:
        proposal = SchemaProposalWorkflow().propose(sources)
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    directory = SCHEMA_DIR / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "schema_proposal.json", proposal.graph_schema)
    (directory / "schema_proposal.shacl.ttl").write_text(proposal.shacl_turtle, encoding="utf-8")
    if proposal.owl_turtle:
        (directory / "schema_proposal.owl.ttl").write_text(proposal.owl_turtle, encoding="utf-8")
    return {
        "status": "ok",
        "node_types": [node.name for node in proposal.graph_schema.node_types],
        "relationship_types": [rel.name for rel in proposal.graph_schema.relationship_types],
        "review_state": proposal.review_state,
        "shacl_path": str((directory / "schema_proposal.shacl.ttl").relative_to(ROOT)),
        "owl_path": str((directory / "schema_proposal.owl.ttl").relative_to(ROOT))
        if proposal.owl_turtle
        else None,
    }


def run_structured_techniques(dataset_id: str, sources: list[SourceRecord]) -> list[dict[str, Any]]:
    if not any(source.records for source in sources):
        return []
    rows: list[dict[str, Any]] = []
    _, direct_summary = run_workflow(
        dataset_id=dataset_id,
        technique_name="direct_table_mapper",
        sources=sources,
        mappers=[DirectTableMapper()],
    )
    rows.append(direct_summary)

    _, dotted_summary = run_workflow(
        dataset_id=dataset_id,
        technique_name="dotted_path_json_mapper",
        sources=sources,
        mappers=[DottedPathJsonMapper()],
    )
    rows.append(dotted_summary)
    return rows


def run_json_techniques(dataset_id: str, sources: list[SourceRecord]) -> list[dict[str, Any]]:
    if not any(source.source_type == "json" for source in sources):
        return []
    rows: list[dict[str, Any]] = []
    _, jsonpath_summary = run_workflow(
        dataset_id=dataset_id,
        technique_name="jsonpath_mapper",
        sources=sources,
        mappers=[JsonPathMapper()],
    )
    rows.append(jsonpath_summary)
    _, dotted_summary = run_workflow(
        dataset_id=dataset_id,
        technique_name="dotted_path_json_mapper_json",
        sources=sources,
        mappers=[DottedPathJsonMapper()],
    )
    rows.append(dotted_summary)
    return rows


def run_text_techniques(dataset_id: str, sources: list[SourceRecord]) -> list[dict[str, Any]]:
    has_text = any(source.content for source in sources)
    if not has_text:
        return []
    rows: list[dict[str, Any]] = []
    _, regex_summary = run_workflow(
        dataset_id=dataset_id,
        technique_name="regex_entity_extractor",
        sources=sources,
        extractors=[RegexEntityExtractor()],
    )
    rows.append(regex_summary)
    _, gazetteer_summary = run_workflow(
        dataset_id=dataset_id,
        technique_name="gazetteer_entity_extractor",
        sources=sources,
        extractors=[GazetteerEntityExtractor(DEFAULT_GAZETTEER)],
    )
    rows.append(gazetteer_summary)
    _, hybrid_summary = run_workflow(
        dataset_id=dataset_id,
        technique_name="hybrid_rules_gazetteer_regex",
        sources=sources,
        extractors=[RegexEntityExtractor(), GazetteerEntityExtractor(DEFAULT_GAZETTEER)],
    )
    rows.append(hybrid_summary)
    rows.append(run_offline_ollama_graph(dataset_id, sources))
    return rows


class StaticJsonProvider:
    """Small provider used to exercise Ollama adapter boundaries without live Ollama."""

    def __init__(self, payload: dict[str, Any], *, model: str = "offline-static-provider") -> None:
        self.payload = payload
        self.config = SimpleNamespace(
            provider="offline_static",
            model=model,
            evidence_required=True,
        )

    def complete_json(self, prompt: str, validator: Any | None = None) -> dict[str, Any]:
        _ = prompt
        if validator is not None:
            validator(self.payload)
        return self.payload


def first_evidence_text(sources: list[SourceRecord]) -> str:
    for source in sources:
        text = (source.content or "").strip()
        if text:
            return text.splitlines()[0][:200]
    return ""


def run_offline_ollama_graph(dataset_id: str, sources: list[SourceRecord]) -> dict[str, Any]:
    evidence_text = first_evidence_text(sources)
    if not evidence_text:
        return {
            "technique": "ollama_graph_extractor_offline_adapter",
            "status": "skipped",
            "reason": "No text evidence was available.",
        }
    payload = {
        "nodes": [
            {
                "id_hint": "source_document",
                "name": "Source document",
                "type": "Document",
                "properties": {"adapter_mode": "offline_static"},
                "evidence_text": evidence_text,
                "confidence": 0.8,
            }
        ],
        "relationships": [],
    }
    _, summary = run_workflow(
        dataset_id=dataset_id,
        technique_name="ollama_graph_extractor_offline_adapter",
        sources=sources,
        extractors=[
            OllamaGraphExtractor(
                StaticJsonProvider(payload),
                allowed_node_types=["Document", "Person", "Organization", "System", "Concept", "Event"],
            )
        ],
    )
    summary["notes"] = "Live Ollama was not called; this validates the structured-output adapter path."
    return summary


class WikiAnnGoldSpanModel:
    """GLiNER protocol stand-in backed by WikiANN gold spans."""

    def __init__(self, jsonl_path: Path, max_rows: int = 250) -> None:
        self.text, self.entities = self._load(jsonl_path, max_rows=max_rows)

    @staticmethod
    def _load(jsonl_path: Path, *, max_rows: int) -> tuple[str, list[dict[str, Any]]]:
        lines: list[str] = []
        entities: list[dict[str, Any]] = []
        offset = 0
        with jsonl_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= max_rows:
                    break
                payload = json.loads(line)
                text = str(payload.get("text", ""))
                lines.append(text)
                for span in payload.get("spans", []):
                    start = offset + int(span["start"])
                    end = offset + int(span["end"])
                    entities.append(
                        {
                            "text": text[int(span["start"]): int(span["end"])],
                            "label": str(span["label"]),
                            "start": start,
                            "end": end,
                            "score": 1.0,
                        }
                    )
                offset += len(text) + 1
        return "\n".join(lines), entities

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
    ) -> list[dict[str, Any]]:
        _ = (text, threshold, flat_ner)
        allowed = {label.casefold() for label in labels}
        return [entity for entity in self.entities if str(entity["label"]).casefold() in allowed]


class StaticBatchExtractor:
    """Return a precomputed extraction batch through the extractor interface."""

    name = "static_precomputed_batch_extractor"
    technique = "precomputed_batch_adapter"

    def __init__(self, batch: ConstructionBatch) -> None:
        self.batch = batch

    def extract(self, source: SourceRecord) -> ConstructionBatch:
        _ = source
        return self.batch


def run_gliner_label_pipeline() -> dict[str, Any]:
    jsonl_path = DATA_DIR / "wikiann" / "wikiann_en_sample.jsonl"
    if not jsonl_path.exists():
        return {
            "technique": "ollama_label_proposal_plus_gliner",
            "status": "skipped",
            "reason": "WikiANN JSONL sample was not available.",
        }
    model = WikiAnnGoldSpanModel(jsonl_path)
    source = SourceRecord.from_text(
        model.text,
        source_type="text",
        source_name=str(DATA_DIR / "wikiann" / "wikiann_en_sample.txt"),
    )
    label_payload = {
        "labels": [
            {"name": "Person", "confidence": 0.95, "description": "People in text"},
            {"name": "Organization", "confidence": 0.95, "description": "Organizations in text"},
            {"name": "Location", "confidence": 0.95, "description": "Locations in text"},
        ]
    }
    proposer = OllamaLabelProposalExtractor(
        StaticJsonProvider(label_payload, model="offline-label-proposal"),
        settings=LabelProposalSettings(allowed_label_count=5, auto_approve=True),
    )
    proposal = proposer.propose(source.content or "")
    started = time.perf_counter()
    try:
        batch = GlinerEntityExtractor(
            GlinerConfig(
                model_name="wikiann-gold-span-protocol-model",
                labels=proposal.reviewed_label_names,
                threshold=0.5,
            ),
            model=model,
        ).extract(source, labels=proposal.reviewed_label_names)
        result = ConstructionWorkflow(
            extractors=[StaticBatchExtractor(batch)],
            resolvers=[
                NormalizedStringResolver(),
                AliasTableResolver(ALIAS_TABLE),
            ],
            validators=[EvidenceRequiredValidator()],
            pipeline_name="wikiann_offline_label_proposal_plus_gliner",
        ).run([source])
        summary = summarize_result(
            "ollama_label_proposal_plus_gliner_offline_adapter",
            result,
            time.perf_counter() - started,
        )
        write_graph_artifacts("wikiann_en_ner", summary["technique"], result)
        summary["shacl"] = validate_with_generated_shacl("wikiann_en_ner", summary["technique"], result)
        summary.update(
            {
                "status": "ok",
                "proposed_labels": [label.name for label in proposal.proposed_labels],
                "reviewed_labels": proposal.reviewed_label_names,
                "notes": (
                    "Label proposal used the same OllamaLabelProposalExtractor review logic with "
                    "an offline static provider. GLiNER extraction used the implemented model protocol "
                    "with WikiANN gold spans to keep this dataset experiment deterministic; the "
                    "mandatory release hard pass executes the locked GLiNER package and weights."
                ),
            }
        )
        return summary
    except Exception as exc:
        return {
            "technique": "ollama_label_proposal_plus_gliner_offline_adapter",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_resolution_adapter_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    text_source = SourceRecord.from_text(
        "Recommendation API, recommendation api, Recs API, HDFS and hdfs.",
        source_name="memory://entity-resolution-adapter-check",
    )
    batch = GazetteerEntityExtractor(
        {
            "System": ["Recommendation API", "recommendation api", "Recs API", "HDFS", "hdfs"],
        }
    ).extract(text_source)
    for resolver in (
        NormalizedStringResolver(),
        AliasTableResolver(ALIAS_TABLE),
        RapidFuzzResolver(),
    ):
        try:
            resolved = resolver.resolve(batch)
            checks.append(
                {
                    "technique": resolver.technique,
                    "resolver": resolver.name,
                    "status": "ok",
                    "entity_count": len(resolved.entities),
                    "audit_clusters": len(resolved.metadata.get("entity_resolution_audit", [])),
                    "sample_audit": resolved.metadata.get("entity_resolution_audit", [])[:3],
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "technique": getattr(resolver, "technique", resolver.__class__.__name__),
                    "resolver": getattr(resolver, "name", resolver.__class__.__name__),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    for resolver in (SplinkProbabilisticResolver(), DedupeActiveLearningResolver()):
        try:
            resolver.resolve(batch)
            status = "ok"
            error = None
        except AdapterNotConfigured as exc:
            status = "requires_research_configuration"
            error = str(exc)
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        checks.append(
            {
                "technique": resolver.technique,
                "resolver": resolver.name,
                "status": status,
                "error": error,
            }
        )
    return checks


def run_benchmark(dataset: DatasetRun) -> dict[str, Any]:
    if dataset.benchmark_source_type is None or not dataset.path.exists():
        return {"status": "skipped", "reason": "No benchmark-compatible source type."}
    output_dir = BENCHMARK_DIR / dataset.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    config = TechniqueEvaluationConfig(
        evaluation=EvaluationSettings(
            deterministic_runtime=True,
            fixed_latency_seconds=0.0,
            fixed_throughput=1000.0,
            fixed_memory_peak_mb=0.0,
        )
    )
    try:
        report = TechniqueEvaluationRunner(config=config).run(
            input_path=dataset.path,
            source_type=dataset.benchmark_source_type,
            techniques="all",
            output_dir=None,
        )
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    summary = {
        "status": "ok",
        "best_technique": report.best_technique,
        "run_metadata": report.run_metadata,
        "ranked_scores": [
            {
                "rank": index,
                "technique": score.technique_name,
                "pipeline": score.pipeline_name,
                "final_score": round(score.final_score, 6),
                "quality_score": round(score.quality_score, 6),
                "runtime_score": round(score.runtime_score, 6),
                "confidence_interval": score.confidence_interval,
                "confidence_interval_method": score.confidence_interval_method,
            }
            for index, score in enumerate(report.ranked_scores, start=1)
        ],
        "candidate_status": [
            {
                "technique": candidate.technique_name,
                "pipeline": candidate.pipeline_name,
                "error": candidate.error,
                "entities": len(candidate.batch.entities) if candidate.batch else 0,
                "relations": len(candidate.batch.relations) if candidate.batch else 0,
            }
            for candidate in report.candidates
        ],
    }
    write_json(output_dir / "benchmark_summary.json", summary)
    return summary


def run_dataset(dataset: DatasetRun, *, sql_limit: int) -> dict[str, Any]:
    if not dataset.path.exists():
        return {
            "dataset_id": dataset.dataset_id,
            "display_name": dataset.display_name,
            "path": str(dataset.path),
            "status": "missing",
            "notes": "Run scripts/download_experiment_datasets.py first.",
        }
    started = time.perf_counter()
    try:
        sources = load_sources(dataset, sql_limit=sql_limit)
    except Exception as exc:
        return {
            "dataset_id": dataset.dataset_id,
            "display_name": dataset.display_name,
            "path": str(dataset.path),
            "status": "failed_to_load",
            "error": f"{type(exc).__name__}: {exc}",
        }
    profiler = DatasetProfiler()
    profile = profiler.profile_sources(sources, input_path=str(dataset.path))
    technique_summaries: list[dict[str, Any]] = []
    technique_summaries.extend(run_structured_techniques(dataset.dataset_id, sources))
    technique_summaries.extend(run_json_techniques(dataset.dataset_id, sources))
    technique_summaries.extend(run_text_techniques(dataset.dataset_id, sources))
    schema_summary = propose_schema(dataset.dataset_id, sources)
    benchmark_summary = run_benchmark(dataset)
    return {
        "dataset_id": dataset.dataset_id,
        "display_name": dataset.display_name,
        "path": str(dataset.path.relative_to(ROOT)),
        "loader": dataset.loader,
        "status": "ok",
        "notes": dataset.notes,
        "source_summary": source_summary(sources),
        "dataset_profile": profile,
        "schema_proposal": schema_summary,
        "techniques": technique_summaries,
        "benchmark": benchmark_summary,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def write_matrix_csv(results: list[dict[str, Any]]) -> None:
    path = REPORT_DIR / "dataset_technique_matrix.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_id",
        "loader",
        "technique",
        "status",
        "entity_count",
        "relation_count",
        "rdf_triples",
        "rdf_quads",
        "evidence_coverage",
        "unsupported_extraction_rate",
        "validation_findings",
        "shacl_conforms",
        "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for technique in result.get("techniques", []):
                shacl = technique.get("shacl", {})
                writer.writerow(
                    {
                        "dataset_id": result.get("dataset_id"),
                        "loader": result.get("loader"),
                        "technique": technique.get("technique"),
                        "status": technique.get("status"),
                        "entity_count": technique.get("entity_count", 0),
                        "relation_count": technique.get("relation_count", 0),
                        "rdf_triples": technique.get("rdf_triples", 0),
                        "rdf_quads": technique.get("rdf_quads", 0),
                        "evidence_coverage": technique.get("evidence_coverage", 0),
                        "unsupported_extraction_rate": technique.get("unsupported_extraction_rate", 0),
                        "validation_findings": technique.get("validation_findings", 0),
                        "shacl_conforms": shacl.get("conforms"),
                        "elapsed_seconds": technique.get("elapsed_seconds", 0),
                    }
                )


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def benchmark_score_rows(results: list[dict[str, Any]]) -> list[list[Any]]:
    """Return all ranked benchmark scores with explicit tie labels."""

    rows: list[list[Any]] = []
    for result in results:
        benchmark = result.get("benchmark", {})
        ranked_scores = benchmark.get("ranked_scores", [])
        if benchmark.get("status") != "ok" or not ranked_scores:
            continue
        scores_by_value: Counter[float] = Counter(
            round(float(score.get("final_score", 0.0)), 6)
            for score in ranked_scores
        )
        for score in ranked_scores:
            final_score = round(float(score.get("final_score", 0.0)), 6)
            rank = score.get("rank")
            rank_label = f"{rank} - tie" if scores_by_value[final_score] > 1 else str(rank)
            method = str(score.get("confidence_interval_method", "not_estimated"))
            interval = score.get("confidence_interval", [0.0, 0.0])
            uncertainty = "not estimated"
            if method != "not_estimated" and isinstance(interval, list | tuple) and len(interval) == 2:
                uncertainty = (
                    f"{method}: {float(interval[0]):.3f}-{float(interval[1]):.3f}"
                )
            rows.append(
                [
                    result.get("dataset_id"),
                    rank_label,
                    score.get("technique"),
                    f"{final_score:.6f}",
                    f"{float(score.get('quality_score', 0.0)):.6f}",
                    f"{float(score.get('runtime_score', 0.0)):.6f}",
                    uncertainty,
                ]
            )
    return rows


def winner_score_rows(results: list[dict[str, Any]]) -> list[list[Any]]:
    """Return one score interpretation row per benchmarked dataset."""

    rows: list[list[Any]] = []
    for result in results:
        benchmark = result.get("benchmark", {})
        ranked_scores = benchmark.get("ranked_scores", [])
        if benchmark.get("status") != "ok" or not ranked_scores:
            continue
        top_score = round(float(ranked_scores[0].get("final_score", 0.0)), 6)
        tied = [
            str(score.get("technique"))
            for score in ranked_scores
            if round(float(score.get("final_score", 0.0)), 6) == top_score
        ]
        rows.append(
            [
                result.get("dataset_id"),
                result.get("loader"),
                benchmark.get("best_technique"),
                f"{top_score:.6f}",
                ", ".join(tied) if len(tied) > 1 else "-",
                score_interpretation(str(result.get("dataset_id")), str(benchmark.get("best_technique"))),
            ]
        )
    return rows


def score_interpretation(dataset_id: str, best_technique: str) -> str:
    """Explain why a winner or tie makes sense for the dataset shape."""

    interpretations = {
        "nyc_311_service_requests": (
            "Tabular rows have stable identifiers and columns, so deterministic mapping is the safest path. "
            "JSONPath ties because CSV rows are already record-shaped after loading."
        ),
        "sec_apple_companyfacts": (
            "Nested financial JSON produces path evidence, while text-like scalar values also expose exact years, dates, and numeric patterns. "
            "Hybrid and regex tie under the current proxy weights."
        ),
        "pubmed_baseline": (
            "XML articles are flattened into structured records, so direct mapping and JSONPath-style record mapping tie after loading."
        ),
        "dblp_publication_api": (
            "Publication API payloads contain nested result objects plus many exact textual patterns. "
            "Hybrid and regex tie on score, while JSONPath produces the richest relation count in the technique execution table."
        ),
        "loghub_hdfs_raw": (
            "Raw logs are pattern-heavy, so regex and hybrid extraction score strongly. "
            "Hybrid is preferred when known system vocabulary should be captured alongside exact IDs and versions."
        ),
        "loghub_hdfs_structured": (
            "Structured log rows are already event records, so direct table mapping is the lowest-risk path."
        ),
        "wikiann_en_ner": (
            "The deterministic benchmark sees little domain vocabulary, so regex and hybrid tie. "
            "The separate GLiNER label workflow is the better NER-specific proof for this dataset."
        ),
        "enron_email_subset_text": (
            "Email text has both exact patterns and known business vocabulary, making hybrid extraction a sensible first path."
        ),
        "enron_email_subset_eml": (
            "The wrapped EML sample is short; all text extractors tie on proxy score, and gazetteer wins by tie policy and actual match count."
        ),
    }
    return interpretations.get(dataset_id, f"{best_technique} has the strongest weighted score for this dataset.")


def write_markdown_report(
    results: list[dict[str, Any]],
    gliner_summary: dict[str, Any],
    resolution_checks: list[dict[str, Any]],
) -> None:
    path = REPORT_DIR / "dataset_experiment_report.md"
    rows: list[list[Any]] = []
    for result in results:
        source = result.get("source_summary", {})
        techniques = [item for item in result.get("techniques", []) if item.get("status") == "ok"]
        best = result.get("benchmark", {}).get("best_technique", "-")
        rows.append(
            [
                result.get("dataset_id"),
                result.get("loader"),
                source.get("record_count", 0),
                source.get("text_character_count", 0),
                len(techniques),
                best or "-",
                result.get("status"),
            ]
        )

    technique_rows: list[list[Any]] = []
    for result in results:
        for item in result.get("techniques", []):
            shacl = item.get("shacl", {})
            technique_rows.append(
                [
                    result.get("dataset_id"),
                    item.get("technique"),
                    item.get("status"),
                    item.get("entity_count", 0),
                    item.get("relation_count", 0),
                    item.get("rdf_triples", 0),
                    item.get("evidence_coverage", 0),
                    shacl.get("conforms", "-"),
                ]
            )

    benchmark_rows: list[list[Any]] = []
    for result in results:
        ranked = result.get("benchmark", {}).get("ranked_scores", [])
        for score in ranked[:3]:
            benchmark_rows.append(
                [
                    result.get("dataset_id"),
                    score.get("rank"),
                    score.get("technique"),
                    score.get("final_score"),
                    score.get("quality_score"),
                    score.get("runtime_score"),
                ]
            )

    all_score_rows = benchmark_score_rows(results)
    winner_rows = winner_score_rows(results)

    resolution_rows = [
        [
            item.get("resolver"),
            item.get("technique"),
            item.get("status"),
            item.get("audit_clusters", "-"),
            item.get("error", ""),
        ]
        for item in resolution_checks
    ]

    content = [
        "# Data-to-Knowledge Graph Dataset Experiment Report",
        "",
        f"Generated from downloaded public samples in `{DATA_DIR.relative_to(ROOT)}`.",
        "",
        "## Dataset coverage",
        "",
        markdown_table(
            ["Dataset", "Loader", "Records", "Text chars", "Techniques ok", "Best benchmark", "Status"],
            rows,
        ),
        "",
        "## Dataset source references",
        "",
        markdown_table(
            ["Dataset", "Source", "Reference", "Sampling note"],
            DATASET_SOURCE_REFERENCES,
        ),
        "",
        "## Technique execution summary",
        "",
        markdown_table(
            [
                "Dataset",
                "Technique",
                "Status",
                "Entities",
                "Relations",
                "RDF triples",
                "Evidence coverage",
                "SHACL conforms",
            ],
            technique_rows,
        ),
        "",
        "## Deterministic benchmark top ranks",
        "",
        markdown_table(
            ["Dataset", "Rank", "Technique", "Final", "Quality", "Runtime"],
            benchmark_rows,
        )
        if benchmark_rows
        else "No deterministic benchmark rows were produced.",
        "",
        "## How to read the scores",
        "",
        "The benchmark scores follow the same evaluation story described in the article: each technique is loaded, profiled, executed through the construction workflow, validated, materialized into graph outputs, measured, and then ranked. The report uses deterministic runtime settings, so runtime columns are fixed proxies instead of noisy wall-clock measurements.",
        "",
        markdown_table(
            ["Score field", "Meaning"],
            [
                [
                    "`Final`",
                    "Weighted quality plus runtime score after penalties. Higher is better, bounded from 0 to 1.",
                ],
                [
                    "`Quality`",
                    "Accuracy/provenance proxy, validation, coverage, entity-resolution, retrieval-readiness, and maintainability signals minus trust penalties.",
                ],
                [
                    "`Runtime`",
                    "Scalability and cost-efficiency proxy minus latency penalty. Fixed in this deterministic run to avoid score drift.",
                ],
                [
                    "`Score uncertainty`",
                    "Statistical uncertainty for the weighted score. It is reported as not estimated unless a valid resampling or repeated-run method is supplied.",
                ],
                [
                    "`Tie`",
                    "Same rounded final score. Tie policy then uses production thresholds, latency, maintainability, and technique name.",
                ],
            ],
        ),
        "",
        "## Dataset score interpretation",
        "",
        markdown_table(
            ["Dataset", "Loader", "Recommended path", "Top score", "Tied paths", "Interpretation"],
            winner_rows,
        ),
        "",
        "## Full benchmark score table",
        "",
        markdown_table(
            [
                "Dataset",
                "Rank",
                "Technique",
                "Final",
                "Quality",
                "Runtime",
                "Score uncertainty",
            ],
            all_score_rows,
        )
        if all_score_rows
        else "No benchmark score rows were produced.",
        "",
        "## GLiNER plus LLM-label workflow",
        "",
        "This deterministic dataset report exercises the GLiNER model protocol with WikiANN gold "
        "spans and the Ollama label-proposal review logic with a static provider. The mandatory "
        "release hard pass separately executes the locked GLiNER and Ollama runtimes.",
        "",
        "```json",
        json.dumps(model_dump(gliner_summary), indent=2, sort_keys=True),
        "```",
        "",
        "## Entity resolution adapter checks",
        "",
        markdown_table(
            ["Resolver", "Technique", "Status", "Audit clusters", "Error or note"],
            resolution_rows,
        ),
        "",
        "## Output artifacts",
        "",
        f"- Graph artifacts: `{GRAPH_DIR.relative_to(ROOT)}`",
        f"- Schema and SHACL artifacts: `{SCHEMA_DIR.relative_to(ROOT)}`",
        f"- Benchmark summaries: `{BENCHMARK_DIR.relative_to(ROOT)}`",
        f"- Machine-readable report: `{(REPORT_DIR / 'dataset_experiment_results.json').relative_to(ROOT)}`",
        f"- Technique matrix CSV: `{(REPORT_DIR / 'dataset_technique_matrix.csv').relative_to(ROOT)}`",
        "",
        "## Read this result correctly",
        "",
        "These results prove that the implemented technique boundaries can run across multiple data shapes: "
        "relational tables, CSV, nested JSON/API payloads, XML, logs, NER text, and email-like text. "
        "They are not a claim that every technique is best for every dataset. The benchmark runner uses "
        "deterministic runtime settings here so score movement comes from graph-construction quality signals, "
        "not wall-clock noise.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql-limit", type=int, default=50, help="Rows per SQL table.")
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset IDs to run, or 'all'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    selected = (
        {item.strip() for item in args.datasets.split(",") if item.strip()}
        if args.datasets != "all"
        else {dataset.dataset_id for dataset in DATASET_RUNS}
    )
    results = [
        run_dataset(dataset, sql_limit=args.sql_limit)
        for dataset in DATASET_RUNS
        if dataset.dataset_id in selected
    ]
    gliner_summary = run_gliner_label_pipeline()
    resolution_checks = run_resolution_adapter_checks()
    full_report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_dir": str(DATA_DIR),
        "results": results,
        "gliner_label_pipeline": gliner_summary,
        "entity_resolution_checks": resolution_checks,
    }
    write_json(REPORT_DIR / "dataset_experiment_results.json", full_report)
    write_matrix_csv(results)
    write_markdown_report(results, gliner_summary, resolution_checks)
    print(f"Wrote experiment report: {REPORT_DIR / 'dataset_experiment_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
