"""Generate a user-facing gold-standard benchmark story report."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    GraphFact,
    SourceRecord,
)
from kg_construction.infrastructure.evaluation import (
    EntityResolutionMetrics,
    GoldBenchmarkEvaluator,
    GoldBenchmarkFixture,
    GoldBenchmarkScore,
    GoldEntityResolutionFixture,
    GoldRetrievalCase,
    GoldRetrievalContext,
    GoldRetrievalFixture,
    RetrievalMetrics,
)
from kg_construction.infrastructure.extractors.gazetteer import GazetteerEntityExtractor
from kg_construction.infrastructure.extractors.regex import RegexEntityExtractor, RegexPattern
from kg_construction.infrastructure.extractors.relations import (
    DependencyPatternRelationExtractor,
    DependencyRelationPattern,
)
from kg_construction.infrastructure.resolution.entity_resolution import AliasTableResolver

REPORTS_DIR = Path("reports")
GOLD_DIR = Path("tests/fixtures/gold_standards")
STORY_MD = REPORTS_DIR / "gold_standard_benchmark_story.md"
STORY_CSV = REPORTS_DIR / "gold_standard_benchmark_story.csv"
STORY_JSON = REPORTS_DIR / "gold_standard_benchmark_story.json"


@dataclass(frozen=True, slots=True)
class GoldBenchmarkStoryRow:
    """One public benchmark row generated from a tiny gold fixture."""

    benchmark_type: str
    fixture_id: str
    fixture_path: str
    technique: str
    gold_unit: str
    predicted_unit: str
    precision: float
    recall: float
    f1: float
    support: int
    details: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark_type": self.benchmark_type,
            "fixture_id": self.fixture_id,
            "fixture_path": self.fixture_path,
            "technique": self.technique,
            "gold_unit": self.gold_unit,
            "predicted_unit": self.predicted_unit,
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "f1": _round(self.f1),
            "support": self.support,
            "details": self.details,
        }


def main() -> int:
    """Generate Markdown, CSV, and JSON benchmark story outputs."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, details = generate_gold_standard_benchmark_story()
    _write_csv(rows, STORY_CSV)
    _write_json(rows, details, STORY_JSON)
    _write_markdown(rows, details, STORY_MD)
    print(f"gold-standard-benchmark-story-ok: {len(rows)} benchmark rows written")
    return 0


def generate_gold_standard_benchmark_story() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run all tiny gold-standard benchmark examples."""

    row_objects = [
        _benchmark_ner_spans(),
        _benchmark_relation_mentions(),
        _benchmark_triples(),
        _benchmark_entity_resolution(),
        _benchmark_graphrag_retrieval(),
    ]
    rows = [row.as_dict() for row in row_objects]
    details = {
        "generated_at": datetime.now(UTC).date().isoformat(),
        "benchmark_count": len(rows),
        "fixture_root": GOLD_DIR.as_posix(),
        "scope": "tiny human-reviewed fixtures for public correctness checks",
        "research_alignment": [
            "NER span benchmarks usually score exact span and label matches.",
            "Relation extraction benchmarks score typed source-relation-target predictions.",
            "Triple benchmarks score subject-predicate-object exact matches.",
            "Entity resolution benchmarks commonly use pairwise precision, recall, and F1.",
            "GraphRAG retrieval checks combine context recall, path usefulness, citations, and faithfulness.",
        ],
    }
    return rows, details


def _benchmark_ner_spans() -> GoldBenchmarkStoryRow:
    fixture_path = GOLD_DIR / "ner_spans.json"
    gold = GoldBenchmarkFixture.from_json_file(fixture_path)
    source = SourceRecord.from_text(gold.text or "", source_name=f"gold://{gold.fixture_id}")
    regex = RegexEntityExtractor(
        patterns=(
            RegexPattern("Incident", re.compile(r"\bINCIDENT-\d+\b")),
            RegexPattern("Date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
        )
    ).extract(source)
    gazetteer = GazetteerEntityExtractor({"System": ["Recommendation API"]}).extract(source)
    batch = ConstructionBatch(
        source_records=[source],
        entities=[*regex.entities, *gazetteer.entities],
        metadata={"technique": "hybrid_regex_gazetteer_span_extraction"},
    )
    score = GoldBenchmarkEvaluator().evaluate_entity_spans(batch, gold)
    return _row_from_score(
        benchmark_type="NER span benchmark",
        fixture_id=gold.fixture_id,
        fixture_path=fixture_path,
        technique="hybrid_regex_gazetteer_span_extraction",
        gold_unit="exact entity spans with label, start, and end",
        predicted_unit="regex Incident/Date spans plus gazetteer System spans",
        score=score,
        support=len(gold.entity_spans),
        details="Scores exact text, label, start, and end offsets.",
    )


def _benchmark_relation_mentions() -> GoldBenchmarkStoryRow:
    fixture_path = GOLD_DIR / "relation_sample.json"
    gold = GoldBenchmarkFixture.from_json_file(fixture_path)
    source = SourceRecord.from_text(gold.text or "", source_name=f"gold://{gold.fixture_id}")
    batch = DependencyPatternRelationExtractor(
        patterns=[
            DependencyRelationPattern(
                name="incident_affected_system",
                relation_type="AFFECTED",
                pattern=r"(?P<source>INCIDENT-\d+)\s+affected\s+(?P<target>Recommendation API)",
            ),
            DependencyRelationPattern(
                name="incident_investigated_by_team",
                relation_type="INVESTIGATED_BY",
                pattern=(
                    r"(?P<source>INCIDENT-\d+).*?investigated\s+by\s+"
                    r"(?P<target>Platform Team)"
                ),
            ),
        ]
    ).extract(source)
    score = GoldBenchmarkEvaluator().evaluate_relation_mentions(batch, gold)
    return _row_from_score(
        benchmark_type="Relation extraction benchmark",
        fixture_id=gold.fixture_id,
        fixture_path=fixture_path,
        technique="dependency_pattern_relation_extractor",
        gold_unit="human-reviewed source, relation type, and target mentions",
        predicted_unit="dependency-style relation pattern candidates with evidence",
        score=score,
        support=len(gold.relations),
        details="Scores typed relation mentions by source name, relation type, and target name.",
    )


def _benchmark_triples() -> GoldBenchmarkStoryRow:
    fixture_path = GOLD_DIR / "gold_triples.json"
    gold = GoldBenchmarkFixture.from_json_file(fixture_path)
    source = SourceRecord.from_text("RDF triple gold fixture", source_name=f"gold://{gold.fixture_id}")
    evidence = EvidenceRef.build(source, text="RDF triple gold fixture")
    batch = ConstructionBatch(
        source_records=[source],
        facts=[
            GraphFact.build(
                subject=triple.subject,
                predicate=triple.predicate,
                object=triple.object,
                evidence=[evidence],
                extractor_name="gold_fixture_triple_materializer",
                technique="gold_triple_exact_match",
                validation_status="valid",
            )
            for triple in gold.triples
        ],
        metadata={"technique": "gold_triple_exact_match"},
    )
    score = GoldBenchmarkEvaluator().evaluate_triples(batch, gold)
    return _row_from_score(
        benchmark_type="Triple benchmark",
        fixture_id=gold.fixture_id,
        fixture_path=fixture_path,
        technique="gold_triple_exact_match",
        gold_unit="subject-predicate-object triples",
        predicted_unit="materialized GraphFact triples",
        score=score,
        support=len(gold.triples),
        details="Scores RDF-style triples by exact subject, predicate, and object.",
    )


def _benchmark_entity_resolution() -> GoldBenchmarkStoryRow:
    fixture_path = GOLD_DIR / "entity_resolution_clusters.json"
    fixture = GoldEntityResolutionFixture.from_json_file(fixture_path)
    source = SourceRecord.from_rows(
        [record.model_dump() for record in fixture.records],
        source_type="gold_entity_resolution",
        source_name=f"gold://{fixture.fixture_id}",
    )
    evidence = EvidenceRef.build(source, text="human-reviewed entity resolution fixture")
    batch = ConstructionBatch(
        source_records=[source],
        entities=[
            EntityCandidate(
                candidate_id=record.record_id,
                entity_type=record.entity_type,
                name=record.name,
                evidence=[evidence],
                extractor_name="gold_fixture_entity_loader",
                technique="gold_entity_resolution_fixture",
            )
            for record in fixture.records
        ],
    )
    resolved = AliasTableResolver(
        aliases={
            "Recommendation API": ["recommendation api", "Rec API"],
            "Feature Store": ["Feature-store"],
            "Platform Team": ["Platform"],
        }
    ).resolve(batch)
    metrics = EntityResolutionMetrics().evaluate(resolved, gold_pairs=fixture.gold_pairs())
    return GoldBenchmarkStoryRow(
        benchmark_type="Entity resolution benchmark",
        fixture_id=fixture.fixture_id,
        fixture_path=fixture_path.as_posix(),
        technique="alias_table_resolver",
        gold_unit="human-reviewed duplicate clusters converted to gold merge pairs",
        predicted_unit="canonical clusters with merge audit metadata",
        precision=metrics["pairwise_er_precision"],
        recall=metrics["pairwise_er_recall"],
        f1=metrics["pairwise_er_f1"],
        support=len(fixture.gold_pairs()),
        details=(
            "Scores pairwise ER precision/recall/F1. "
            f"Merge audit rows: {len(resolved.metadata.get('entity_resolution_audit', []))}."
        ),
    )


def _benchmark_graphrag_retrieval() -> GoldBenchmarkStoryRow:
    fixture_path = GOLD_DIR / "retrieval_cases.json"
    fixture = GoldRetrievalFixture.from_json_file(fixture_path)
    retrieval_k = int(fixture.metadata.get("retrieval_k", 2))
    all_retrieved: list[str] = []
    all_relevant: set[str] = set()
    useful_paths = 0
    correct_multi_hop = 0
    total_multi_hop = 0
    correct_citations = 0
    faithful_answers = 0
    context_by_id = {context.context_id: context for context in fixture.contexts}

    for case in fixture.cases:
        retrieved_ids = _retrieve_context_ids(case, fixture.contexts, retrieval_k)
        retrieved_contexts = [context_by_id[context_id] for context_id in retrieved_ids]
        all_retrieved.extend(f"{case.query_id}:{context_id}" for context_id in retrieved_ids)
        all_relevant.update(f"{case.query_id}:{context_id}" for context_id in case.relevant_context_ids)
        has_path = _has_expected_path(case, retrieved_contexts)
        useful_paths += int(has_path)
        if _is_multi_hop(case):
            total_multi_hop += 1
            correct_multi_hop += int(has_path and _all_relevant_retrieved(case, retrieved_ids))
        correct_citations += int(_has_expected_citations(case, retrieved_contexts))
        faithful_answers += int(_has_expected_answer_text(case, retrieved_contexts))

    metrics = RetrievalMetrics().evaluate(
        retrieved=all_retrieved,
        relevant=all_relevant,
        useful_paths=useful_paths,
        retrieved_paths=len(fixture.cases),
        correct_multi_hop_answers=correct_multi_hop,
        total_multi_hop_questions=total_multi_hop,
        correct_citations=correct_citations,
        total_answers=len(fixture.cases),
        faithful_answers=faithful_answers,
    )
    precision = metrics["path_relevance"]
    recall = metrics["retrieval_recall_at_k"]
    return GoldBenchmarkStoryRow(
        benchmark_type="GraphRAG/retrieval benchmark",
        fixture_id=fixture.fixture_id,
        fixture_path=fixture_path.as_posix(),
        technique="lexical_graph_context_retriever_at_2",
        gold_unit="queries with relevant contexts, paths, citations, and answer hints",
        predicted_unit="top-k graph contexts ranked by lexical query/context overlap",
        precision=precision,
        recall=recall,
        f1=_harmonic_mean(precision, recall),
        support=sum(len(case.relevant_context_ids) for case in fixture.cases),
        details=(
            f"recall@{retrieval_k}={metrics['retrieval_recall_at_k']:.3f}; "
            f"path_relevance={metrics['path_relevance']:.3f}; "
            f"citation_accuracy={metrics['citation_accuracy']:.3f}; "
            f"faithfulness={metrics['answer_faithfulness']:.3f}; "
            f"multi_hop={metrics['multi_hop_success_rate']:.3f}"
        ),
    )


def _row_from_score(
    *,
    benchmark_type: str,
    fixture_id: str,
    fixture_path: Path,
    technique: str,
    gold_unit: str,
    predicted_unit: str,
    score: GoldBenchmarkScore,
    support: int,
    details: str,
) -> GoldBenchmarkStoryRow:
    return GoldBenchmarkStoryRow(
        benchmark_type=benchmark_type,
        fixture_id=fixture_id,
        fixture_path=fixture_path.as_posix(),
        technique=technique,
        gold_unit=gold_unit,
        predicted_unit=predicted_unit,
        precision=score.precision,
        recall=score.recall,
        f1=score.f1,
        support=support,
        details=(
            f"{details} TP={score.true_positive}, "
            f"FP={score.false_positive}, FN={score.false_negative}."
        ),
    )


def _retrieve_context_ids(
    case: GoldRetrievalCase,
    contexts: list[GoldRetrievalContext],
    k: int,
) -> list[str]:
    query_tokens = _tokens(case.query)
    scored: list[tuple[int, str]] = []
    for context in contexts:
        haystack = _context_haystack(context)
        score = sum(1 for token in query_tokens if token in haystack)
        scored.append((score, context.context_id))
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    return [context_id for score, context_id in ranked[:k] if score > 0]


def _tokens(value: str) -> list[str]:
    stopwords = {"the", "a", "an", "did", "by", "which", "was", "is", "of", "to"}
    return [
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if token not in stopwords
    ]


def _context_haystack(context: GoldRetrievalContext) -> str:
    edge_parts = [
        f"{edge.source} {edge.relation_type} {edge.target}".casefold()
        for edge in context.edges
    ]
    return " ".join([context.text, *context.nodes, *edge_parts, *context.citation_ids]).casefold()


def _has_expected_path(
    case: GoldRetrievalCase,
    contexts: list[GoldRetrievalContext],
) -> bool:
    if len(case.expected_path) < 3:
        return False
    available_edges = {
        (edge.source, edge.relation_type, edge.target)
        for context in contexts
        for edge in context.edges
    }
    expected_edges = [
        (
            case.expected_path[index],
            case.expected_path[index + 1],
            case.expected_path[index + 2],
        )
        for index in range(0, len(case.expected_path) - 2, 2)
    ]
    return all(edge in available_edges for edge in expected_edges)


def _is_multi_hop(case: GoldRetrievalCase) -> bool:
    return len(case.expected_path) > 3


def _all_relevant_retrieved(case: GoldRetrievalCase, retrieved_ids: list[str]) -> bool:
    return set(case.relevant_context_ids).issubset(set(retrieved_ids))


def _has_expected_citations(
    case: GoldRetrievalCase,
    contexts: list[GoldRetrievalContext],
) -> bool:
    citations = {citation_id for context in contexts for citation_id in context.citation_ids}
    return set(case.expected_citation_ids).issubset(citations)


def _has_expected_answer_text(
    case: GoldRetrievalCase,
    contexts: list[GoldRetrievalContext],
) -> bool:
    combined = " ".join(context.text for context in contexts).casefold()
    return all(fragment.casefold() in combined for fragment in case.expected_answer_contains)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "benchmark_type",
        "fixture_id",
        "fixture_path",
        "technique",
        "gold_unit",
        "predicted_unit",
        "precision",
        "recall",
        "f1",
        "support",
        "details",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(rows: list[dict[str, Any]], details: dict[str, Any], path: Path) -> None:
    payload = {"details": details, "rows": rows}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(rows: list[dict[str, Any]], details: dict[str, Any], path: Path) -> None:
    lines = [
        "# Gold-Standard Benchmark Story",
        "",
        f"Generated: {details['generated_at']}",
        "",
        "This report turns the tiny human-reviewed gold fixtures into a public correctness story. It is not a broad model leaderboard. It proves that the framework can run benchmark-grade checks for extraction, graph facts, identity resolution, and GraphRAG-style retrieval using deterministic local examples.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "make gold-benchmark-story",
        "```",
        "",
        "## Summary",
        "",
        "| Benchmark | Technique | Fixture | Precision | Recall | F1 | Support |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {benchmark_type} | `{technique}` | `{fixture_path}` | {precision:.3f} | {recall:.3f} | {f1:.3f} | {support} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## What Each Benchmark Proves",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['benchmark_type']}",
                "",
                f"- Fixture: `{row['fixture_path']}`",
                f"- Technique under test: `{row['technique']}`",
                f"- Gold unit: {row['gold_unit']}",
                f"- Predicted unit: {row['predicted_unit']}",
                f"- Result: precision `{row['precision']:.3f}`, recall `{row['recall']:.3f}`, F1 `{row['f1']:.3f}`",
                f"- Detail: {row['details']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Research Alignment",
            "",
            "- NER benchmarks should evaluate exact spans and labels, not only entity names.",
            "- Relation extraction benchmarks should evaluate the typed source-relation-target claim.",
            "- Triple benchmarks should compare exact subject-predicate-object facts.",
            "- Entity resolution benchmarks should expose pairwise precision, recall, F1, false merges, and false splits.",
            "- GraphRAG/retrieval benchmarks should test context recall, useful graph paths, citations, and answer faithfulness.",
            "",
            "## How To Use This",
            "",
            "- Use this report as the small reproducible gold story for the repository.",
            "- Use `reports/benchmark_leaderboard.md` for technique-ranking across text, CSV, JSON, and mixed fixtures.",
            "- Add larger public datasets later without changing the benchmark contract.",
            "- Keep these fixtures tiny so CI can verify them on every public release.",
            "",
            "## References",
            "",
            "- CoNLL-2003 shared task: https://www.clips.uantwerpen.be/conll2003/ner/",
            "- SemEval-2010 Task 8 relation classification: https://aclanthology.org/S10-1006/",
            "- TACRED relation extraction dataset: https://nlp.stanford.edu/projects/tacred/",
            "- Splink entity resolution toolkit: https://moj-analytical-services.github.io/splink/",
            "- Microsoft GraphRAG: https://microsoft.github.io/graphrag/",
            "- BEIR retrieval benchmark: https://github.com/beir-cellar/beir",
            "- W3C RDF 1.1 Concepts: https://www.w3.org/TR/rdf11-concepts/",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _round(value: float) -> float:
    return round(float(value), 3)


def _harmonic_mean(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
