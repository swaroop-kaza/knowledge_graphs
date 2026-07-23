# Gold-Standard Benchmark Story

Generated: 2026-07-22

This report turns the tiny human-reviewed gold fixtures into a public correctness story. It is not a broad model leaderboard. It proves that the framework can run benchmark-grade checks for extraction, graph facts, identity resolution, and GraphRAG-style retrieval using deterministic local examples.

Regenerate with:

```bash
make gold-benchmark-story
```

## Summary

| Benchmark | Technique | Fixture | Precision | Recall | F1 | Support |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| NER span benchmark | `hybrid_regex_gazetteer_span_extraction` | `tests/fixtures/gold_standards/ner_spans.json` | 1.000 | 1.000 | 1.000 | 3 |
| Relation extraction benchmark | `dependency_pattern_relation_extractor` | `tests/fixtures/gold_standards/relation_sample.json` | 1.000 | 1.000 | 1.000 | 2 |
| Triple benchmark | `gold_triple_exact_match` | `tests/fixtures/gold_standards/gold_triples.json` | 1.000 | 1.000 | 1.000 | 2 |
| Entity resolution benchmark | `alias_table_resolver` | `tests/fixtures/gold_standards/entity_resolution_clusters.json` | 1.000 | 1.000 | 1.000 | 5 |
| GraphRAG/retrieval benchmark | `lexical_graph_context_retriever_at_2` | `tests/fixtures/gold_standards/retrieval_cases.json` | 0.667 | 0.750 | 0.706 | 4 |

## What Each Benchmark Proves

### NER span benchmark

- Fixture: `tests/fixtures/gold_standards/ner_spans.json`
- Technique under test: `hybrid_regex_gazetteer_span_extraction`
- Gold unit: exact entity spans with label, start, and end
- Predicted unit: regex Incident/Date spans plus gazetteer System spans
- Result: precision `1.000`, recall `1.000`, F1 `1.000`
- Detail: Scores exact text, label, start, and end offsets. TP=3, FP=0, FN=0.

### Relation extraction benchmark

- Fixture: `tests/fixtures/gold_standards/relation_sample.json`
- Technique under test: `dependency_pattern_relation_extractor`
- Gold unit: human-reviewed source, relation type, and target mentions
- Predicted unit: dependency-style relation pattern candidates with evidence
- Result: precision `1.000`, recall `1.000`, F1 `1.000`
- Detail: Scores typed relation mentions by source name, relation type, and target name. TP=2, FP=0, FN=0.

### Triple benchmark

- Fixture: `tests/fixtures/gold_standards/gold_triples.json`
- Technique under test: `gold_triple_exact_match`
- Gold unit: subject-predicate-object triples
- Predicted unit: materialized GraphFact triples
- Result: precision `1.000`, recall `1.000`, F1 `1.000`
- Detail: Scores RDF-style triples by exact subject, predicate, and object. TP=2, FP=0, FN=0.

### Entity resolution benchmark

- Fixture: `tests/fixtures/gold_standards/entity_resolution_clusters.json`
- Technique under test: `alias_table_resolver`
- Gold unit: human-reviewed duplicate clusters converted to gold merge pairs
- Predicted unit: canonical clusters with merge audit metadata
- Result: precision `1.000`, recall `1.000`, F1 `1.000`
- Detail: Scores pairwise ER precision/recall/F1. Merge audit rows: 3.

### GraphRAG/retrieval benchmark

- Fixture: `tests/fixtures/gold_standards/retrieval_cases.json`
- Technique under test: `lexical_graph_context_retriever_at_2`
- Gold unit: queries with relevant contexts, paths, citations, and answer hints
- Predicted unit: top-k graph contexts ranked by lexical query/context overlap
- Result: precision `0.667`, recall `0.750`, F1 `0.706`
- Detail: recall@2=0.750; path_relevance=0.667; citation_accuracy=0.667; faithfulness=1.000; multi_hop=0.000

## Research Alignment

- NER benchmarks should evaluate exact spans and labels, not only entity names.
- Relation extraction benchmarks should evaluate the typed source-relation-target claim.
- Triple benchmarks should compare exact subject-predicate-object facts.
- Entity resolution benchmarks should expose pairwise precision, recall, F1, false merges, and false splits.
- GraphRAG/retrieval benchmarks should test context recall, useful graph paths, citations, and answer faithfulness.

## How To Use This

- Use this report as the small reproducible gold story for the repository.
- Use `reports/benchmark_leaderboard.md` for technique-ranking across text, CSV, JSON, and mixed fixtures.
- Add larger public datasets later without changing the benchmark contract.
- Keep these fixtures tiny so CI can verify them on every public release.

## References

- CoNLL-2003 shared task: https://www.clips.uantwerpen.be/conll2003/ner/
- SemEval-2010 Task 8 relation classification: https://aclanthology.org/S10-1006/
- TACRED relation extraction dataset: https://nlp.stanford.edu/projects/tacred/
- Splink entity resolution toolkit: https://moj-analytical-services.github.io/splink/
- Microsoft GraphRAG: https://microsoft.github.io/graphrag/
- BEIR retrieval benchmark: https://github.com/beir-cellar/beir
- W3C RDF 1.1 Concepts: https://www.w3.org/TR/rdf11-concepts/
