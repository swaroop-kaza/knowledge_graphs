# Schema & Instance Quality Report

[Reports index](README.md) · [Evaluation guide](../docs/evaluation_metrics.md) · [Testing guide](../docs/testing_guide.md)

| Metadata | Value |
| --- | --- |
| Generated | 2026-07-26 |
| Thresholds | `schema-quality-v2` |
| Overall | **PASS** |

Computed on the policy-driven semantic graph of `examples/data_to_kg/sample.csv` (the semantic mapper). Categorical duplication is policy-aware.

## Release gates

| Gate | Metric | Operator | Threshold | Actual | Passed |
| --- | --- | --- | ---: | ---: | :---: |
| node_conciseness | duplicate_node_rate | <= | 0 | 0 | PASS |
| connectivity | orphan_rate | <= | 0.05 | 0 | PASS |
| consistency | consistency_violation_count | <= | 0 | 0 | PASS |
| categorical_conciseness | categorical_duplication_rate | <= | 0 | 0 | PASS |
| categorical_occurrence_reconciliation | categorical_source_occurrence_coverage | >= | 1 | 1 | PASS |
| property_completeness | property_completeness | >= | 1 | 1 | PASS |
| relationship_completeness | relationship_completeness | >= | 1 | 1 | PASS |
| shacl_execution | shacl_executed | >= | 1 | 1 | PASS |
| shacl_conformance | shacl_violation_count | <= | 0 | 0 | PASS |

## Conciseness (freedom from redundancy)

| Metric | Value |
| --- | ---: |
| `categorical_duplication_rate` | 0.0 |
| `literal_that_should_be_node_rate` | 0.0 |
| `duplicate_node_rate` | 0.0 |

## Completeness

| Metric | Value |
| --- | ---: |
| `schema_completeness` | 0.8571428571428571 |
| `property_completeness` | 1.000 |
| `relationship_completeness` | 1.000 |
| `class_population` | {'Employee': 3, 'Team': 3} |

## Consistency

| Metric | Value |
| --- | ---: |
| `domain_range_violation_rate` | 0.0 |
| `cardinality_violation_rate` | 0.0 |
| `shacl_executed` | True |
| `shacl_violation_count` | 0 |
| `consistency_violation_count` | 0 |

## Structure (OntoQA)

| Metric | Value |
| --- | ---: |
| `node_count` | 6 |
| `edge_count` | 5 |
| `average_degree` | 1.667 |
| `orphan_rate` | 0.0 |
| `connected_component_count` | 1 |
| `class_richness` | 1.0 |
| `relationship_richness` | 0.250 |
| `attribute_richness` | 2.500 |

## Reproduce

```bash
make schema-quality
```
