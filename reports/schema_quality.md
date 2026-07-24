# Schema & Instance Quality Report

Generated: 2026-07-24
Thresholds: `schema-quality-v2`

Computed on the policy-driven semantic graph of `examples/data_to_kg/sample.csv` (the semantic mapper). Categorical duplication is policy-aware.

## Release gates

| Gate | Metric | Operator | Threshold | Actual | Passed |
| --- | --- | --- | ---: | ---: | :---: |
| node_conciseness | duplicate_node_rate | <= | 0 | 0 | PASS |
| connectivity | orphan_rate | <= | 0.05 | 0 | PASS |
| consistency | consistency_violation_count | <= | 0 | 0 | PASS |
| categorical_conciseness | categorical_duplication_rate | <= | 0 | 0 | PASS |
| property_completeness | property_completeness | >= | 1 | 1 | PASS |
| relationship_completeness | relationship_completeness | >= | 1 | 1 | PASS |
| shacl_execution | shacl_executed | >= | 1 | 1 | PASS |
| shacl_conformance | shacl_violation_count | <= | 0 | 0 | PASS |

**Overall: PASS**

## Conciseness (freedom from redundancy)
- categorical_duplication_rate: 0.0
- literal_that_should_be_node_rate: 0.0
- duplicate_node_rate: 0.0

## Completeness
- schema_completeness: 0.8571428571428571
- property_completeness: 1.000
- relationship_completeness: 1.000
- class_population: {'Employee': 3, 'Team': 3}

## Consistency
- domain_range_violation_rate: 0.0
- cardinality_violation_rate: 0.0
- shacl_executed: True
- shacl_violation_count: 0
- consistency_violation_count: 0

## Structure (OntoQA)
- node_count: 6 | edge_count: 5
- average_degree: 1.667 | orphan_rate: 0.0
- connected_component_count: 1
- class_richness: 1.0 | relationship_richness: 0.217
- attribute_richness: 3.000

Regenerate with `make schema-quality`.
