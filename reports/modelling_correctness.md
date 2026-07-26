# Modelling Correctness Report

[Reports index](README.md) · [Evaluation guide](../docs/evaluation_metrics.md) · [Testing guide](../docs/testing_guide.md)

| Metadata | Value |
| --- | --- |
| Generated | 2026-07-26 |
| Thresholds | `kg-quality-v3` |
| Overall | **PASS** |

Runs the gold modelling cases through the real pipeline and verifies the output conforms to the expected graph (RML/R2RML conformance style). CI fails on regression.

## structured_categorical_and_fk_v1 (structured) - PASS

| Check | Result |
| --- | :---: |
| record_node_count | PASS |
| categorical_duplication_zero | PASS |
| shared_nodes:Department | PASS |
| edges:has_department | PASS |
| edges:manager | PASS |
| property_kept:status | PASS |
| absent:Status | PASS |
| no_literal:department | PASS |
| no_literal:manager_id | PASS |

| Metric | Value |
| --- | ---: |
| `categorical_duplication_rate` | 0 |
| `edges:has_department` | 8 |
| `edges:manager` | 7 |
| `nodes:Department` | 3 |
| `orphan_rate` | 0 |
| `record_nodes` | 8 |

## unstructured_salient_entities_v1 (unstructured) - PASS

| Check | Result |
| --- | :---: |
| entity_coverage_meets_minimum | PASS |
| claims_have_complete_provenance | PASS |
| intents_evaluated | PASS |

| Metric | Value |
| --- | ---: |
| `claim_coverage` | 1 |
| `entity_coverage` | 0.8 |
| `gold_entity_count` | 5 |
| `intent_coverage` | 0 |
| `matched_entity_count` | 4 |
| `minimum_entity_coverage` | 0.8 |
| `nil_rate` | 1 |
| `promoted_claim_count` | 0 |

## Reproduce

```bash
make modelling-correctness
```
