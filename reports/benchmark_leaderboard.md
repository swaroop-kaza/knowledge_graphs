# Small Benchmark Leaderboard

Generated: 2026-07-14

This committed leaderboard is intentionally small and reproducible. It uses tiny fixtures under `tests/fixtures/golden_benchmarks/`, deterministic runtime settings, and the same scoring path used by `TechniqueEvaluationRunner`. Big multi-dataset experiment outputs remain generated artifacts outside the committed repo.

Runtime/resource columns are deterministic controls for comparability:

- latency: `0.01` seconds
- memory/resource usage: `0.25` MB
- throughput: `1000.0` items/second

Regenerate with:

```bash
PYTHONPATH=src python scripts/generate_benchmark_leaderboard.py
```

| Dataset type | Rank | Technique | Entity coverage | Relation coverage | Evidence coverage | Validation pass rate | Latency | Memory/resource | Final score |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 1 | `gazetteer_entity_extractor` | 1.000 | 0.000 | 1.000 | 1.000 | 0.010s | 0.250 MB | 0.777 |
| text | 2 | `hybrid_rules_gazetteer_regex` | 1.000 | 0.000 | 1.000 | 1.000 | 0.010s | 0.250 MB | 0.777 |
| text | 3 | `regex_entity_extractor` | 1.000 | 0.000 | 1.000 | 1.000 | 0.010s | 0.250 MB | 0.777 |
| csv | 1 | `direct_table_mapper` | 0.214 | 0.000 | 1.000 | 1.000 | 0.010s | 0.250 MB | 0.752 |
| csv | 2 | `jsonpath_mapper` | 0.214 | 0.000 | 1.000 | 1.000 | 0.010s | 0.250 MB | 0.752 |
| json | 1 | `jsonpath_mapper` | 0.125 | 1.000 | 1.000 | 1.000 | 0.010s | 0.250 MB | 0.816 |
| json | 2 | `direct_table_mapper` | 0.062 | 0.000 | 1.000 | 1.000 | 0.010s | 0.250 MB | 0.747 |
| mixed | 1 | `hybrid_rules_gazetteer_regex` | 0.643 | 0.000 | 1.000 | 1.000 | 0.010s | 0.250 MB | 0.764 |
| mixed | 2 | `gazetteer_entity_extractor` | 0.357 | 0.000 | 1.000 | 1.000 | 0.010s | 0.250 MB | 0.753 |
| mixed | 3 | `regex_entity_extractor` | 0.286 | 0.000 | 1.000 | 1.000 | 0.010s | 0.250 MB | 0.750 |
| mixed | 4 | `hybrid_structured_then_text` | 0.429 | 0.000 | 1.000 | 1.000 | 0.010s | 0.250 MB | 0.748 |
| mixed | 5 | `jsonpath_mapper` | 0.214 | 0.000 | 1.000 | 1.000 | 0.010s | 0.250 MB | 0.740 |
| mixed | 6 | `direct_table_mapper` | 0.143 | 0.000 | 1.000 | 1.000 | 0.010s | 0.250 MB | 0.724 |

## How To Read This

- `entity_coverage`: extracted or mapped entities divided by the profile-estimated entity count.
- `relation_coverage`: extracted or mapped relations divided by the profile-estimated relation count.
- `evidence_coverage`: fraction of candidates with source evidence.
- `validation_pass_rate`: `1 - validation_failure_rate` from the validation metrics.
- `latency` and `memory/resource`: deterministic operational controls, not hardware benchmarks.
- `final_score`: weighted quality plus runtime score after penalties.

## Dataset Notes

- `text`: Pattern-heavy operational text with email, date, incident ID, version, teams, and service names.
- `csv`: Small structured table with IDs, names, departments, emails, and manager references.
- `json`: Nested JSON service payload with owner metadata, dependencies, and last incident details.
- `mixed`: Directory fixture combining CSV, JSON, and text sources.
