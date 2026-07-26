# Research boundary

[Research guide](../../../research/README.md) · [Evaluation metrics](../../../docs/evaluation_metrics.md) · [Reports policy](../../../reports/README.md)

This package contains non-authoritative comparison and experimentation code:

| Module | Research responsibility |
| --- | --- |
| [`api.py`](api.py) | Public `evaluate_techniques(...)` entry point |
| [`simple_benchmark.py`](simple_benchmark.py) | Small candidate-graph benchmark runner |
| [`benchmark_runner.py`](benchmark_runner.py) | Configured weighted technique evaluation, gold summaries, and research hard gates |
| [`scoring.py`](scoring.py) | Research weights, penalties, thresholds, and technique scores |
| [`technique_selector.py`](technique_selector.py) · [`planner.py`](planner.py) · [`recommendation_engine.py`](recommendation_engine.py) | Applicability, experiment planning, and comparative recommendations |
| [`graph_ml.py`](graph_ml.py) | Graph analytics, embeddings, and link-prediction experiments |
| [`graph_store_benchmark.py`](graph_store_benchmark.py) | Graph-store comparison fixtures |
| [`hybrid_gliner.py`](hybrid_gliner.py) | Comparative GLiNER and label-proposal experiments |
| [`workflow.py`](workflow.py) | Research-owned draft workflow composition |

Research output can reveal trade-offs or regressions, but it cannot establish that a knowledge graph is correctly formed or release-eligible. Product construction lives in `kg_construction.structured` and `kg_construction.unstructured`; graph-bound correctness metrics live in `kg_construction.evaluation`; release authorization lives in `kg_construction.certification`.

The [compatibility modules under `kg_construction.application`](../application/README.md) are temporary re-export shims. New imports must use `kg_construction.research`.

Research reports—including weighted leaderboards and synthetic cross-source resolver comparisons—are not product evaluation reports. The underlying false-merge/false-split metrics remain product metrics in `kg_construction.evaluation`; only the comparative benchmark story is research.

## Public entry point

```python
from kg_construction.research import evaluate_techniques

report = evaluate_techniques(
    "examples/data_to_kg/sample.txt",
    source_type="text",
    techniques="regex_entity_extractor,hybrid_rules_gazetteer_regex",
)
print(report.best_technique)
```

Run the checked-in example from the repository root:

```bash
uv run python examples/research_technique_comparison.py
```

See the top-level [research guide](../../../research/README.md) for fixtures, commands, interpretation rules, and generated reports.
