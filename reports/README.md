# Reports

This directory keeps concise reports that describe the current repository implementation.

Current source-of-truth reports:

- `benchmark_leaderboard.md`
- `benchmark_leaderboard.csv`
- `gold_standard_benchmark_story.md`
- `gold_standard_benchmark_story.csv`
- `gold_standard_benchmark_story.json`
- `current_repo_capability_summary.md`
- `current_registry_summary.json`
- `current_registry_matrix.csv`

These reports include both:

- internal registry `status`: `implemented`, `adapter`, or `stub`
- public `support_level`: `stable`, `optional_adapter`, `experimental`, or `planned`

Refresh them with:

```bash
make benchmark-leaderboard
make gold-benchmark-story
make registry-reports
```

Generated experiment outputs belong under `reports/experiment_datasets/` and are intentionally not kept after cleanup because they can become large and stale. Regenerate them with `scripts/run_experiment_datasets.py` when you need a fresh dataset run.
