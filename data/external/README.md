# External Data Workspace

[Dataset scripts](../../scripts/) · [Security policy](../../SECURITY.md) · [Research boundary](../../research/README.md)

This directory is the local workspace for downloaded or generated public dataset samples. The generated `experiment_datasets/` subtree is ignored by Git.

Do not commit downloaded datasets, archives, warehouse exports, vendor data, private data, or raw enterprise data. Use the reproducible dataset scripts from the repository root:

```bash
uv run python scripts/download_experiment_datasets.py
uv run python scripts/run_experiment_datasets.py
```

| Script | Role |
| --- | --- |
| [`download_experiment_datasets.py`](../../scripts/download_experiment_datasets.py) | Download and prepare declared public experiment inputs |
| [`run_experiment_datasets.py`](../../scripts/run_experiment_datasets.py) | Execute experiment-dataset construction and generate research output |

Before adding any dataset fixture to the repository, confirm every gate:

| Gate | Requirement |
| --- | --- |
| License | Redistribution and derivative-fixture use are explicitly allowed |
| Size | The reviewed fixture is small enough for ordinary Git workflows |
| Privacy | No secrets, PII, proprietary text, or private operational data |
| Provenance | Source URL, version, retrieval date, transformation, and license are documented |
| Reproducibility | A script or exact procedure can recreate the local input |

Downloaded data remains local unless a maintainer reviews and intentionally commits a redistributable fixture.

Versioned synthetic or redistributable fixtures belong under [`benchmarks/`](../../benchmarks/) with a manifest, provenance, license, explicit limitations, and reviewed gold judgments when applicable. Generated comparisons belong in the research workspace described by the [reports policy](../../reports/README.md).
