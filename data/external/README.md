# External Data Workspace

This directory is reserved for downloaded or generated public dataset samples.

Do not commit downloaded datasets, archives, warehouse exports, vendor data, private data, or raw enterprise data. Use the dataset scripts to recreate local experiment inputs when needed:

```bash
python3 scripts/download_experiment_datasets.py
python3 scripts/run_experiment_datasets.py
```

Before adding any dataset fixture to the repository, confirm:

- The dataset license allows redistribution.
- The fixture is small enough for GitHub.
- The fixture does not contain secrets, PII, proprietary text, or private operational data.
- The source and license are documented beside the fixture.
