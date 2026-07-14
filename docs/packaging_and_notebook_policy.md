# Packaging And Notebook Policy

This repo uses modern Python packaging through `pyproject.toml`.

## Setup files

Required:

- `pyproject.toml`: build system, package metadata, dependencies, optional extras, console scripts, and tool config.
- `LICENSE`: Apache-2.0 license text.
- `README.md`: package long description and public onboarding.
- `src/kg_construction/py.typed`: marks the package as typed.

Not required:

- `setup.py`
- `setup.cfg`
- `requirements.txt`

Reason:

- `pyproject.toml` is the source of truth for editable installs, builds, package metadata, optional extras, and tool configuration.
- Separate requirements files would duplicate dependency declarations and can drift from package metadata.
- Legacy setup files are only useful if a downstream environment cannot consume PEP 517/518 style projects.

## Recommended install commands

Developer install:

```bash
python -m pip install -e ".[dev,rdf,graph]"
```

Notebook install:

```bash
python -m pip install -e ".[dev,rdf,graph,notebook]"
```

Optional integration install:

```bash
python -m pip install -e ".[gliner,llm,linking,dedup,ml]"
```

## Notebook policy

The notebooks are examples and tutorials, not generated reports.

They should remain:

- Output-free.
- Execution-count-free.
- Path-portable.
- Offline-friendly where possible.
- Focused on calling repo APIs rather than duplicating implementation logic.

Notebook capability coverage is explicit:

- `notebooks/capability_coverage.yaml` maps implemented and adapter registry techniques to notebook examples.
- `scripts/check_notebook_coverage.py` fails when a public technique is added without a notebook entry.
- Stub techniques are documented as extension points, not treated as already executable examples.

Run:

```bash
make notebook-check
```

This check fails if a notebook contains stored outputs, execution counts, absolute local paths, stale article references, generated experiment paths, or missing registry coverage.

## Configuration Policy

Public YAML examples live under `configs/`.

Run:

```bash
make config-check
```

This validates every example config for YAML syntax, expected top-level sections, known technique names, generated-output path policy, and model-backed settings where the repo has a typed config model.
