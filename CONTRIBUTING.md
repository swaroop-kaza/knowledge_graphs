# Contributing

Thanks for helping improve the Data-to-Knowledge Graph Construction Framework.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,rdf,graph]"
```

Optional extras:

```bash
python -m pip install -e ".[gliner,llm,linking,dedup,ml]"
```

## Local checks

Run the same checks expected before opening a pull request:

```bash
make quality
make verify
```

## Contribution rules

- Keep techniques reusable and technique-centered, not tied to one use case.
- Add or update tests for every loader, mapper, extractor, resolver, validator, graph store, or evaluator change.
- Return evidence-backed candidates from extractors; do not write directly to graph stores from extraction code.
- Keep generated outputs out of commits. Use `data/processed/` and `reports/experiment_datasets/` only as local workspaces.
- Keep optional integrations optional. If a feature needs GLiNER, Ollama, Neo4j, Kuzu, PyTorch Geometric, or external services, gate it behind an optional extra or explicit config.
- Prefer small pull requests with a clear explanation, test evidence, and any compatibility notes.

## Pull request checklist

- [ ] Tests were added or updated.
- [ ] `make quality` passes.
- [ ] `make verify` passes.
- [ ] Docs or examples were updated when behavior changed.
- [ ] Generated artifacts, secrets, downloaded datasets, and local notebooks outputs were not committed.
