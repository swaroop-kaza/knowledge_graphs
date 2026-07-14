# GitHub Open-Source Readiness Plan

This document tracks what the repository needs before it is made public and promoted.

## Current readiness snapshot

Already strong:

- Source tree is compact and technique-centered.
- `make quality` and `make verify` pass locally.
- `make config-check` validates all public YAML examples.
- `make notebook-check` validates notebook hygiene and registry-to-notebook capability coverage.
- CI exists for Python 3.11 and 3.12.
- CI builds source/wheel distributions, runs `python -m twine check dist/*`, and smoke-tests installation from the built wheel.
- Generated smoke outputs and old article artifacts were removed.
- Current implementation inventory and registry reports exist.
- Stable top-level Python API exists for loading, profiling, construction, evaluation, config, workflows, results, and registry access.
- README has a core-install 60-second data scientist quickstart backed by `examples/quickstart_60_second.py` and `tests/test_quickstart.py`.
- README/docs clearly mark Stable, Optional adapter, Experimental, and Planned support levels for registered techniques.
- Reports include a small committed benchmark leaderboard for text, CSV, JSON, and mixed fixtures.

Recently added for GitHub readiness:

- `LICENSE`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- `SUPPORT.md`
- `CHANGELOG.md`
- `CITATION.cff`
- `.env.example`
- `.github/ISSUE_TEMPLATE/`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/dependabot.yml`
- `.github/CODEOWNERS`
- `data/external/README.md`

## Must finish before publishing

1. Confirm the open-source license.
   - Current default: Apache-2.0.
   - Use Apache-2.0 when you want a permissive license with explicit patent language.
   - Switch to MIT only if you want the simplest permissive license and do not need Apache-style patent terms.

2. Create the GitHub repository metadata.
   - Description: `Production-oriented Data-to-Knowledge Graph construction, evaluation, and technique selection framework.`
   - Topics: `knowledge-graphs`, `graph-rag`, `rdf`, `shacl`, `neo4j`, `entity-resolution`, `information-extraction`, `llm`, `gliner`, `data-engineering`.
   - Enable Issues.
   - Enable Discussions.
   - Add a social preview image later.

3. Update repository URLs after the remote exists.
   - Add `[project.urls]` in `pyproject.toml`.
   - Update `CITATION.cff` with the repository URL.
   - Update issue template security contact link.

4. Enable GitHub security features.
   - Dependabot alerts.
   - Dependabot security updates.
   - Secret scanning.
   - Code scanning with CodeQL.
   - Private vulnerability reporting.

5. Add branch protection.
   - Require pull requests before merging.
   - Require CI status checks.
   - Require conversation resolution.
   - Require linear history if desired.
   - Restrict force pushes on `main`.

6. Decide dataset policy.
   - Keep `data/external/experiment_datasets/` out of Git.
   - Keep only tiny synthetic fixtures under `tests/fixtures/`.
   - Document source/license for any future committed real-world fixture.

## Recommended public README improvements

Add a stronger first screen:

- One-line value proposition.
- A small architecture diagram or Mermaid pipeline.
- Capability matrix.
- Example output preview.
- Links to notebooks and docs.
- Badges for CI, license, Python versions, and package typing.

## Release readiness

Before the first GitHub release:

- Tag `v0.1.0`.
- Build the package with `python -m build`.
- Run `python -m twine check dist/*`.
- Install the generated wheel in a fresh environment and run `kg-construction doctor --skip-services`.
- Attach release notes from `CHANGELOG.md`.
- Publish a small example dataset run separately from the source repo if it is large.

## Visibility checklist

- Pin the repository on the GitHub profile.
- Add a clear repository description and topics.
- Publish one technical blog post that links to the repo.
- Add screenshots or diagrams to the README.
- Keep issues labeled and triaged.
- Create `good first issue` and `help wanted` labels after the first public release.
