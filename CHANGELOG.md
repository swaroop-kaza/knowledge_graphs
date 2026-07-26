# Changelog

[Documentation](docs/README.md) · [Contributing](CONTRIBUTING.md) · [Support](SUPPORT.md) · [Security](SECURITY.md)

All notable changes to this project will be documented here.

The project follows a simple public-release flow until formal semantic versioning starts. Entries follow the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- Added the Data-to-Knowledge Graph construction framework with loaders, mapping, extraction, validation, graph building, entity resolution, graph ML enrichment, production ingestion, and evaluation support.
- Added current registry and modelling reports under [`reports/`](reports/README.md).
- Added GitHub readiness files for public open-source publishing.
- Added conservative source-scoped structured identities and per-source routing for heterogeneous inputs.
- Added duplicate-control, referential-integrity, provenance-preservation, and benchmark-publication foundations.

### Changed

- Updated the package and GitHub-facing description to emphasize evidence, multi-source identity, validation, and reproducible benchmarks.
- Reorganized the product around the canonical [`structured`](src/kg_construction/structured/), [`unstructured`](src/kg_construction/unstructured/), and [`evaluation`](src/kg_construction/evaluation/) pillars.
- Made the platform-applicable GLiNER, Transformers, Ollama, MLX, graph-service, identity, and validation stack mandatory in certification.
- Replaced the former evaluation wildcard façade with canonical implementations and explicit compatibility exports.

### Migration warning

> **Rebuild required for earlier graph stores.**

`kg-identity-v2` changes text snapshot hashes, default structured canonical IDs, entity/relation/fact assertion IDs, and semantic edge IDs. Existing Neo4j, Kuzu, or exported property graphs created by an earlier checkout must be rebuilt or migrated before incremental ingestion; appending v2 output to a v1 store can leave obsolete nodes and edges beside their replacements.
