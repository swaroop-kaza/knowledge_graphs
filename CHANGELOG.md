# Changelog

All notable changes to this project will be documented here.

The project follows a simple public-release flow until formal semantic versioning starts.

## Unreleased

- Added Data-to-Knowledge Graph construction framework with loaders, mapping, extraction, validation, graph building, entity resolution, graph ML enrichment, production ingestion, and evaluation support.
- Added current registry reports under `reports/`.
- Added GitHub readiness files for public open-source publishing.
- Added conservative source-scoped structured identities and per-source routing for heterogeneous inputs.
- Added duplicate-control, referential-integrity, provenance-preservation, and benchmark-publication foundations.
- Updated the package and GitHub-facing description to emphasize evidence, multi-source identity, validation, and reproducible benchmarks.

### Migration warning

- `kg-identity-v2` changes text snapshot hashes, default structured canonical IDs, entity/relation/fact assertion IDs, and semantic edge IDs. Existing Neo4j, Kuzu, or exported property graphs created by an earlier checkout must be rebuilt or migrated before incremental ingestion; appending v2 output to a v1 store can leave obsolete nodes and edges beside their replacements.
