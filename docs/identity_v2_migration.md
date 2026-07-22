# `kg-identity-v2` Migration

This development release introduces a safer graph identity policy. It is intentionally breaking for graphs materialized by earlier `0.1.0` source-tree revisions.

## What Changed

- Text source snapshots now hash exact text instead of normalized text, so case and whitespace changes are observable.
- Evidence locations use canonical nested serialization, so evidence IDs can change while becoming independent of mapping key order.
- Structured rows use source-scoped configured keys or conservative row-content fallbacks instead of display names.
- Missing and redacted key values remain separate rather than sharing one unknown identifier.
- Name-only exact matching is source-scoped by default; global name matching is an explicit benchmark/legacy option.
- Entity, relation, and fact assertion IDs include property/provenance payloads so distinct governed keys or occurrences cannot overwrite one another.
- Canonical property-graph edges use semantic claim keys and consolidate repeated assertions while preserving all evidence.

New property and RDF graph outputs declare `metadata.identity_policy_version = "kg-identity-v2"`.

## Required Action

Do not append v2 output directly to an existing v1 graph. Choose one of these approaches:

1. For reproducible or development graphs, drop the generated graph and rebuild it from versioned source snapshots.
2. For retained graphs, export a v1 ID-to-source-evidence inventory, generate v2 output in a staging graph, validate counts/competency queries, redirect consumers, and only then retire v1 objects.
3. For stores that cannot be rebuilt, write an application-specific migration keyed by authoritative source identifiers. Do not match old and new objects by display name alone.

Before cutover, compare entity and semantic-edge counts, false-merge samples, dangling edges, provenance coverage, and domain competency queries. Keep the old graph or export until the v2 run is accepted.
