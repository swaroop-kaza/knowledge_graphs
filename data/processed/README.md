# Processed Data Workspace

[Getting started](../../docs/getting_started.md) · [Testing guide](../../docs/testing_guide.md) · [Reports policy](../../reports/README.md)

This directory is reserved for generated local outputs. Git ignores everything here except this README.

| Output class | Typical files | Authority |
| --- | --- | --- |
| Graph projections | Property graph JSON, RDF JSON, evidence graph JSON, and Cypher | Inspectable draft output |
| Evaluation | Quality and intended-use JSON reports | Computed metrics; not a release certificate |
| Research | Technique-comparison Markdown, CSV, JSON, and candidate graphs | Non-authoritative comparison |
| Operations | Dead-letter queues, manifests, and local graph-store workspaces | Local runtime state |

Regenerate the normal sample outputs from the checked-in [examples](../../examples/README.md) and [configurations](../../configs/):

```bash
make product-smoke
make research-smoke
```

These files are reproducible from the source code, examples, and configs, so they are not kept as canonical repo artifacts.

> **Treat generated evidence as source data.** Property, evidence, dead-letter, and report outputs may retain source text, identifiers, or extracted personal data. Do not commit or share them without reviewing the [security policy](../../SECURITY.md) and the source-data license.
