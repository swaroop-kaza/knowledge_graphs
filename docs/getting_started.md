# Getting started: mandatory, zero-skip runtime

This repository has one product contract: construct a well-formed graph from structured and unstructured data, evaluate it, and fail when a required provider did not run. There is no reduced “core install”, no model-download-on-first-use, and no successful result containing skipped capabilities.

Research and roadmap code remains visible in `research_registry()`, but it is not part of the release registry and is never counted as certified.

## 1. Supported execution profiles

The hard pass selects exactly one profile from `models.manifest.yaml`.

| Host | Required execution |
| --- | --- |
| Apple Silicon macOS | Torch MPS, MLX/MLX-LM, GLiNER, Transformers, spaCy, Qwen3 through Transformers + MLX + Ollama |
| Linux x86-64 with CUDA | CUDA, GLiNER, Transformers, spaCy, Qwen3 through Transformers + Ollama |
| Linux x86-64 without CUDA | Torch CPU, GLiNER, Transformers, spaCy, Qwen3 through Transformers + Ollama |

Every profile also requires Ollama, Neo4j, Memgraph, and embedded Ladybug. A requested profile that does not match the detected host is rejected; an accelerated profile never silently falls back to CPU.

Windows is not a certification profile. Use a supported Linux environment rather than treating a partial Windows run as equivalent.

## 2. Install the bootstrap tools

Install Python 3.11 or 3.12, Git, Docker with the Buildx and Compose plugins, and [`uv`](https://docs.astral.sh/uv/).

On Apple Silicon:

```bash
brew install python@3.12 git uv
brew install --cask docker
```

On Linux, install Docker Engine from its distribution instructions, then:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify the host tools before provisioning:

```bash
python3 --version
uv --version
docker version
docker buildx version
docker compose version
```

## 3. Clone and create immutable locks

```bash
git clone https://github.com/swaroop-kaza/knowledge_graphs.git
cd knowledge_graphs
make release-locks
```

This one maintainer target, in order:

- **resolves** every container tag;
- **starts** the version-pinned bootstrap services;
- **resolves** every Hugging Face model to an immutable commit;
- **downloads and hashes** every artifact; and
- **pulls** any missing mandatory Ollama model and records its content digest.

It always locks all three profiles; a host-only/partial release lock is rejected.

The generated inputs are:

- `uv.lock`: every Python distribution;
- `certification/services.lock.yaml`: every container image by immutable digest;
- `models.lock.yaml`: model commit, file list, SHA-256, size, device, license, and Ollama digest.

Commit all three locks for a reproducible release. Do not hand-edit them.

## 4. Provision the selected profile

```bash
make setup
```

`make setup`, in order:

- **installs** every dependency group;
- **starts** the digest-pinned services;
- **hydrates** the locked model snapshots and **validates** their bytes;
- **waits** for semantic service probes; and
- **runs** the strict doctor.

The first provisioning run needs network access and significant disk space. The hard pass later disconnects package, model, and container registries; mandatory remote identity and reconciliation endpoint checks still require network access.

On Apple Silicon, the setup must find both `mlx` and `mlx_lm`, and Torch must report MPS as built and available. On CUDA Linux, CUDA must be available. These are hard conditions, not preferences.

Inspect the selected contract:

```bash
uv run python -m kg_construction.interfaces.cli.doctor --strict
uv run python -m kg_construction.interfaces.cli.doctor --strict --json
```

Doctor has only `ok` and `fail` outcomes. Missing models, services, packages, or configuration are failures.

## 5. Construct graphs

Structured sources use the semantic source catalog by default:

```bash
uv run kg-construction \
  --input examples/data_to_kg/sample.csv \
  --source-type csv \
  --config configs/data_to_kg_complete.example.yaml \
  --evaluation-suite configs/graph_evaluation_suite.sample_csv.yaml \
  --output data/processed/sample.property.json \
  --rdf-output data/processed/sample.rdf.json \
  --evidence-output data/processed/sample.evidence.json \
  --neo4j-output data/processed/sample.neo4j.cypher
```

Unstructured sources load the immutable model lock and run every provider required by the host:

```bash
uv run kg-construction \
  --input examples/data_to_kg/sample.txt \
  --source-type text \
  --config configs/data_to_kg_complete.example.yaml \
  --evaluation-suite configs/graph_evaluation_suite.sample_text.yaml \
  --output data/processed/text.property.json \
  --rdf-output data/processed/text.rdf.json \
  --evidence-output data/processed/text.evidence.json
```

The unstructured command runs, on every applicable profile:

- **GLiNER**, **Transformers NER**, **spaCy**, **GLiNER RelEx**;
- **Qwen3** through Transformers and Ollama; and
- **Qwen3 MLX** additionally on Apple Silicon.

A provider failure is aggregated after all providers have been attempted, then fails the run. Output values and qualifiers are accepted only when their end-exclusive offsets quote the immutable source snapshot exactly.

PII redaction creates a derived source snapshot whenever it changes text or rows. The derived `source_id` is also the new document version; it binds the redacted payload, transformation version, and parent source version. The safe metadata records `_source_lineage`, and the evidence graph materializes a `DERIVED_FROM` edge to a provenance-only parent snapshot node. Consequently, an evaluation suite for redacted text must use the post-redaction source/version and offsets. `graph_evaluation_suite.sample_text.yaml` is sealed against that derived snapshot, not the unredacted file identity.

Python uses the same contract:

```python
from kg_construction import construct_kg

result = construct_kg(
    "examples/data_to_kg/sample.csv",
    source_type="csv",
    config="configs/data_to_kg_complete.example.yaml",
    evaluation_suite_path="configs/graph_evaluation_suite.sample_csv.yaml",
)
assert result.validation_report.valid
assert result.publication_gate.passed
assert result.publication_contract is not None
assert result.publication_contract.release_eligible
print(result.quality_report)
print(result.intended_use_report)
```

The fixture's own SHA-256 detects accidental edits but does not grant release trust. The package-owned `src/kg_construction/resources/publication_contracts_v1.yaml` also binds the fixture file, raw input snapshot, deterministic post-PII runtime snapshot, semantic configuration, `models.lock.yaml`, governed unstructured schema, held-out claim set, graph-builder version, and non-weakenable threshold policy. `construct_kg` checks that allowlist before models, services, graph construction, or output writes. A custom suite can still be loaded with `VersionedGraphEvaluationSuite.from_path` and used in an explicitly non-publishable `ConstructionWorkflow`; it is classified as research-only until maintainers review and commit a new release contract. The CLI persists the accepted binding beside the graph as `*.publication-contract.json` (or at `--publication-contract-output`) so the source/config/artifact/policy digests travel with the release evidence.

For multi-table sources, declare primary keys, foreign keys, composite keys, identity scope, controlled vocabularies, and association-table semantics under `structured.tables` in the configuration. Undeclared structure can be profiled, but confidence is retained and missing foreign-key targets are never fabricated.

## 6. Evaluate by intended use

Construction quality and downstream fitness are different questions. The hard evaluation suite records whether each metric actually executed.

- Universal formation gates: evidence/provenance coverage, duplicate nodes and edges, categorical duplication, dangling edges, SHACL conformance, datatype, domain/range, cardinality, and orphan rates.
- Structured gold data: key preservation, property/relationship completeness, join accuracy, controlled-vocabulary uniqueness, and round-trip checks.
- Unstructured gold data: entity/span, relation, event-role, intent, fact, negation, modality, temporal, attribution, evidence, and provenance precision/recall/F1.
- Entity resolution: pairwise and cluster precision/recall/F1, false-merge and false-split rates, and record-universe coverage.
- Graph query / SQL-like use: row precision/recall/F1, exact result and aggregation accuracy, and p50/p95 latency.
- NoSQL traversal: node precision/coverage, exact traversal result, path F1, and p50/p95 latency.
- GraphRAG: Recall@k, MRR, nDCG, path F1, multi-hop success, citation F1, and answer faithfulness.
- Vector retrieval: index coverage, Recall@k, MRR, nDCG, filter correctness, p50/p95 latency, and peak memory.
- Link prediction: filtered MRR and Hits@k, reported separately for head/torso/tail popularity strata.
- Recommendation: Precision/Recall/nDCG@k, catalog coverage, novelty, and intra-list diversity.
- Reasoning: competency-question accuracy, inferred-fact F1, and p95 runtime.

Run the repository reports:

```bash
make modelling-correctness
make schema-quality
make benchmark-leaderboard
```

A missing gold fixture or judgment set is “not executed” and fails a hard gate that requires it; it is never converted into a zero score that looks measured.

## 7. Run the complete hard pass

After provisioning, disconnecting package/model/container registries must not change the result:

```bash
bash run_all_approaches.sh
```

The script, in order:

- **sets** `UV_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`;
- **starts** only locally cached digest-pinned images;
- **verifies** locked model bytes;
- **runs** quality and tests;
- **dispatches** every capability certifier with a fresh nonce; and
- **validates**:

```text
declared = configured = instantiated = executed = evaluated = passed
```

Live remote identity/reconciliation certifiers still use their documented endpoints; those calls are not model-artifact downloads. The final certificate requires zero skips and zero fallbacks. See [`../certification/README.md`](../certification/README.md) for the evidence format.

The OpenRefine-compatible identity check uses the public Wikimedia Cloud Reconciliation API at `https://wikidata-reconciliation.wmcloud.org/en/api`. “OpenRefine-compatible” describes the batch reconciliation protocol; it does not imply that a local OpenRefine desktop server is running. Like the Wikidata and DBpedia linkers, this mandatory certifier requires live network access and upstream availability.

## 8. Troubleshooting

| Failure | Meaning and action |
| --- | --- |
| `missing uv.lock` | Run `uv lock` during the provisioning phase and commit it. |
| `models.lock.yaml is missing` | Start Ollama with Qwen3 present, then run `uv run python scripts/lock_models.py`. |
| model snapshot/file digest mismatch | Rehydrate from the committed lock; do not let runtime download another revision. |
| service lock missing | Run `python3 scripts/lock_service_digests.py` with Docker Buildx online. |
| Ollama digest mismatch | The local tag does not contain the locked model bytes; restore the locked artifact. |
| MPS/MLX failure on Apple Silicon | Install the locked dependencies and run outside an environment that hides the GPU. |
| Neo4j or Memgraph probe fails | Inspect `docker compose -f certification/services.lock.yaml ps` and service logs. |
| certification plan mismatch | Every release-manifest capability needs an executable semantic certifier contract. |
| SHACL or modelling gate fails | Fix the mapping/schema or quarantine the candidate; do not suppress the gate. |

Research experiments can be run separately, but their outputs do not weaken or replace this release contract.
