# Mandatory locked-artifact certification

A release passes only when all six capability sets are identical:

```text
declared = configured = instantiated = executed = evaluated = passed
```

The final report must also contain zero skipped and zero fallback executions.
`scripts/certify.py` dispatches every capability itself under a fresh run nonce;
it does not accept a pre-authored report as trusted input.

## Mandatory profiles

- `darwin-arm64`: MLX, Torch MPS, Ollama, GLiNER, Transformers, spaCy, Neo4j,
  Memgraph, and embedded Ladybug.
- `linux-cpu`: Torch CPU plus the same model and graph-service surface.
- `linux-cuda`: CUDA plus the same model and graph-service surface. CPU fallback
  is a certification failure.

The Qwen generation family is consistent across runtimes:
`Qwen/Qwen3-1.7B`, `mlx-community/Qwen3-1.7B-4bit`, and
`qwen3:1.7b-q4_K_M`.

## Immutable inputs

| Source | Generated lock |
| --- | --- |
| `pyproject.toml` | `uv.lock` |
| `models.manifest.yaml` | `models.lock.yaml` |
| `certification/services.yaml` | `certification/services.lock.yaml` |

Model locks contain immutable revisions and a SHA-256/size for every file.
Service locks are valid Docker Compose files whose image references use
`repository@sha256:digest`. Ollama model content has its own digest in the model
lock. Both generated locks include a SHA-256 of their human-authored source, so
editing a manifest invalidates its lock. The model lock also records unique
snapshot and Ollama byte totals so disk planning is based on the resolved
artifacts rather than duplicated per-profile rows.

Lock generation is the explicit online operation:

```bash
make release-locks
```

`lock_models.py` always locks every profile; it has no partial-profile mode. It
also pulls a missing mandatory Ollama model before recording its digest. This
online lock operation can be large because GLiNER, GLiNER RelEx, Transformers,
sentence-transformers, spaCy, Qwen3, and Darwin MLX artifacts are all mandatory.

Provision the locked artifacts once:

```bash
make setup
```

Then run with dependency/model registries disconnected:

```bash
bash run_all_approaches.sh
```

“Registry-disconnected” is intentionally narrower than “no network.” The
certificate proves there are no package, Hugging Face model, Ollama-model, or
container downloads during the run. Capabilities whose purpose is remote
identity/reconciliation still call their live endpoints and fail when those
endpoints are unavailable. Ollama, Neo4j, and Memgraph remain mandatory local
services; Ladybug remains a mandatory embedded runtime.

`openrefine_reconciler` exercises the OpenRefine Reconciliation API v0.2 batch
protocol against `https://wikidata-reconciliation.wmcloud.org/en/api`. This is a
live Wikimedia Cloud service check, not a probe of a local OpenRefine desktop
application.

`certification/execution_plan.yaml` must contain a real certifier command for
every row in `capability_manifest.yaml`. Each command receives a fresh
`KG_CERT_RUN_ID` and must write semantic evidence to
`KG_CERT_EVIDENCE_PATH`. Missing contracts, stale evidence, non-zero commands,
missing stage evidence, skips, and fallbacks are all hard failures.

The plan is intentionally not pre-populated with weak import-only checks.
Until every capability owns a semantic certifier, the repository remains
truthfully uncertified.
