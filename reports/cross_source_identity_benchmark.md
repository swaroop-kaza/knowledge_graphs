# Cross-Source Identity and Deduplication Benchmark

Benchmark: `cross_source_identity_dedup` v1.0.0

Status: **gold-candidate synthetic correctness fixture**. It is published for review and CI, not presented as a production-scale or domain-release leaderboard.

The fixture contains aliases, legal suffix variations, exact repeated names, and a deliberate namesake. A failed gate is reported honestly; rows are never removed because they score poorly.

## Reproduce

```bash
make cross-source-identity-benchmark
```

- Run ID: `e9ad00a9d718dff0`
- Dataset hash: `4ed31b012643663e3aa32ad3b6c0228bf9f0be7b375d757f6ab0d8f4729fe3cb`
- Input hash: `36acf7cb9eedd83454f647f1542c8990474dd2231bf5212be40e0db21689fb0b`
- Gold hash: `11825ea8262338d1b1f60997ffe54d9d96d26b6ee3ca4640119fa3e4dbe91432`
- Manifest hash: `9107ee02d7ba4eed59ef5612f0642c7832006f4cb261d01c523652eb0198cf19`
- Config hash: `bd04d1bfe1fe2cd1c70359e6bf17fd51b0050dafb97eecfd1e98f033b4f228f2`
- Git commit: `42d0100ae71f2d46320b90532ceb9f00ddcec51c`
- Git tree state: `dirty`
- Generator hash: `c403349333a3c5a221c8b8bbd33cee44893d4aecf357ac7bde82c0c687164222`
- Source tree hash: `1055f176c82051bb3aea6e0a52e506094fe17d5bc98b0c4767ba70d0ab41a96a`
- Results hash: `e7356ff86936af5b693240fd52d102e998c0212b9953f28ce4851ed4c9da54e3`
- Python: `3.12.9`

## Results

| Baseline | Pairs | Precision | Recall | F1 | False merge | False split | Evidence | Order invariant | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `exact_normalized_type_and_name` | 4 | 0.500 | 0.250 | 0.333 | 0.500 | 0.750 | 1.000 | True | FAIL |
| `normalized_name_and_legal_suffix` | 8 | 0.750 | 0.750 | 0.750 | 0.250 | 0.250 | 1.000 | True | FAIL |
| `curated_alias_table` | 10 | 0.800 | 1.000 | 0.889 | 0.200 | 0.000 | 1.000 | True | FAIL |
| `trusted_identifier_exact` | 8 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | True | PASS |

## Hard Gates

- Pairwise precision must be at least `0.98`.
- False-merge rate must be at most `0.02`.
- Evidence coverage must be at least `1.00`.
- Source-record order invariance is required: `True`.

## Interpretation

The name-based baselines expose the core multi-source identity risk. Exact names miss aliases and merge namesakes. Normalization and curated aliases improve recall but do not make a person name unique. The trusted-identifier reference baseline passes this fixture because its identifiers are explicitly scoped and authoritative.

Do not generalize the perfect trusted-identifier score beyond this fixture: real systems must validate identifier issuer, scope, reuse, and temporal validity.

## Limitations

- Tiny synthetic correctness fixture; it is not representative of production scale.
- It evaluates clustering only, not field survivorship or temporal updates.
- The trusted-identifier reference is deliberately easy and is not independent evidence that a learned resolver generalizes.
