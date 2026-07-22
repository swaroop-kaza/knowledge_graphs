# Cross-Source Identity and Deduplication Benchmark v1

This small synthetic benchmark tests whether entity-resolution techniques preserve source records while identifying cross-source co-reference. It deliberately includes aliases, legal suffixes, punctuation variants, repeated names, and one hard namesake.

The three source files contain no gold cluster labels. The separate gold fixture defines eight positive record pairs. In particular, two `Alex Kim` records refer to one person while a third refers to a different person; a name-only merge therefore creates false positives.

Run and publish the benchmark with:

```bash
make cross-source-identity-benchmark
```

The command regenerates semantically stable Markdown, CSV, and JSON results under `reports/`. Dataset, configuration, generator, evaluated source tree, and result hashes identify the logical run; Git state and environment fields may legitimately differ between machines. A baseline may appear in the table even when it fails a correctness gate. Gate failure is a benchmark result, not a report-generation failure.

This fixture is intentionally too small for production model selection. It is a committed CI contract for gold wiring, false-merge accounting, provenance, order invariance, and publication reproducibility.
