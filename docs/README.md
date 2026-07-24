# Documentation

## Start here

- [`architecture.md`](architecture.md): the whole system — hexagonal layers,
  core data flow, the technique registry, and the certification hard pass, with
  Mermaid diagrams.

## Techniques (how each conversion works, technically)

- [`techniques_structured_to_kg.md`](techniques_structured_to_kg.md): every
  structured/semi-structured → graph technique — semantic profiling, the resolved
  plan and six projections, the source catalog, semantic materialization,
  standards mappings (R2RML/RML/CSVW/JSON-LD/OBDA), and JSONPath/event mappers.
- [`techniques_unstructured_to_kg.md`](techniques_unstructured_to_kg.md): the
  evidence-first pipeline — the mention ensemble, claim/relation/event
  extraction, attribute cues, identity resolution and linking, and the
  SHACL/provenance promotion gate.

## Evaluation and testing

- [`evaluation_metrics.md`](evaluation_metrics.md): every metric — what it
  measures, why, how it is calculated, the weighted scoring logic, and the ideal
  threshold for each gate.
- [`testing_guide.md`](testing_guide.md): the test ↔ technique map, gold
  fixtures, report gates, and how to test each technique end to end.

## Reference

- [`knowledge_graph_modelling.md`](knowledge_graph_modelling.md): the structured,
  unstructured, and evaluation design rules, with stage gates and metric
  definitions.
- [`stage_by_stage_implementation.md`](stage_by_stage_implementation.md): the
  executable implementation order, owning modules, artifacts, and hard exit
  condition for every stage.
- [`getting_started.md`](getting_started.md): provision the mandatory
  platform-specific runtime, immutable model/service locks, graph construction,
  and the offline zero-skip certification pass.
- [`../certification/README.md`](../certification/README.md): certification
  evidence and execution contracts.
