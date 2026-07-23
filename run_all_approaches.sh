#!/usr/bin/env bash
# run_all_approaches.sh
# Exercise every available kg_construction approach and store all results in one place.
# Safe to re-run. Optional approaches (Ollama/GLiNER/Neo4j/ML) are attempted and their
# pass/fail is recorded rather than stopping the run. Local helper — delete anytime.
#
# Usage:
#   make setup            # once, if you haven't installed yet
#   bash run_all_approaches.sh

set -uo pipefail
cd "$(dirname "$0")"                      # run from the repo root

# Prefer the project virtualenv's console scripts if present.
[ -x ".venv/bin/kg-construction" ] && export PATH="$PWD/.venv/bin:$PATH"

if ! command -v kg-construction >/dev/null 2>&1; then
  echo "kg-construction not found. Run 'make setup' first (creates .venv and installs)." >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT="results/run_${TS}"
mkdir -p "$OUT"/{00_environment,01_registry_catalogue,02_construction,03_technique_evaluation,04_gold_and_identity,05_optional_services}
RUNLOG="$OUT/run.log"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUNLOG"; }
step() { log ""; log "==== $* ===="; }

log "Results directory: $OUT"

########################################################################
step "Phase 1 - Environment & availability (what approaches are installed?)"
kg-construction doctor           > "$OUT/00_environment/doctor.txt"  2>&1 || true
kg-construction doctor --json    > "$OUT/00_environment/doctor.json" 2>&1 || true
{ python3 --version; python3 -c "import sys;print(sys.executable)"; } > "$OUT/00_environment/python.txt" 2>&1 || true
log "Saved doctor report + python info."

########################################################################
step "Phase 2 - Full technique catalogue (all 150 registered approaches)"
make registry-reports >> "$RUNLOG" 2>&1 || log "registry-reports failed (see run.log)"
cp -f reports/current_registry_matrix.csv       "$OUT/01_registry_catalogue/" 2>/dev/null || true
cp -f reports/current_registry_summary.json     "$OUT/01_registry_catalogue/" 2>/dev/null || true
cp -f reports/current_repo_capability_summary.md "$OUT/01_registry_catalogue/" 2>/dev/null || true
log "Saved the full technique/support-level catalogue."

########################################################################
step "Phase 3 - Construction across ALL 11 source types (loaders + graph builders + exporters)"
# type:input_file  (all run with the core install)
construct_targets=(
  "csv:examples/data_to_kg/sample.csv"
  "json:examples/data_to_kg/sample.json"
  "text:examples/data_to_kg/sample.txt"
  "xml:tests/fixtures/enterprise_loaders/services.xml"
  "excel:tests/fixtures/enterprise_loaders/service_inventory.xlsx"
  "pdf:tests/fixtures/enterprise_loaders/incident_brief.pdf"
  "email:tests/fixtures/enterprise_loaders/incident_email.eml"
  "markdown:tests/fixtures/enterprise_loaders/incident_review.md"
  "report:tests/fixtures/enterprise_loaders/ops_report.report"
  "chat_transcript:tests/fixtures/enterprise_loaders/support_chat.chat"
  "api_response:tests/fixtures/enterprise_loaders/api_response.json"
)
for pair in "${construct_targets[@]}"; do
  t="${pair%%:*}"; f="${pair#*:}"
  d="$OUT/02_construction/$t"; mkdir -p "$d"
  if kg-construction --input "$f" --source-type "$t" \
        --output        "$d/property_graph.json" \
        --rdf-output    "$d/rdf_graph.json" \
        --evidence-output "$d/evidence_graph.json" \
        --neo4j-output  "$d/neo4j.cypher" > "$d/log.txt" 2>&1; then
     log "  OK    $t"
  else
     log "  FAIL  $t (see $d/log.txt)"
  fi
done

########################################################################
step "Phase 4 - Technique evaluation with --techniques all (every approach, side by side)"
# benchmark supports csv/json/text; plus a mixed directory.
bench_targets=(
  "csv:examples/data_to_kg/sample.csv"
  "json:examples/data_to_kg/sample.json"
  "text:examples/data_to_kg/sample.txt"
)
for pair in "${bench_targets[@]}"; do
  t="${pair%%:*}"; f="${pair#*:}"
  d="$OUT/03_technique_evaluation/$t"; mkdir -p "$d"
  if kg-construction-benchmark --input "$f" --source-type "$t" \
        --config configs/evaluation.example.yaml --techniques all \
        --output      "$d/report.md" \
        --json-output "$d/report.json" \
        --csv-output  "$d/leaderboard.csv" \
        --candidate-output-dir "$d/candidates" > "$d/log.txt" 2>&1; then
     log "  OK    benchmark $t (all techniques)"
  else
     log "  FAIL  benchmark $t (see $d/log.txt)"
  fi
done
# Mixed-source directory (heterogeneous routing)
dmix="$OUT/03_technique_evaluation/mixed_directory"; mkdir -p "$dmix"
if kg-construction-benchmark --input tests/fixtures/golden_benchmarks/mixed \
      --config configs/evaluation.example.yaml --techniques all \
      --output "$dmix/report.md" --json-output "$dmix/report.json" \
      --csv-output "$dmix/leaderboard.csv" > "$dmix/log.txt" 2>&1; then
   log "  OK    benchmark mixed_directory"
else
   log "  FAIL  benchmark mixed_directory (see $dmix/log.txt)"
fi

########################################################################
step "Phase 5 - Gold-standard, identity, and leaderboard benchmarks (reproducible reports)"
for tgt in benchmark-leaderboard gold-benchmark-story cross-source-identity-benchmark; do
  make "$tgt" >> "$RUNLOG" 2>&1 && log "  OK    make $tgt" || log "  FAIL  make $tgt (see run.log)"
done
cp -f reports/benchmark_leaderboard.md reports/benchmark_leaderboard.csv \
      reports/gold_standard_benchmark_story.md reports/gold_standard_benchmark_story.csv reports/gold_standard_benchmark_story.json \
      reports/cross_source_identity_benchmark.md reports/cross_source_identity_benchmark.csv reports/cross_source_identity_benchmark.json \
      "$OUT/04_gold_and_identity/" 2>/dev/null || true

########################################################################
step "Phase 6 - Optional local services (only if you have them running)"
if curl -fsS "http://localhost:11434/api/tags" >/dev/null 2>&1; then
  log "  Ollama detected -> running the Ollama graph extractor through the benchmark"
  # Use the EVALUATION config (not the pipeline config) and target an Ollama technique.
  # Requires the model to be pulled:  ollama pull llama3.1
  kg-construction-benchmark --input examples/data_to_kg/sample.txt --source-type text \
     --config configs/evaluation.example.yaml --techniques ollama_graph_extractor \
     --output "$OUT/05_optional_services/ollama_report.md" \
     --json-output "$OUT/05_optional_services/ollama_report.json" \
     > "$OUT/05_optional_services/ollama.log" 2>&1 \
     && log "  OK    Ollama benchmark (see report; pass/fail depends on 'ollama pull llama3.1')" \
     || log "  FAIL  Ollama benchmark (see log; try 'ollama pull llama3.1')"
else
  log "  Ollama not reachable at localhost:11434 - skipped (start it + 'ollama pull llama3.1' to include)."
fi
if [ -n "${NEO4J_URI:-}" ]; then
  log "  NEO4J_URI set - Neo4j ingestion would run here (Cypher export already produced in Phase 3)."
else
  log "  NEO4J_URI not set - live Neo4j ingestion skipped (Cypher export still produced in Phase 3)."
fi

########################################################################
step "Writing index"
{
  echo "# kg_construction - all-approaches run"
  echo ""
  echo "Generated: $TS"
  echo ""
  echo "| Folder | What it contains |"
  echo "| --- | --- |"
  echo "| 00_environment/ | doctor report - which optional approaches are installed |"
  echo "| 01_registry_catalogue/ | every registered technique + support level (the full catalogue) |"
  echo "| 02_construction/<type>/ | property, RDF, evidence graphs + Cypher for all 11 source types |"
  echo "| 03_technique_evaluation/<type>/ | \`--techniques all\` leaderboard: every approach scored side by side |"
  echo "| 04_gold_and_identity/ | gold NER/relation/triple/ER/retrieval + cross-source identity + leaderboard |"
  echo "| 05_optional_services/ | Ollama/Neo4j results (only if those services were available) |"
  echo ""
  echo "Start with: 01_registry_catalogue/current_repo_capability_summary.md and"
  echo "03_technique_evaluation/text/report.md"
} > "$OUT/INDEX.md"

log ""
log "DONE. All results in: $OUT"
log "Open $OUT/INDEX.md first."
echo ""
echo "Note: Phase 2/5 refresh tracked files under reports/ (timestamps may differ)."
echo "      Discard those repo changes with:  git checkout -- reports/   (results copies are kept under $OUT)."
