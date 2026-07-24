#!/usr/bin/env bash
# Run the immutable, registry-disconnected, zero-skip certification gate.

set -euo pipefail
IFS=$'\n\t'
cd "$(dirname "$0")"

command -v uv >/dev/null
command -v docker >/dev/null
test -f uv.lock
test -f models.lock.yaml
test -f certification/services.lock.yaml
test -f certification/execution_plan.yaml

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="results/certification_${timestamp}"
mkdir -p "$output_dir"

export UV_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
profile_name="${KG_CERT_PROFILE:-}"

# Locked images and model artifacts must already be local. These commands are
# intentionally incapable of pulling or resolving artifacts. Mandatory remote
# identity/reconciliation certifiers may still use their live endpoints.
docker compose -f certification/services.lock.yaml up -d --pull never
uv sync --frozen --offline --all-groups
if [[ -n "$profile_name" ]]; then
  PYTHONPATH=src uv run --frozen --offline python scripts/hydrate_models.py \
    --offline --validate-only --profile "$profile_name"
  PYTHONPATH=src uv run --frozen --offline python scripts/wait_for_services.py \
    --profile "$profile_name"
else
  PYTHONPATH=src uv run --frozen --offline python scripts/hydrate_models.py \
    --offline --validate-only
  PYTHONPATH=src uv run --frozen --offline python scripts/wait_for_services.py
fi

make certify \
  CERT_PROFILE="${KG_CERT_PROFILE:-}" \
  CERT_EXECUTION_PLAN=certification/execution_plan.yaml \
  CERT_EXECUTION_REPORT="$output_dir/execution-report.yaml" \
  2>&1 | tee "$output_dir/certification.log"

uv run --frozen --offline python -m kg_construction.interfaces.cli.doctor \
  --strict --json > "$output_dir/doctor.json"
cp capability_manifest.yaml models.lock.yaml certification/services.lock.yaml "$output_dir/"

printf 'certification passed: %s\n' "$output_dir"
