BASE_PYTHON ?= python3
VENV ?= .venv
UV ?= uv
CERT_PROFILE ?=
CERT_PROFILE_ARG = $(if $(CERT_PROFILE),--profile $(CERT_PROFILE),)
CERT_SERVICE_LOCK ?= certification/services.lock.yaml
CERT_EXECUTION_PLAN ?= certification/execution_plan.yaml
CERT_EXECUTION_REPORT ?= certification/execution-report.yaml

# Use the project virtualenv interpreter when it exists, otherwise fall back to
# BASE_PYTHON. This is re-evaluated on each use, so `make setup quickstart` in a
# single command still picks up the venv created earlier in the same run, and CI
# (which never creates .venv) keeps using its system python3 unchanged.
# Override explicitly with `make PYTHON=/path/to/python <target>`.
ifndef PYTHON
PYTHON = $(shell [ -x "$(VENV)/bin/python" ] && echo "$(VENV)/bin/python" || echo "$(BASE_PYTHON)")
endif
PYTHONPATH ?= src

.DEFAULT_GOAL := help
.PHONY: benchmark-leaderboard capability-manifest certify certify-preflight clean compile config-check cross-source-identity-benchmark doctor gold-benchmark-story help hydrate install lock-check modelling-correctness notebook-check package-check quality quickstart registry-reports release-locks schema-quality services-start services-wait setup smoke test venv verify

help:
	@echo "kg-construction - common tasks"
	@echo ""
	@echo "  make release-locks Create every release lock (online, large download)"
	@echo "  make setup         Provision every locked dependency, model, and service"
	@echo "  make install       Install the complete mandatory dependency set"
	@echo "  make doctor        Hard-fail on any missing runtime, model, or service"
	@echo "  make quickstart    Build sample graphs and print a summary"
	@echo "  make test          Run the unit test suite"
	@echo "  make smoke         Run construction + benchmark smoke flows"
	@echo "  make verify        Run tests and smoke flows"
	@echo "  make quality       Run ruff, mypy, vulture, config, and notebook checks"
	@echo ""
	@echo "First run (no virtualenv setup needed - 'setup' creates one for you):"
	@echo "  make setup && make quickstart"

# Create the project virtualenv on demand and upgrade pip inside it.
$(VENV)/bin/python:
	$(BASE_PYTHON) -m venv "$(VENV)"
	"$(VENV)/bin/python" -m pip install --upgrade pip

venv: $(VENV)/bin/python

# Maintainer operation: resolve every supported profile, never a host-only subset.
release-locks:
	$(UV) lock
	$(UV) sync --frozen --all-groups
	$(UV) run python scripts/lock_service_digests.py
	docker compose -f certification/services.lock.yaml up -d
	$(UV) run python scripts/lock_models.py

setup: services-wait
	$(UV) run --frozen python -m kg_construction.interfaces.cli.doctor --strict
	@echo "Mandatory setup complete. Run: make certify"

install: lock-check
	$(UV) sync --frozen --all-groups

services-start: install
	docker compose -f $(CERT_SERVICE_LOCK) up -d

hydrate: services-start
	PYTHONPATH=$(PYTHONPATH) $(UV) run --frozen python scripts/hydrate_models.py $(CERT_PROFILE_ARG)

services-wait: hydrate
	PYTHONPATH=$(PYTHONPATH) $(UV) run --frozen python scripts/wait_for_services.py $(CERT_PROFILE_ARG)

doctor:
	$(UV) run --frozen python -m kg_construction.interfaces.cli.doctor --strict

quickstart:
	$(PYTHON) examples/quickstart_60_second.py

compile:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q src tests

test: compile
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m kg_construction.interfaces.cli.construct \
		--input examples/data_to_kg/sample.txt \
		--source-type text \
		--config configs/data_to_kg_complete.example.yaml \
		--evaluation-suite configs/graph_evaluation_suite.sample_text.yaml \
		--output data/processed/data_to_kg_text_property_graph.json \
		--rdf-output data/processed/data_to_kg_text_rdf.json \
		--evidence-output data/processed/data_to_kg_text_evidence_graph.json \
		--neo4j-output data/processed/data_to_kg_text_neo4j.cypher
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m kg_construction.interfaces.cli.benchmark \
		--input examples/data_to_kg/sample.txt \
		--source-type text \
		--config configs/evaluation.example.yaml \
		--techniques applicable \
		--output data/processed/data_to_kg_technique_evaluation_report.md \
		--json-output data/processed/data_to_kg_technique_evaluation_report.json \
		--csv-output data/processed/data_to_kg_technique_evaluation_report.csv \
		--candidate-output-dir data/processed/data_to_kg_candidates

config-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/check_config_examples.py

notebook-check:
	$(PYTHON) scripts/check_notebook_hygiene.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/check_notebook_coverage.py

registry-reports:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_registry_reports.py

benchmark-leaderboard:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_benchmark_leaderboard.py

gold-benchmark-story:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_gold_standard_benchmark_story.py

cross-source-identity-benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_cross_source_identity_benchmark.py

schema-quality:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_schema_quality_report.py

modelling-correctness:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_modelling_correctness_report.py

capability-manifest:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_capability_manifest.py

package-check:
	rm -rf build dist
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

quality:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m mypy src/kg_construction
	$(PYTHON) -m vulture src tests
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/check_config_examples.py
	$(PYTHON) scripts/check_notebook_hygiene.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/check_notebook_coverage.py

verify: test smoke

# Package/model/container artifacts stay offline; live remote-linking certifiers
# remain network checks and are never replaced by cached or static fallbacks.
certify-preflight: lock-check
	UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(UV) sync --frozen --offline --all-groups
	UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$(PYTHONPATH) \
		$(UV) run --frozen --offline python scripts/hydrate_models.py \
		--offline --validate-only $(CERT_PROFILE_ARG)
	UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$(PYTHONPATH) \
		$(UV) run --frozen --offline python scripts/wait_for_services.py $(CERT_PROFILE_ARG)
	UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$(PYTHONPATH) \
		$(UV) run --frozen --offline python scripts/certify.py \
		--phase preflight $(CERT_PROFILE_ARG)

certify: certify-preflight
	$(MAKE) quality
	$(MAKE) verify
	UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$(PYTHONPATH) \
		$(UV) run --frozen --offline python scripts/certify.py \
		--phase final --execution-plan $(CERT_EXECUTION_PLAN) \
		--report-output $(CERT_EXECUTION_REPORT) $(CERT_PROFILE_ARG)

# Fail fast when any lock is missing or a floating tag slipped in.
lock-check:
	@test -f uv.lock || { echo "missing uv.lock (run: uv lock)"; exit 1; }
	@test -f models.lock.yaml || { echo "missing models.lock.yaml (run: python scripts/lock_models.py)"; exit 1; }
	@test -f certification/services.lock.yaml || { echo "missing certification/services.lock.yaml (run: python scripts/lock_service_digests.py)"; exit 1; }

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache build dist htmlcov
