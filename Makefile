BASE_PYTHON ?= python3
VENV ?= .venv

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
.PHONY: benchmark-leaderboard clean compile config-check cross-source-identity-benchmark doctor gold-benchmark-story help install install-dev notebook-check package-check quality quickstart registry-reports setup smoke test venv verify

help:
	@echo "kg-construction - common tasks"
	@echo ""
	@echo "  make setup         Create .venv, install the package, and run a health check"
	@echo "  make install       Install the package (core, offline) into .venv"
	@echo "  make install-dev   Install with dev, rdf, graph, and notebook extras"
	@echo "  make doctor        Diagnose the install and optional services"
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

setup: install
	@$(PYTHON) -m kg_construction.interfaces.cli.doctor --skip-services || true
	@echo ""
	@echo "Setup complete. Next:  make quickstart"
	@echo "To use the kg-construction command directly:  source $(VENV)/bin/activate"

install: venv
	$(PYTHON) -m pip install -e .

install-dev: venv
	$(PYTHON) -m pip install -e ".[dev,rdf,graph,notebook]"

doctor:
	$(PYTHON) -m kg_construction.interfaces.cli.doctor

quickstart:
	$(PYTHON) examples/quickstart_60_second.py

compile:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q src tests

test: compile
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m kg_construction.interfaces.cli.construct \
		--input examples/data_to_kg/sample.txt \
		--source-type text \
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

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache build dist htmlcov
