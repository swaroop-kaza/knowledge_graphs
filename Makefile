PYTHON ?= python3
PYTHONPATH ?= src

.PHONY: benchmark-leaderboard clean compile config-check cross-source-identity-benchmark gold-benchmark-story notebook-check package-check quality registry-reports smoke test verify

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
