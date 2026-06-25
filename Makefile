.PHONY: help install test lint run clean

PYTHON  := python
PYTEST  := python -m pytest
RUFF    := python -m ruff

help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:        ## Install all dependencies (runtime + dev)
	pip install -e ".[dev]"

test:           ## Run the full test suite with coverage
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=80

test-fast:      ## Run tests without coverage (faster)
	$(PYTEST) tests/ -v

lint:           ## Run ruff linter across src/ and tests/
	$(RUFF) check src/ tests/ run_pipeline.py

lint-fix:       ## Auto-fix ruff lint issues where possible
	$(RUFF) check --fix src/ tests/ run_pipeline.py

run:            ## Execute the full pipeline with default config
	$(PYTHON) run_pipeline.py

run-custom:     ## Run with custom data path (usage: make run-custom DATA=path/to/file.xlsx)
	$(PYTHON) run_pipeline.py --data $(DATA)

clean:          ## Remove generated outputs (figures, metrics, cached model)
	rm -rf reports/figures/*.png
	rm -f  reports/metrics.json
	rm -f  models/best_model.pkl
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
