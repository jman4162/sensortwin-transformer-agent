.PHONY: install install-ml dev lint fmt test data clean

install:           ## Install core package
	pip install -e .

install-ml:        ## Install with ML extras (torch, sklearn, ...)
	pip install -e ".[ml]"

dev:               ## Install everything for development
	pip install -e ".[ml,dev]"

lint:              ## Lint with ruff
	ruff check .

fmt:               ## Format with black + ruff import sort
	black .
	ruff check --fix .

test:              ## Run the test suite
	pytest

data:              ## Generate the quick-demo synthetic dataset
	python -m scripts.generate_synthetic --config configs/synthetic/base.yaml --mode quick_demo

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info
