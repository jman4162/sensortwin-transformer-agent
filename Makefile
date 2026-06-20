.PHONY: install install-ml dev lint fmt typecheck test check data baselines clean

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

typecheck:         ## Static type check the library + scripts
	mypy sensortwin scripts

test:              ## Run the test suite
	pytest

check: lint typecheck test   ## Run all required gates (ruff + mypy + pytest)

data:              ## Generate the quick-demo synthetic dataset
	python -m scripts.generate_synthetic --config configs/synthetic/base.yaml --mode quick_demo

baselines:         ## Train + evaluate the v0.2 baselines (quick-demo) and write the report
	python -m scripts.train_baseline --mode quick_demo --epochs 10

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info
