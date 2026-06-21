.PHONY: install install-ml dev lint fmt typecheck test check data baselines transformer ablate label-efficiency robustness-study interpretability real-data sim2real agent clean

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

transformer:       ## Train SensorPatchTST alongside the baselines (quick-demo)
	python -m scripts.train_baseline --models logreg,xgboost,cnn,lstm,transformer --mode quick_demo --epochs 15

ablate:            ## Run the SensorPatchTST ablation campaign (quick-demo)
	python -m scripts.ablate_transformer --mode quick_demo --epochs 15

label-efficiency:  ## Masked-pretraining label-efficiency sweep (quick-demo; use colab_standard for real)
	python -m scripts.label_efficiency_sweep --mode quick_demo --epochs-pretrain 20 --epochs-finetune 15

robustness-study:  ## Robustness + calibration study (noise/window/missing-channel/domain-shift)
	python -m scripts.robustness_report --mode quick_demo --epochs 10

interpretability:  ## Attention / occlusion / integrated-gradients + faithfulness check
	python -m scripts.interpretability_report --mode quick_demo --epochs 10

real-data:         ## Run the benchmark slate on real C-MAPSS (set RAW_DIR=/path/to/CMAPSSData)
	python -m scripts.fetch_cmapss --raw-dir $(RAW_DIR) --mode quick_demo
	python -m scripts.real_data_report --data data/cmapss_FD001 --mode quick_demo --epochs 10

sim2real:          ## Synthetic->real encoder transfer on C-MAPSS (needs data/cmapss_FD001.npz)
	python -m scripts.sim2real_transfer --data data/cmapss_FD001 --mode quick_demo --epochs-pretrain 20 --epochs-finetune 15

agent:             ## Agentic experiment runner: plan -> run -> review one-variable ablations (quick)
	python -m scripts.run_agent --epochs 6 --seeds 3 --max-experiments 3

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info
