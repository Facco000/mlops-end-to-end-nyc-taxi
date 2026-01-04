.PHONY: install lint format test run_etl feast_apply train clean help

PYTHON := python3

help:
	@echo "Available commands:"
	@echo "  make install      - Install project in editable mode with dependencies"
	@echo "  make lint         - Run structural code checks (ruff)"
	@echo "  make format       - Format code (black)"
	@echo "  make test         - Run tests (pytest)"
	@echo "  make run_etl      - Run the ETL pipeline"
	@echo "  make feast_apply  - Apply Feast feature definitions to registry"
	@echo "  make train        - Train the model with default parameters"
	@echo "  make train-tune   - Run hyperparameter tuning then train"
	@echo "  make reports      - Generate visualization reports (plots/maps)"
	@echo "  make clean        - Clean up cache files"

install:
	pip install -e .

lint:
	ruff check src training tests

format:
	black src training tests
	ruff check --fix src training tests

test:
	pytest tests/

run_etl:
	$(PYTHON) -m src.etl.run_etl

feast_apply:
	cd feature_repo/taxi_features && feast apply

train:
	$(PYTHON) training/train.py

train-tune:
	$(PYTHON) training/train.py --tune

reports:
	$(PYTHON) reports/figures/maps/script.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "mlruns" -exec rm -rf {} +
