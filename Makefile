.PHONY: install lint format test test-all validate smoke cohort evaluate dashboard clean

install:
	pip install -e ".[dev,dashboard]"

lint:
	ruff check src tests dashboard
	ruff format --check src tests dashboard

format:
	ruff check --fix src tests dashboard
	ruff format src tests dashboard

test:
	pytest -m "not slow"

test-all:
	pytest

validate:
	titraterl validate --out reports

smoke:
	titraterl train --config configs/train/smoke.yaml

cohort:
	titraterl cohort --config configs/cohort.yaml --out data/cohort.parquet

evaluate:
	titraterl evaluate --config configs/eval/test.yaml

dashboard:
	streamlit run dashboard/app.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
