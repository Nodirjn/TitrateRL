# Contributing to TitrateRL

Thank you for helping. This is a four-person course project, so the process is light but strict
about quality and reproducibility.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dashboard]"
pre-commit install
pytest -m "not slow"
```

## Workflow

1. Pick or create an issue in the backlog (GitHub Projects) and assign yourself.
2. Branch from `main`: `feat/…`, `fix/…`, `docs/…`, `exp/…`.
3. **Write the test first**, then the code. Keep pull requests small (< ~400 lines).
4. Run `make lint test`. Commit with Conventional Commits.
5. Open a pull request with the template. One approving review and green CI are required.

## Rules

* Never commit real patient data, credentials or large result files (`data/`, `runs/`,
  `results/` are ignored).
* Every experiment is a YAML file in `configs/`; do not change hyperparameters in code.
* Randomness only through explicit `numpy.random.Generator` objects and seeds.
* Keep units in names and docstrings.
* Changes to the model equations or parameters need a reference in `docs/model.md` and must keep
  checks V1–V2 passing.

See [docs/development.md](docs/development.md) for details.
