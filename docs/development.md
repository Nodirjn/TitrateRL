# Development guide

## Process

* Scrum-inspired, two-week sprints. The product backlog lives in GitHub Projects.
* Each sprint: planning → short weekly synchronization → review with the supervisor → retrospective.
* Milestones M1–M4 coincide with sprint reviews and are tagged releases.
* **Definition of done:** the code is merged, its tests pass in CI, it is documented, and any
  experiment it produced is logged with its configuration and commit hash.

## Branches and pull requests (GitHub Flow)

1. Create a short-lived branch from `main`: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`,
   `exp/<topic>` (experiment configs).
2. Write the test first (TDD), then the code.
3. Run `make lint test` locally (pre-commit runs ruff automatically).
4. Open a pull request using the template; CI must pass and one other team member must approve.
5. Squash-merge into `main`. `main` is protected: no direct pushes.

Recommended GitHub settings for `main`: require pull request reviews (1), require status checks
(`lint`, `test`), require branches to be up to date, disallow force pushes.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/):

```
feat(env): add weekly rate limit wrapper
fix(sim): clip negative insulin masses after RK4 step
test(rules): reproduce Treat-to-Target weekly steps
docs(model): add glargine identification
exp(train): add PPO-R3 seed sweep config
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `exp`, `ci`, `chore`.

## Releases

| Tag | Milestone | Content |
|---|---|---|
| `v0.1-env` | M1 (Week 6) | validated simulator and environment, baselines B0–B3 |
| `v0.2-agents` | M2 (Week 8) | PPO R1–R3 trained, reward selected |
| `v0.3-eval` | M3 (Week 12) | evaluation pipeline, test-cohort results, package |
| `v1.0-mvp` | M4 (Week 14) | final MVP release |

Update `CHANGELOG.md` and the version in `pyproject.toml` / `titraterl/__init__.py` with each tag.

## Code ownership

| Area | Owner | Reviewer |
|---|---|---|
| `simulation/`, `cohort/`, `validation/` | I. Abralov | A. Abdusattarov |
| `envs/`, `training/`, rewards, shield | A. Abdusattarov | B. Uchqunov |
| `metrics/`, `evaluation/`, statistics | N. Khabibullaev | I. Abralov |
| packaging, CI, `cli.py`, `dashboard/` | B. Uchqunov | N. Khabibullaev |

Add the members' GitHub handles to `.github/CODEOWNERS` to request reviews automatically.

## Useful commands

```bash
make install        # editable install with dev and dashboard extras
make lint           # ruff check + format check
make test           # fast tests
make test-all       # including slow tests (training smoke run)
make validate       # simulator checks V1-V5
make smoke          # 2 048-step PPO training run
TITRATERL_DISABLE_JIT=1 pytest tests/unit/test_model.py   # debug the model without Numba
```

## Coding conventions

* Python ≥ 3.10, type hints on public functions, Google-style docstrings.
* Units are part of names or docstrings (`dose_u`, `body_weight_kg`, `glucose mg/dL`).
* Randomness only through `numpy.random.Generator` objects passed in explicitly; no global seeds.
* No real patient data in the repository, ever.
