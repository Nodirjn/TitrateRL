# Testing strategy and system testing

TitrateRL is tested at five levels. Lower levels are developed test-first (TDD) and run on every
pull request; the system and evaluation levels are run at milestones and before demonstrations.

| Level | What is verified | Tool | When |
|---|---|---|---|
| Unit | Model equations, integrator, metrics, rewards, rules B0–B4, shield, schema | `pytest tests/unit` | every commit (CI) |
| Integration | Environment + policies + evaluator + CLI + dashboard working together | `pytest tests/integration` | every commit (CI); slow tests on `main` |
| Verification and validation | Simulator checks V1–V5 of decision gate G1 | `titraterl validate` | gate G1, after model changes |
| System | The assembled system against the MVP acceptance criteria (F1–F7) | `titraterl system-test` | milestones, before demos |
| Evaluation | Policies on held-out patients with the pre-specified protocol | `titraterl evaluate` | Weeks 10–13 |

## Unit and integration tests

```bash
pytest -m "not slow"      # ~90 tests, < 1 min
pytest                    # including the slow PPO smoke training and dashboard rendering
```

Notable tests: RK4 vs SciPy `solve_ivp` (`test_model.py`); right-hand side identical to
simglucose (`test_cross_implementation.py`); equilibrium, periodic steady state and
dose–response; published titration steps of AT.LANTUS, Treat-to-Target and INSIGHT
(`test_rules.py`); rate limit, dose bounds and shield (`test_env.py`, `test_shield.py`);
hand-computed metrics (`test_metrics.py`); Parquet schema (`test_cohort_logging.py`);
hypothesis analyses (`test_analysis.py`).

## System tests

`titraterl system-test` writes `reports/system_test.md` and `reports/system_test.json`.

| ID | Requirement | Test | Criterion |
|---|---|---|---|
| ST-01 | all | Unit and integration suite | all pass |
| ST-02 | F1 | Cohort regenerated identically from its seed; 90/30/30 split | identical |
| ST-03 | F1 | Gate G1 checks V1, V2 and V4 | all pass |
| ST-04 | F2, F7 | `TitrateRL/T2D-v0` registered; Gymnasium `check_env` | passes |
| ST-05 | F2 | Runtime of one 168-day episode | < 2 s |
| ST-06 | F2 | Same seed → same trajectory | identical |
| ST-07 | F5 | Common random numbers across policies | identical meals |
| ST-08 | F2 | Weekly rate limit and dose bound for every policy incl. trained agents | ≤ 8 U / 7 d, ≤ 1 U/kg |
| ST-09 | F2 | Safety shield: no increase after level-1, −20 % after level-2 hypoglycemia | exact |
| ST-10 | F3 | Trained agents load and complete an episode | all complete |
| ST-11 | F5 | Evaluation pipeline writes all tables, tests and the report | all files |
| ST-12 | F6 | Dashboard re-run: 3 policies × 24 weeks | < 5 s |

The latest results are in [testing/system_test_report.md](testing/system_test_report.md).

## Evaluation

```bash
titraterl evaluate --config configs/eval/test_pilot.yaml   # Week 11 pilot
titraterl evaluate --config configs/eval/test.yaml         # final protocol (Weeks 11-13)
```

Outputs (`results/<name>/`): `episodes.parquet`, `per_patient.csv`, `ci_table.csv`,
`h1.json`, `h2.json`, `h3.json`, `report.md`, `figures/`. The pilot report is committed in
[docs/results/week11_pilot.md](results/week11_pilot.md).
