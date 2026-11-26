# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the milestone tags.

## [Unreleased] — Week 12: full protocol and test-cohort evaluation

### Added
- `scripts/run_protocol.py`: resumable training queue for phases 1–2 (20 PPO runs, 5 seeds).
- `scripts/select_reward.py`: pre-registered R★ rule over seeds; R★ = R1 confirmed.
- 20 selected full-protocol checkpoints (`models/full/`).
- Full test-cohort evaluation (`configs/eval/test_full.yaml`, `docs/results/full_protocol.md`):
  H1 supported (TIR loss 4.0 pp), H2 not supported (gains 2.2–3.8 pp < 5 pp, TBR non-inferior),
  H3 not supported.
- Missed-injection (non-adherence) scenario option and V4 sensitivity analysis
  (`scripts/v4_sensitivity.py`, `docs/validation.md`); calibrated-world evaluation
  (`configs/eval/test_calibrated.yaml`).
- Seed pooling in the evaluation pipeline (`pool_seeds`).

## Week 11: System Testing and Evaluation

### Added
- `titraterl system-test`: 12 end-to-end system tests against the MVP acceptance criteria
  (F1–F7), with a Markdown/JSON report.
- Evaluation pipeline: 95 % bootstrap CI table, H1 and H3 analyses, `report.md`, parallel
  evaluation (`n_jobs`) and `--analyze-only`.
- Five pilot PPO agents (`models/pilot/`), pilot evaluation config and report
  (`docs/results/week11_pilot.md`); R★ = R1 selected on validation patients.
- Dashboard lists the versioned pilot agents; `DEMO.md`, `docs/testing.md`.

### Fixed
- JSON export of numpy booleans in statistics outputs; H3 reference excludes the primary policy.

## Weeks 1–10 — towards `v0.1-env` (M1)

### Added
- Dalla Man T2D patient model (15 states) with endogenous secretion and time-varying hepatic
  extraction; fixed-step RK4 integrator compiled with Numba.
- Two-compartment glargine absorption identified to satisfy check V2.
- Meal scenario generator, Breton–Kovatchev CGM noise, ISO-style SMBG noise.
- Virtual cohort generator (150 patients, stratified 90/30/30 split, Parquet I/O).
- `T2DTitrationEnv` (`TitrateRL/T2D-v0`) with weekly rate limit, dose bounds, 3-day observation
  stack, rewards R1–R3; `SafetyShield` wrapper.
- Rule-based policies B0–B4 and a Stable-Baselines3 adapter behind a common `Policy` interface.
- PPO/DQN training with validation-based checkpoint selection; YAML configs for every Core and
  Should-tier run.
- CRN evaluator, Table VII metrics, H1–H3 statistics, Parquet logs with schema validation,
  evaluation pipeline and figures.
- Simulator validation checks V1–V5 (`titraterl validate`).
- Streamlit dashboard, `titraterl` CLI, CI workflow, documentation and model card.

### Known issues
- Check V4 (calibration against Treat-to-Target) not yet passed: final dose 0.57 U/kg is in range,
  but week-24 FPG is 99.6 mg/dL (trial 117) and 56 % reach FPG ≤ 100 mg/dL (trial 36 %).
