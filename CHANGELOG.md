# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the milestone tags.

## [Unreleased] — towards `v0.1-env` (M1)

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
