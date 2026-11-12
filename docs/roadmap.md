# Roadmap and milestones

Proposal Table XI. Week 12 is a schedule buffer.

| Weeks | Target deliverable | Responsible | Status |
|---|---|---|---|
| 1–3 | Requirements and architecture; repository, CI and packaging skeleton; T2D parameters extracted; patient model adapted from simglucose and extended with the T2D blocks; glargine model; integrator and cross-implementation tests | N. Khabibullaev (requirements); B. Uchqunov (architecture, CI, packaging); I. Abralov (model, tests); A. Abdusattarov (MDP spec, SB3 prototype) | ✅ implemented |
| 4–6 | Cohort generator and calibration (Weeks 4–5); environment with observation, reward, rate limit and shield; B0–B3; validation checks; **gate G1**; **M1 `v0.1-env`** | I. Abralov; A. Abdusattarov; B. Uchqunov; N. Khabibullaev | 🟡 code done; V4 calibration open |
| 7–8 | PPO with R1–R3 (shield off, 5 seeds); reward selection and power check (**gate G2**); start shielded PPO-R★; **M2 `v0.2-agents`** | A. Abdusattarov; N. Khabibullaev; B. Uchqunov | ⏳ |
| 9–10 | Evaluation pipeline with CRN, statistics, dashboard v1; shielded PPO-R★ (5 seeds); Should-tier runs; RQ1 test-cohort evaluation (**O1**) | N. Khabibullaev; A. Abdusattarov; I. Abralov; B. Uchqunov | 🟡 pipeline and dashboard implemented |
| 11–12 | Full test-cohort evaluation (**O2**, end of Week 11); packaging and dashboard finalization; buffer; **M3 `v0.3-eval`** | N. Khabibullaev; B. Uchqunov; A. Abdusattarov, I. Abralov | ⏳ |
| 13 | RQ3 analyses (correlation, dose behavior, generalization gap) (**O3**); Could tier only if M3 on time | N. Khabibullaev, A. Abdusattarov; I. Abralov; B. Uchqunov | ⏳ |
| 14 | Final report, README, user guide, model card, presentation; **M4 `v1.0-mvp`** | all | ⏳ |

## Scope tiers (proposal Table X)

| Tier | Content | Gate / fallback |
|---|---|---|
| **Core (MVP)** | F1, F2, F4 (B0–B3), F5–F7; PPO with R1–R3 and shielded PPO with R★, five seeds each; RQ1–RQ3 analyses | G1 (Week 5): V1–V4 pass, otherwise fallback model. G2 (Week 8): if no PPO configuration meets the TBR targets on validation, tuning stops and RQ2 is answered with the best available policy as a negative result |
| **Should** | DQN; PPO-FPG-only; λ grid for R3; B4; robustness scenarios | dropped in reverse order if M2 is > 1 week late |
| **Could** | oracle fine-tuning; 2013 hypoglycemia extensions; LSTM policy; SAC | only if M3 is on time |

## Risks (proposal Table XII)

| Risk | L / I | Mitigation |
|---|---|---|
| Errors in the re-implemented T2D model | M / H | TDD; cross-implementation test against simglucose (in place); G1 with fallback |
| Cohort cannot be calibrated to Treat-to-Target | M / M | deviations reported and attributed; relative comparisons — **currently materialized, see [validation.md](validation.md)** |
| Model exaggerates hypoglycemia | H / M | comparative interpretation; V5; Could-tier 2013 model |
| RL does not converge / unsafe policies | M / H | reward normalization; shield; validation-based checkpoints; G2 |
| Simulation too slow | L / M | Numba (≈ 0.15 s per 168-day episode, measured); vectorized environments |
| Over-fitting to the simulator | H / M | held-out patients, scenario variability, noise, robustness set; no clinical claims |
| Integration problems | M / M | fixed interfaces (`Policy`, Gymnasium API, log schema); integration tests in CI |
| Schedule slip | M / M | scope tiers and gates; Week 12 buffer |
