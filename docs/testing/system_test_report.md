# System test report — Week 12

Generated with `titraterl system-test` (Python 3.13, 2 CPU cores) after the full training protocol;
ST-10 now loads the 20 full-protocol agents in `models/full/` (plus one pilot agent kept for the
R3 + shield ablation). ST-03 still fails on purpose: calibration check V4 fails in the main
cohort; the gap is explained by missed injections and the calibrated world passes V4
(see [../validation.md](../validation.md)). Every other acceptance criterion of the MVP is met.

**11 passed, 1 failed, 0 skipped** of 12 system tests.

| ID | Req. | Test | Measured | Criterion | Result | Time (s) |
|---|---|---|---|---|---|---|
| ST-01 | all | Unit and integration test suite | 92 passed, 2 deselected in 7.23s | all tests pass | PASS | 8.41 |
| ST-02 | F1 | Cohort generated reproducibly, 90/30/30 split | 150 patients {'train': 90, 'val': 30, 'test': 30}; regenerated identically: True | identical; sizes | PASS | 0.03 |
| ST-03 | F1 | Simulator checks of gate G1 (V1, V2, V4) | V1 pass, V2 pass, V4 FAIL (week-24 FPG 99.7 mg/dL (trial 117), week-12 105.2; final dose 0.52 U/kg (trial 0.48); FPG <= 100 in 53% (trial 36 %)) | all pass | **FAIL** | 8.28 |
| ST-04 | F2, F7 | Environment registered and passes Gymnasium check | TitrateRL/T2D-v0 registered; check_env passed; obs shape (33,) | check_env passes | PASS | 0.03 |
| ST-05 | F2 | Runtime of one 168-day episode | 168-day episode: mean 0.212 s, worst 0.228 s | < 2 s | PASS | 0.64 |
| ST-06 | F2 | Same seed gives the same trajectory | identical metrics for repeated seed: True | identical | PASS | 0.43 |
| ST-07 | F5 | Common random numbers across policies | identical meals across policies over 14 days: True | identical meals | PASS | 0.04 |
| ST-08 | F2 | Rate limit and dose bound for every policy | 120 episodes (24 policies): max 7-day increase 8.0 U, max dose 1.00 U/kg | <= 8 U / 7 d; <= 1 U/kg | PASS | 37.24 |
| ST-09 | F2 | Safety shield blocks increases and reduces dose | level-1: +4 U -> +0.0 U; level-2: +4 U -> -2.00 U (dose 10.0 U) | 0 U; -20 % | PASS | 0.0 |
| ST-10 | F3 | Trained agents load and run end to end | 21 agents load and complete an episode: ppo_r1_s0 (TIR 53%), ppo_r1_s1 (TIR 56%), ppo_r1_s2 (TIR 56%), ppo_r1_s3 (TIR 56%), ppo_r1_s4 (TIR 56%), ppo_r2_s0 (TIR 56%), ppo_r2_s1 (TIR 56%), ppo_r2_s2 (TIR 56%), ppo_r2_s3 (TIR 51%), ppo_r2_s4 (TIR 51%), ppo_r3_s0 (TIR 51%), ppo_r3_s1 (TIR 54%), ppo_r3_s2 (TIR 53%), ppo_r3_s3 (TIR 53%), ppo_r3_s4 (TIR 52%), ppo_rstar_shield_s0 (TIR 56%), ppo_rstar_shield_s1 (TIR 56%), ppo_rstar_shield_s2 (TIR 56%), ppo_rstar_shield_s3 (TIR 56%), ppo_rstar_shield_s4 (TIR 56%), ppo_r3_shield_s0 (TIR 55%) | all complete | PASS | 6.5 |
| ST-11 | F5 | Evaluation pipeline produces all outputs | pipeline outputs present: True | all files | PASS | 3.15 |
| ST-12 | F6 | Dashboard 24-week re-run for 3 policies | 3 policies x 24 weeks re-run in 0.67 s | < 5 s | PASS | 0.67 |
