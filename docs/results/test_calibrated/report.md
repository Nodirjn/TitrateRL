# Evaluation report

Split: **test**, 30 patients x 10 scenarios (common random numbers). Endpoints from noise-free glucose. In silico results - not medical advice.

| Policy | Maintenance TIR % (95% CI) | TBR<70 % (95% CI) | TBR<54 % | Final dose U/kg | Meets targets |
|---|---|---|---|---|---|
| ppo_r1 | 54.6 (50.9-64.3) | 2.4 (1.8-3.4) | 0.2 | 0.59 | 73.3% |
| ppo_rstar_shield | 54.5 (50.6-64.2) | 1.1 (0.8-1.3) | 0.0 | 0.54 | 100.0% |
| ppo_r3 | 54.4 (50.7-62.2) | 0.4 (0.2-1.5) | 0.0 | 0.53 | 83.3% |
| b2_treat_to_target | 54.2 (50.6-63.6) | 0.2 (0.1-0.4) | 0.0 | 0.49 | 100.0% |
| ppo_r2 | 53.8 (50.8-63.5) | 3.1 (1.9-4.5) | 0.4 | 0.58 | 63.3% |
| b3_insight | 53.4 (50.2-61.5) | 0.0 (0.0-0.1) | 0.0 | 0.48 | 100.0% |
| b1_atlantus | 52.6 (49.4-61.1) | 0.0 (0.0-0.0) | 0.0 | 0.46 | 100.0% |
| b0_fixed | 35.1 (21.7-57.8) | 0.0 (0.0-0.0) | 0.0 | 0.20 | 100.0% |

## H2 (RQ2)

| Baseline | HL TIR gain (pp) | p (Holm) | mean TBR diff (pp) | upper CI (pp) | supported |
|---|---|---|---|---|---|
| b1_atlantus | 1.8 | 0.0000 | 1.03 | 1.18 | **False** |
| b2_treat_to_target | -0.1 | 0.6723 | 0.73 | 0.88 | **False** |
| b3_insight | 1.0 | 0.0002 | 0.99 | 1.13 | **False** |

## H3 (RQ3)

ppo_rstar_shield vs b2_treat_to_target: Spearman rho = -0.307, one-sided p = 0.0496; supported: **True**

| SI tertile | n | median TIR difference (pp) |
|---|---|---|
| 1 | 10 | 0.2 |
| 2 | 10 | 0.2 |
| 3 | 10 | -0.9 |
