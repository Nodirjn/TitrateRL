# Evaluation report

Split: **test**, 30 patients x 10 scenarios (common random numbers). Endpoints from noise-free glucose. In silico results - not medical advice.

| Policy | Maintenance TIR % (95% CI) | TBR<70 % (95% CI) | TBR<54 % | Final dose U/kg | Meets targets |
|---|---|---|---|---|---|
| ppo_r1 | 59.3 (54.2-67.1) | 0.5 (0.3-0.8) | 0.0 | 0.63 | 96.7% |
| ppo_r2 | 58.5 (53.4-65.7) | 0.4 (0.2-1.1) | 0.0 | 0.60 | 90.0% |
| ppo_rstar_shield | 58.0 (52.9-66.0) | 0.4 (0.2-0.5) | 0.0 | 0.59 | 100.0% |
| ppo_r3 | 55.3 (52.5-62.4) | 0.0 (0.0-0.0) | 0.0 | 0.54 | 100.0% |
| b2_treat_to_target | 55.1 (51.4-63.0) | 0.0 (0.0-0.0) | 0.0 | 0.51 | 100.0% |
| b3_insight | 54.6 (49.8-62.2) | 0.0 (0.0-0.0) | 0.0 | 0.52 | 100.0% |
| b1_atlantus | 54.1 (49.2-60.8) | 0.0 (0.0-0.0) | 0.0 | 0.51 | 100.0% |
| b0_fixed | 35.0 (21.2-53.2) | 0.0 (0.0-0.0) | 0.0 | 0.20 | 100.0% |

## H1 (RQ1)

ppo_r3 vs ppo_r1: one-sided Wilcoxon p = 0.0000; median TIR loss 4.0 pp; supported: **True**

## H2 (RQ2)

| Baseline | HL TIR gain (pp) | p (Holm) | mean TBR diff (pp) | upper CI (pp) | supported |
|---|---|---|---|---|---|
| b1_atlantus | 3.8 | 0.0000 | 0.38 | 0.48 | **False** |
| b2_treat_to_target | 2.2 | 0.0000 | 0.38 | 0.48 | **False** |
| b3_insight | 2.9 | 0.0000 | 0.38 | 0.48 | **False** |

## H3 (RQ3)

ppo_rstar_shield vs b2_treat_to_target: Spearman rho = -0.181, one-sided p = 0.1688; supported: **False**

| SI tertile | n | median TIR difference (pp) |
|---|---|---|
| 1 | 10 | 2.4 |
| 2 | 10 | 1.4 |
| 3 | 10 | 1.8 |
