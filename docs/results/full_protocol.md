# Full-protocol results (Week 12)

In silico results on virtual patients; not medical advice and not for dosing real people.

## What was run

| Phase | Agents | Seeds | Training | Selection |
|---|---|---|---|---|
| 1 (RQ1) | PPO with R1, R2, R3, shield off | 0–4 | 10⁶ transitions (days) per run | best checkpoint on the 30 validation patients |
| 2 (RQ2) | PPO with R★ = R1 and the safety shield | 0–4 | 10⁶ transitions per run | same |

20 runs in total (about 30–50 min each on one CPU core). The selected checkpoints are in
`models/full/`.

**Reward selection (pre-registered rule, median over 5 seeds of the validation scores;
[`reward_selection_full.json`](reward_selection_full.json)):**

| Reward | Median validation TIR | Share of patients meeting the hypoglycemia targets |
|---|---|---|
| R1 | 60.7 % | 97 % |
| R2 | 60.5 % | 93 % |
| R3 | 59.5 % | 100 % |

All three pass the ≥ 90 % threshold, and R1 has the highest TIR, so **R★ = R1** (as in the pilot).

## Main evaluation: 30 test patients × 10 scenarios, CRN

Config `configs/eval/test_full.yaml`; full outputs in [`test_full/`](test_full/report.md).
Per-patient metrics of the PPO configurations are averaged over the 5 training seeds.

| Policy | Maintenance TIR % (95 % CI) | TBR<70 % | TBR<54 % | Final dose U/kg | Meets targets |
|---|---|---|---|---|---|
| PPO-R1 | 59.3 (54.2–67.1) | 0.5 | 0.0 | 0.63 | 97 % |
| PPO-R2 | 58.5 (53.4–65.7) | 0.4 | 0.0 | 0.60 | 90 % |
| **PPO-R★ + shield (primary)** | **58.0 (52.9–66.0)** | **0.4** | **0.0** | **0.59** | **100 %** |
| PPO-R3 | 55.3 (52.5–62.4) | 0.0 | 0.0 | 0.54 | 100 % |
| B2 Treat-to-Target | 55.1 (51.4–63.0) | 0.0 | 0.0 | 0.51 | 100 % |
| B3 INSIGHT | 54.6 (49.8–62.2) | 0.0 | 0.0 | 0.52 | 100 % |
| B1 ATLANTUS | 54.1 (49.2–60.8) | 0.0 | 0.0 | 0.51 | 100 % |
| B0 fixed dose | 35.0 (21.2–53.2) | 0.0 | 0.0 | 0.20 | 100 % |

Training seeds agree closely: the primary policy's per-seed median TIR is 57.6–58.2 %.

### Hypotheses

| | Pre-registered criterion | Result | Verdict |
|---|---|---|---|
| **H1** (RQ1) | R3 reduces TBR vs R1 at a median TIR loss ≤ 5 pp | TBR 0.0 % vs 0.5 %, one-sided Wilcoxon p < 0.001; TIR loss 4.0 pp | **supported** |
| **H2** (RQ2) | primary gains ≥ 5 pp TIR over each of B1–B3 and is TBR non-inferior (margin 1 pp) | HL gains 3.8 / 2.2 / 2.9 pp (all p < 0.001, Holm); TBR upper CI 0.48 pp < 1 pp | **not supported** (gain significant but below 5 pp; safety non-inferiority met) |
| **H3** (RQ3) | the TIR gain over the best clinical rule (B2) decreases with insulin sensitivity | Spearman ρ = −0.18, one-sided p = 0.17 | **not supported** |

Compared with the pilot (1 seed, 3×10⁵ days), H1 moved from a 5.1 pp loss (not supported) to a
4.0 pp loss (supported). H2 and H3 conclusions are unchanged.

## Sensitivity evaluation: V4-calibrated world

Config `configs/eval/test_calibrated.yaml`: test patients regenerated with median ρ_SI 3.0 and
15 % missed injections (the setting that passes V4, see [validation.md](../validation.md)).
Agents were **not** retrained. Full outputs in [`test_calibrated/`](test_calibrated/report.md).

| Policy | Maintenance TIR % | TBR<70 % | TBR<54 % | Meets targets |
|---|---|---|---|---|
| PPO-R1 | 54.6 | 2.4 | 0.2 | 73 % |
| **PPO-R★ + shield** | **54.5** | **1.1** | **0.0** | **100 %** |
| PPO-R3 | 54.4 | 0.4 | 0.0 | 83 % |
| B2 Treat-to-Target | 54.2 | 0.2 | 0.0 | 100 % |
| PPO-R2 | 53.8 | 3.1 | 0.4 | 63 % |
| B3 INSIGHT | 53.4 | 0.0 | 0.0 | 100 % |
| B1 ATLANTUS | 52.6 | 0.0 | 0.0 | 100 % |
| B0 fixed dose | 35.1 | 0.0 | 0.0 | 100 % |

- The TIR advantage of RL over the clinical rules almost disappears (gain over B2: −0.1 pp, p = 0.67).
- Without the shield, the agents become less safe when patients are more insulin-sensitive and
  miss injections: only 63–83 % of patients meet the hypoglycemia targets with R1–R3. **With the
  shield all patients still meet them**, although TBR rises to 1.1 % and the H2 non-inferiority
  margin (upper CI 0.88 pp vs B2) is close to the 1 pp limit.
- H3 reaches ρ = −0.31, one-sided p = 0.0496 here. This is a secondary sensitivity analysis, not
  the pre-registered test, so it is reported as exploratory.

## Interpretation

1. RL agents improve TIR over the guideline rules by about 2–4 pp in the main cohort, with low
   hypoglycemia. The gain is statistically clear but smaller than the 5 pp clinically meaningful
   threshold set in advance.
2. The reward design behaves as hypothesized: R3 trades about 4 pp TIR for near-zero hypoglycemia.
3. The safety shield is what keeps the agents safe outside the training distribution
   (non-adherence, higher sensitivity). This supports shielded RL rather than unconstrained RL.
4. Results depend on the virtual cohort; the calibrated world shows that conclusions about the
   size of the benefit are sensitive to patient adherence and insulin sensitivity.

Next (Week 13, O3): dose-behavior analysis, generalization gap (validation vs test), per-tertile
plots, and the paper's Results section.
