# Evaluation protocol and statistics

Source: `src/titraterl/evaluation/`, `src/titraterl/metrics/`.

## Protocol

* Every policy is evaluated on the **30 test patients × 10 scenarios** of 168 days.
* **Common random numbers:** the scenario seed of (patient, scenario k) is
  `SeedSequence([base_seed, patient_number, k])`; meals, CGM noise and SMBG noise are identical for
  every policy.
* All endpoints are computed from **noise-free** plasma glucose; CGM-based TIR/TBR are secondary.
* Per patient, metrics are averaged over the 10 scenarios and then over the 5 training seeds, so
  every patient contributes **one** paired value per comparison. Bootstrap CIs resample patients.

## Metrics (proposal Table VII)

| Metric | Definition | Target / role |
|---|---|---|
| TIR | % time 70–180 mg/dL | > 70 %; primary efficacy (maintenance, weeks 21–24) |
| TBR<70, TBR<54 | % time below 70 / 54 mg/dL | < 4 %, < 1 %; primary safety (full period) |
| TAR>180, TAR>250 | % time above 180 / 250 mg/dL | < 25 %, < 5 % |
| Mean, CV | mean glucose; SD / mean | CV ≤ 36 % |
| GMI | 3.31 + 0.02392 × mean (mg/dL) | HbA1c proxy |
| LBGI, HBGI | Kovatchev low/high blood-glucose indices | risk |
| Time-to-target | first day of 3 consecutive fasting values in 80–130 mg/dL | titration speed |
| Fasting in target | % maintenance days with fasting 80–130 mg/dL | secondary efficacy |
| Nocturnal | TIR and TBR<70 between 00:00 and 06:00 | secondary safety |
| Overbasalization | bedtime (22:00) − morning (07:00) ≥ 50 mg/dL; final dose > 0.5 U/kg | appropriateness of basal dosing |
| Dose burden | final dose (U/kg); dose changes per week | treatment burden |
| Shield statistics | % days with intervention; share without true glucose < 70 in 72 h | safety and sensor-noise effect |

Column names in `episodes.parquet` are `<phase>_<metric>` with phases `titration` (weeks 1–12),
`maintenance` (weeks 21–24) and `full`.

## Hypotheses and decision rules

| | Test | Function |
|---|---|---|
| H1 | R3 vs R1: one-sided paired Wilcoxon on TBR<70 (α = 0.05) **and** median TIR ≤ 5 pp lower | `statistics.evaluate_h1` |
| H2 | Against each of B1–B3: (i) Holm-adjusted one-sided paired Wilcoxon on maintenance TIR, (ii) Hodges–Lehmann gain ≥ 5 pp, (iii) upper 95 % bootstrap CI of the mean paired TBR<70 difference < 1 pp | `statistics.evaluate_h2` |
| H3 | One-sided Spearman correlation between `ρ_SI` and the per-patient TIR difference (expected negative); tertile dose behavior at matched fasting glucose; train–test gap | `statistics.evaluate_h3` |

**Power.** With 30 patients and the Holm level 0.05/3, the one-sided Wilcoxon test has power ≥ 0.8
to detect a true 5 pp gain if the SD of paired differences is ≤ ~8.5 pp. This SD is estimated on
the validation cohort in Week 8; if larger, more test patients are generated from the same
distributions before the test cohort is evaluated.

## Running

```bash
titraterl evaluate --config configs/eval/test.yaml
```

Outputs: `episodes.parquet`, `per_patient.csv`, `summary.csv` (median and IQR per policy),
`h2.json` (if `primary_policy` is set), `figures/*.png`, and optionally 5-minute trajectories
in `trajectories/` (`log_trajectories: true`).
