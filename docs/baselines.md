# Rule-based policies (B0–B4)

Source: `src/titraterl/policies/rules.py`. All rules use measured values only and are evaluated
as published, with their own hypoglycemia rules and **without** the safety shield.

A *hypoglycemic reading below X* is a fasting SMBG value below X or a CGM event below X
(≥ 15 consecutive minutes).

| ID | Class | Start | Titration | Hypoglycemia rules |
|---|---|---|---|---|
| B0 | `FixedDose` | 0.2 U/kg | none | none |
| B1 | `ATLantus` | 10 U | every 3 days: +2 U if 3-day mean fasting > 100 mg/dL | no increase after < 72 since last adjustment; −10 % after < 70 |
| B2 | `TreatToTarget` | 10 U | every 7 days, mean of last 2 fasting values: ≥ 180 → +8; 140–180 → +6; 120–140 → +4; 100–120 → +2 | no increase after < 72 in the preceding week; −4 U after < 56 |
| B3 | `Insight` | 10 U | daily: +1 U if fasting > 100 mg/dL | −10 % after < 70 (ADA; INSIGHT had no reduction rule) |
| B4 | `ThresholdRule` | 10 U | best of 36: threshold {100, 110, 120, 130} × step {1, 2, 4} U × interval {1, 3, 7} days | as B1 |

## Documented deviations from the trials

* **B1:** the trial's discretionary 0–2 U band for 100–119 mg/dL is implemented as +2 U; the trial
  started insulin-naive subjects at a dose (U) equal to their highest fasting glucose in mmol/L,
  whereas B1 starts at 10 U.
* **B2:** the trial's 2–4 U decrease is implemented as −4 U (upper end).
* **B3:** INSIGHT specified no dose-reduction rule, so the ADA 10 % reduction is used.
* **Sensitivity analysis:** B1–B3 accept `target=130` (ADA upper fasting goal).

## Tuning B4

```bash
titraterl tune-b4 --out results/b4_tuning.csv
```

The 36 rules are evaluated on the training cohort and the one with the highest median validation
TIR among those meeting the consensus hypoglycemia targets is selected. B4 is not a clinical
algorithm; it separates the benefit of *learning* from simulator-specific tuning and is excluded
from the H2 test family.
