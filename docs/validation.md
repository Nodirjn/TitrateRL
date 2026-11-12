# Simulator validation — decision gate G1

Before any RL training run, the simulator must pass checks V1–V4 by the end of Week 5
(proposal IV-C11). V5 is documentation. Run:

```bash
titraterl validate --out reports           # writes reports/validation.md and .json
titraterl validate --strict                # exit code 1 if G1 fails (for CI)
```

## Current status (first implementation, default cohort: median ρ_SI = 2.2)

| Check | Result | Summary |
|---|---|---|
| V1 | ✅ PASS | RK4 vs SciPy `solve_ivp` 6·10⁻⁸ mg/dL; equilibrium drift 3·10⁻¹⁴; periodic-steady-state change 6·10⁻⁷ mg/dL/day; dose–response monotone; right-hand side identical to simglucose (relative 10⁻⁹) for 3 patients with secretion off |
| V2 | ✅ PASS | glargine: max deviation from the 2–24 h mean 23.7 % (≤ 25 %); 90.3 % of steady state by the 4th dose (≥ 90 %) |
| V3 | documented | 75 g meal, nominal subject: peak 312 mg/dL at 130 min, back within 10 mg/dL after 334 min. To be compared with the Dalla Man 2007 T2D mean curve |
| V4 | ❌ FAIL | Treat-to-Target on the 90 training patients: **final dose 0.57 U/kg (0.4–0.6 ✅)**, week-24 FPG 99.6 mg/dL (110–125 ❌), FPG ≤ 100 mg/dL in 56 % (25–50 % ❌); week-12 FPG 105.5 mg/dL |
| V5 | documented | at 1.5 × the dose that reaches FPG ≤ 100 mg/dL (5 patients): nadir 37–51 mg/dL, time < 54 mg/dL 2–16 % |

**Gate G1: not passed** (V4).

Two further observations for the team:

* With the default cohort, Treat-to-Target produces almost **no hypoglycemia** (TBR<70 ≈ 0 %).
  The safety comparison (H2 non-inferiority) is still meaningful because RL agents can over-dose
  (V5 shows deep lows at 1.5 × the target dose), but the baselines' TBR will be close to zero.
* With the nominal median ρ_SI = 1.0, 36–40 % of patients reached the 1 U/kg dose bound under
  Treat-to-Target, which would make policies indistinguishable for those patients. This is why
  the first calibration pass moved the median to 2.2 (no patient at the bound).

## Calibration log (V4)

Grid over the medians of the cohort factors (30 training patients, Treat-to-Target, 24 weeks):

| median ρ_SI | median ρ_β | week-24 FPG (mg/dL) | final dose (U/kg) | FPG ≤ 100 |
|---|---|---|---|---|
| 1.0 | 1.0 | 107.9 | 0.78 | 37 % |
| 1.0 | 0.5 | 101.7 | 0.68 | 60 % |
| 1.5 | 1.0 | 100.2 | 0.65 | 60 % |
| 2.0 | 1.0 | 99.0 | 0.56 | 60 % |
| 2.0 | 0.5 | 97.2 | 0.44 | 70 % |
| 2.5 | 1.0 | 100.0 | 0.49 | 53 % |
| 2.2 (default, 90 patients) | 1.0 | 99.6 | 0.57 | 56 % |
| trial | | 117 | 0.48 | 36 % |

**Finding.** Scaling insulin sensitivity or beta-cell function moves the final dose into the trial
range, but then the virtual patients reach the fasting target more often than the trial
participants. In the simulator Treat-to-Target almost always drives fasting glucose to ≈ 100 mg/dL;
in the trial it plateaued at 117 mg/dL. This is consistent with the deviations anticipated in the
proposal: virtual patients never miss an injection, and > 80 % of trial participants also took a
sulfonylurea, which is not modeled.

**Next steps (Weeks 4–5, I. Abralov):**

1. verify the nominal T2D parameters against Table I of Dalla Man et al. (2007) (see
   [model.md](model.md));
2. calibrate the spread (σ) as well as the median of ρ_SI, and test a correlation between ρ_SI
   and `Gb`;
3. compare the trajectory of the fasting glucose (week 12 plateau) with the trial curve;
4. report and attribute the remaining deviation; only deviations attributable to physiology are
   used to adjust the cohort.

If V4 still fails at the end of Week 5, the proposal's rule applies: the remaining deviation is
documented, results are interpreted as relative comparisons, and the decision is recorded at the
sprint review. (The fallback model is reserved for V1–V2 failures.)
