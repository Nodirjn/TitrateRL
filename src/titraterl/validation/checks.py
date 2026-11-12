"""Simulator verification and validation: decision gate G1 (proposal IV-C11).

======  ===============================================================
Check   Criterion
======  ===============================================================
V1      integrator, equilibrium, periodic-steady-state and dose-response
        tests pass (cross-implementation test: ``tests/``)
V2      glargine: after one dose, ``Ra_I`` within +/-25 % of its mean
        between 2 h and 24 h; with daily dosing >= 90 % of the
        steady-state daily exposure by the 4th dose
V3      postprandial response of the nominal subject is reported for
        comparison with the published T2D curves
V4      Treat-to-Target (B2) on the training cohort reproduces the trial:
        week-24 FPG 110-125 mg/dL, final dose 0.4-0.6 U/kg,
        25-50 % of patients with FPG <= 100 mg/dL
V5      hypoglycemia behavior documented: nadir and time < 54 mg/dL at a
        dose 50 % above the dose that reaches the fasting target
======  ===============================================================

Run ``titraterl validate`` to produce ``reports/validation.json`` and a
markdown summary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from titraterl.cohort.generator import Cohort, default_cohort
from titraterl.evaluation.runner import evaluate
from titraterl.policies.rules import TreatToTarget
from titraterl.simulation.model import derivatives, simulate_day
from titraterl.simulation.parameters import (
    MINUTES_PER_DAY,
    NOMINAL_T2D,
    PMOL_PER_UNIT,
    P,
    PatientParameters,
    S,
)
from titraterl.simulation.patient import VirtualPatient
from titraterl.simulation.scenario import empty_day


@dataclass
class CheckResult:
    check: str
    passed: bool | None  # None = documented only (no pass/fail criterion)
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------- V1
def nominal_patient(weight: float = 90.0, gb: float = 196.0) -> PatientParameters:
    return PatientParameters("NOMINAL", weight, gb, nominal=NOMINAL_T2D)


def integrator_error(patient: PatientParameters | None = None, minutes: int = 600) -> float:
    """Max |RK4 - solve_ivp| of plasma glucose (mg/dL) after a basal dose, no meals."""
    patient = patient or nominal_patient()
    p = patient.vector()
    x0 = patient.initial_state()
    x0[S.ISC1] += 20 * PMOL_PER_UNIT / patient.body_weight_kg
    sol = solve_ivp(
        lambda t, x: derivatives(x, p, 0.0, 0.0),
        (0, minutes),
        x0,
        t_eval=np.arange(minutes + 1),
        rtol=1e-9,
        atol=1e-9,
        method="LSODA",
    )
    z = np.zeros(MINUTES_PER_DAY)
    # simulate_day integrates whole days; compare over the first `minutes` minutes
    _, g, _, _ = simulate_day(x0, p, z, z, 0.0, -1, 0.0)
    ref = sol.y[S.GP] / p[P.VG]
    return float(np.max(np.abs(g[:minutes] - ref[:minutes])))


def equilibrium_drift(cohort: Cohort) -> float:
    """Max change of plasma glucose over 24 h without meals or insulin."""
    worst = 0.0
    z = np.zeros(MINUTES_PER_DAY)
    for cp in cohort:
        p = cp.params.vector()
        x0 = cp.params.initial_state()
        _, g, _, _ = simulate_day(x0, p, z, z, 0.0, -1, 0.0)
        worst = max(worst, float(np.max(np.abs(g - cp.params.fpg_preinsulin))))
    return worst


def fasting_after_constant_dose(
    patient: PatientParameters, dose_u: float, days: int = 20, meals: bool = False
) -> np.ndarray:
    """07:00 glucose (noise-free) for every day of a constant daily dose."""
    vp = VirtualPatient(patient, scenario_seed=0, sensor_noise=False, meals_enabled=meals)
    return np.array([vp.step_day(dose_u).fasting_true for _ in range(days)])


def check_v1(cohort: Cohort, n_patients: int = 10) -> CheckResult:
    err = integrator_error()
    drift = equilibrium_drift(cohort)
    pss = []
    mono = True
    for cp in cohort.patients[:n_patients]:
        f = fasting_after_constant_dose(cp.params, 20.0, days=30)
        pss.append(abs(f[-1] - f[-2]))
        prev = np.inf
        for dose in (0, 10, 20, 30, 40, 60):
            v = fasting_after_constant_dose(cp.params, dose, days=30)[-1]
            if v > prev + 1e-6:
                mono = False
            prev = v
    ok = err < 0.5 and drift < 1e-6 and max(pss) < 0.1 and mono
    return CheckResult(
        "V1",
        ok,
        f"integrator error {err:.2e} mg/dL; equilibrium drift {drift:.1e}; "
        f"periodic-steady-state change {max(pss):.2e} mg/dL; dose-response monotone: {mono}",
        {
            "integrator_error": err,
            "equilibrium_drift": drift,
            "pss_change": float(max(pss)),
            "dose_response_monotone": mono,
        },
    )


# ---------------------------------------------------------------------- V2
def glargine_profile(days: int = 6) -> np.ndarray:
    """Plasma appearance ``Ra_I`` (pmol/kg/min) for one dose of 1 pmol/kg at t = 0."""
    n = NOMINAL_T2D
    t = np.arange(days * MINUTES_PER_DAY, dtype=float)
    k = n.kd + n.ka1
    isc1 = np.exp(-k * t)
    isc2 = n.kd / (n.ka2 - k) * (np.exp(-k * t) - np.exp(-n.ka2 * t))
    return n.ka1 * isc1 + n.ka2 * isc2


def check_v2() -> CheckResult:
    ra = glargine_profile()
    window = ra[120:1440]
    dev = float(np.max(np.abs(window / window.mean() - 1.0)))
    cdf = np.cumsum(ra)  # 1-min steps -> integral
    frac_day4 = float(cdf[4 * MINUTES_PER_DAY - 1] / 1.0) if cdf[-1] > 0 else 0.0
    # exposure during day 4 of once-daily dosing equals the fraction of one dose
    # absorbed within 96 h (linear superposition); steady state = 1 dose per day
    ok = dev <= 0.25 and frac_day4 >= 0.90
    return CheckResult(
        "V2",
        ok,
        f"max deviation from mean (2-24 h) {dev:.1%}; day-4 exposure {frac_day4:.1%} of steady state",
        {"max_rel_deviation": dev, "day4_fraction_of_steady_state": frac_day4},
    )


# ---------------------------------------------------------------------- V3
def check_v3() -> CheckResult:
    """Single 75 g meal at minute 60 for the nominal subject (no insulin)."""
    pt = nominal_patient()
    p = pt.vector()
    x0 = pt.initial_state()
    day = empty_day()
    day.cho_mg_min[60:70] = 75_000 / 10
    day.meal_start_mg[60] = 75_000
    _, g, _, _ = simulate_day(x0, p, day.cho_mg_min, day.meal_start_mg, 0.0, -1, 0.0)
    peak = float(g.max())
    t_peak = int(np.argmax(g)) - 60
    back = int(np.argmax(g[60 + t_peak :] <= pt.fpg_preinsulin + 10)) + t_peak
    return CheckResult(
        "V3",
        None,
        f"75 g meal: peak {peak:.0f} mg/dL at {t_peak} min, back within 10 mg/dL of "
        f"basal after {back} min (compare with Dalla Man 2007, T2D mean curve)",
        {
            "peak_mg_dl": peak,
            "time_to_peak_min": t_peak,
            "return_min": back,
            "trace_5min": g[::5].round(1).tolist(),
        },
    )


# ---------------------------------------------------------------------- V4
def check_v4(cohort: Cohort, n_patients: int | None = None) -> CheckResult:
    patients = cohort.split("train")[:n_patients] if n_patients else cohort.split("train")
    df, res = evaluate([TreatToTarget()], cohort, patients, n_scenarios=1, keep_records=True)
    w24 = np.array([np.mean([r.fasting_true for r in e.records[-7:]]) for e in res])
    w12 = np.array([np.mean([r.fasting_true for r in e.records[77:84]]) for e in res])
    dose = float(df["final_dose_u_per_kg"].mean())
    share = float(np.mean(w24 <= 100.0))
    crit = {
        "fpg_week24_110_125": bool(110 <= w24.mean() <= 125),
        "dose_0.4_0.6": bool(0.4 <= dose <= 0.6),
        "share_at_target_25_50": bool(0.25 <= share <= 0.50),
    }
    return CheckResult(
        "V4",
        all(crit.values()),
        f"week-24 FPG {w24.mean():.1f} mg/dL (trial 117), week-12 {w12.mean():.1f}; "
        f"final dose {dose:.2f} U/kg (trial 0.48); FPG <= 100 in {share:.0%} (trial 36 %)",
        {
            "fpg_week24": float(w24.mean()),
            "fpg_week12": float(w12.mean()),
            "final_dose_u_per_kg": dose,
            "share_fpg_le_100": share,
            "criteria": crit,
            "n_patients": len(patients),
        },
    )


# ---------------------------------------------------------------------- V5
def check_v5(cohort: Cohort, n_patients: int = 5) -> CheckResult:
    rows = []
    for cp in cohort.split("train")[:n_patients]:
        target_dose = None
        for dose in range(0, 151, 2):
            if fasting_after_constant_dose(cp.params, dose, days=14, meals=True)[-1] <= 100:
                target_dose = dose
                break
        if target_dose is None:
            rows.append({"patient_id": cp.patient_id, "target_dose_u": None})
            continue
        vp = VirtualPatient(cp.params, scenario_seed=1, sensor_noise=False)
        traces = [vp.step_day(1.5 * target_dose).glucose_true for _ in range(14)]
        g = np.concatenate(traces[-7:])
        rows.append(
            {
                "patient_id": cp.patient_id,
                "target_dose_u": target_dose,
                "nadir_mg_dl": float(g.min()),
                "tbr54": float(np.mean(g < 54)),
            }
        )
    return CheckResult(
        "V5",
        None,
        "nadir and time < 54 mg/dL at 1.5 x the dose that reaches FPG <= 100 mg/dL "
        "(no counter-regulation in the model)",
        {"patients": rows},
    )


def run_all(cohort: Cohort | None = None, quick: bool = False) -> list[CheckResult]:
    """Run V1-V5. ``quick`` uses fewer patients (CI)."""
    cohort = cohort or default_cohort()
    return [
        check_v1(cohort, n_patients=3 if quick else 10),
        check_v2(),
        check_v3(),
        check_v4(cohort, n_patients=10 if quick else None),
        check_v5(cohort, n_patients=2 if quick else 5),
    ]


def gate_g1(results: list[CheckResult]) -> bool:
    """G1 passes when V1-V4 pass (V5 is documentation)."""
    return all(r.passed for r in results if r.check in ("V1", "V2", "V4")) and all(
        r.passed is not False for r in results
    )


def to_markdown(results: list[CheckResult]) -> str:
    lines = ["# Simulator validation (gate G1)", "", "| Check | Result | Summary |", "|---|---|---|"]
    for r in results:
        status = "documented" if r.passed is None else ("PASS" if r.passed else "FAIL")
        lines.append(f"| {r.check} | {status} | {r.summary} |")
    lines += ["", f"**Gate G1: {'PASS' if gate_g1(results) else 'NOT PASSED'}**"]
    return "\n".join(lines)


def as_dicts(results: list[CheckResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]
