"""Episode-level outcome metrics (proposal Table VII).

All endpoints are computed from **noise-free** plasma glucose. CGM-based TIR and
TBR are reported as secondary metrics.

Phases (day = 07:00 -> 07:00):
    * titration: weeks 1-12 (days 0-83)
    * maintenance: weeks 21-24 (days 140-167)
    * full: all days
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from titraterl.metrics.glycemic import summarize, time_below, time_in_range
from titraterl.simulation.scenario import clock_to_minute

if TYPE_CHECKING:
    from titraterl.envs.titration_env import DayRecord

PHASES = {
    "titration": (0, 84),
    "maintenance": (140, 168),
    "full": (0, 10_000),
}
NOCTURNAL = (clock_to_minute(0.0), clock_to_minute(6.0))  # 00:00-06:00
BEDTIME_MINUTE = clock_to_minute(22.0)
FASTING_TARGET = (80.0, 130.0)


def _phase_days(records: Sequence[DayRecord], phase: str) -> list[DayRecord]:
    lo, hi = PHASES[phase]
    return [r for r in records if lo <= r.day < hi]


def time_to_target(fasting: Sequence[float], low: float = 80.0, high: float = 130.0) -> float:
    """First day of three consecutive fasting values in [low, high]; NaN if never."""
    run = 0
    for day, g in enumerate(fasting):
        run = run + 1 if low <= g <= high else 0
        if run == 3:
            return float(day - 2)
    return float("nan")


def episode_metrics(records: Sequence[DayRecord], body_weight_kg: float) -> dict[str, float]:
    """Compute every Table VII metric for one simulated titration course."""
    if not records:
        raise ValueError("no day records")
    out: dict[str, float] = {}
    for phase in PHASES:
        days = _phase_days(records, phase)
        if not days:
            continue
        true5 = np.concatenate([r.result.glucose_true_5min for r in days])
        s = summarize(true5)
        for k, v in s.items():
            out[f"{phase}_{k}"] = v
        cgm = np.concatenate([r.result.glucose_cgm for r in days])
        out[f"{phase}_cgm_tir"] = time_in_range(cgm)
        out[f"{phase}_cgm_tbr70"] = time_below(cgm, 70.0)

    # nocturnal metrics over the full period (true glucose, 1-min)
    lo, hi = NOCTURNAL
    noct = np.concatenate([r.result.glucose_true[lo:hi] for r in records])
    out["nocturnal_tir"] = time_in_range(noct)
    out["nocturnal_tbr70"] = time_below(noct, 70.0)

    fasting_true = [r.fasting_true for r in records]
    out["time_to_target_days"] = time_to_target(fasting_true)
    maint = _phase_days(records, "maintenance")
    if maint:
        f = np.array([r.fasting_true for r in maint])
        out["fasting_in_target"] = float(np.mean((f >= FASTING_TARGET[0]) & (f <= FASTING_TARGET[1])))
        diff = [r.result.glucose_true[BEDTIME_MINUTE] - r.fasting_true for r in maint]
        out["bedtime_morning_diff"] = float(np.mean(diff))
    final_dose = records[-1].dose_u
    out["final_dose_u"] = final_dose
    out["final_dose_u_per_kg"] = final_dose / body_weight_kg
    out["overbasalization_diff"] = float(out.get("bedtime_morning_diff", 0.0) >= 50.0)
    out["overbasalization_dose"] = float(final_dose / body_weight_kg > 0.5)
    changes = sum(1 for r in records[1:] if abs(r.applied_u) > 1e-9)
    out["dose_changes_per_week"] = changes / (len(records) / 7.0)

    shield_days = [r for r in records[1:] if r.shield_active]
    out["shield_rate"] = len(shield_days) / max(len(records) - 1, 1)
    spurious = 0
    by_day = {r.day: r for r in records}
    for r in shield_days:
        window = [by_day[d] for d in range(r.day - 3, r.day) if d in by_day]
        if not any(np.min(w.result.glucose_true) < 70.0 for w in window):
            spurious += 1
    out["shield_spurious_share"] = spurious / len(shield_days) if shield_days else 0.0
    return out
