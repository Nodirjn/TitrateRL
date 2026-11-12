"""Glycemic metrics (proposal Table VII).

All functions take glucose values in mg/dL sampled on a regular grid. Time-based
metrics are returned as fractions in [0, 1]; multiply by 100 for percent.

References:
    * Battelino et al., Time-in-range consensus, Diabetes Care 2019.
    * Kovatchev et al., symmetrized blood-glucose risk function (LBGI/HBGI).
    * Bergenstal et al., glucose management indicator (GMI), Diabetes Care 2018.
"""

from __future__ import annotations

import numpy as np

TARGET_LOW = 70.0
TARGET_HIGH = 180.0
LEVEL2_LOW = 54.0
LEVEL2_HIGH = 250.0


def _arr(g: np.ndarray | list[float]) -> np.ndarray:
    a = np.asarray(g, dtype=float)
    if a.size == 0:
        raise ValueError("empty glucose trace")
    return a


def time_in_range(g, low: float = TARGET_LOW, high: float = TARGET_HIGH) -> float:
    """Fraction of samples with ``low <= g <= high`` (TIR, default 70-180 mg/dL)."""
    a = _arr(g)
    return float(np.mean((a >= low) & (a <= high)))


def time_below(g, threshold: float = TARGET_LOW) -> float:
    """Fraction of samples strictly below ``threshold`` (TBR)."""
    return float(np.mean(_arr(g) < threshold))


def time_above(g, threshold: float = TARGET_HIGH) -> float:
    """Fraction of samples strictly above ``threshold`` (TAR)."""
    return float(np.mean(_arr(g) > threshold))


def mean_glucose(g) -> float:
    return float(np.mean(_arr(g)))


def coefficient_of_variation(g) -> float:
    """CV = SD / mean (fraction). Target <= 0.36."""
    a = _arr(g)
    m = a.mean()
    return float(a.std() / m) if m > 0 else 0.0


def gmi(g) -> float:
    """Glucose management indicator (%): 3.31 + 0.02392 * mean glucose (mg/dL)."""
    return 3.31 + 0.02392 * mean_glucose(g)


def _risk_f(g) -> np.ndarray:
    a = np.clip(_arr(g), 1.0, None)
    return 1.509 * (np.log(a) ** 1.084 - 5.381)


def risk(g) -> np.ndarray:
    """Symmetrized blood-glucose risk ``rho(G)`` in [0, 100] for every sample."""
    return 10.0 * _risk_f(g) ** 2


def lbgi(g) -> float:
    """Low blood glucose index."""
    f = _risk_f(g)
    return float(np.mean(np.where(f < 0, 10.0 * f**2, 0.0)))


def hbgi(g) -> float:
    """High blood glucose index."""
    f = _risk_f(g)
    return float(np.mean(np.where(f > 0, 10.0 * f**2, 0.0)))


def risk_index(g) -> float:
    """Mean risk ``rho(G)`` (= LBGI + HBGI)."""
    return float(np.mean(risk(g)))


def count_events(
    g, threshold: float, period_min: float = 5.0, min_duration_min: float = 15.0, below: bool = True
) -> int:
    """Number of CGM hypoglycemic (or hyperglycemic) events.

    An event starts after at least ``min_duration_min`` consecutive minutes
    below the threshold and ends after at least ``min_duration_min`` consecutive
    minutes at or above it (international consensus definition for CGM events).
    """
    a = _arr(g)
    n_req = int(np.ceil(min_duration_min / period_min))
    cond = a < threshold if below else a > threshold
    events = 0
    in_event = False
    run_in = run_out = 0
    for c in cond:
        if c:
            run_in += 1
            run_out = 0
            if not in_event and run_in >= n_req:
                in_event = True
                events += 1
        else:
            run_out += 1
            run_in = 0
            if in_event and run_out >= n_req:
                in_event = False
    return events


def summarize(g, period_min: float = 5.0) -> dict[str, float]:
    """Standard summary of one glucose trace."""
    return {
        "tir": time_in_range(g),
        "tbr70": time_below(g, TARGET_LOW),
        "tbr54": time_below(g, LEVEL2_LOW),
        "tar180": time_above(g, TARGET_HIGH),
        "tar250": time_above(g, LEVEL2_HIGH),
        "mean": mean_glucose(g),
        "cv": coefficient_of_variation(g),
        "gmi": gmi(g),
        "lbgi": lbgi(g),
        "hbgi": hbgi(g),
        "events70": float(count_events(g, TARGET_LOW, period_min)),
        "events54": float(count_events(g, LEVEL2_LOW, period_min)),
    }
