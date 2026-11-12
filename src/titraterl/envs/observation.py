"""Daily feature vectors and the 3-day observation stack (proposal Table V)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from titraterl.metrics.glycemic import (
    coefficient_of_variation,
    mean_glucose,
    time_above,
    time_below,
    time_in_range,
)

STACK_DAYS = 3

FULL_FEATURES = (
    "fasting_smbg",
    "dose_per_kg",
    "applied_change",
    "cgm_tir",
    "cgm_tbr70",
    "cgm_tar180",
    "cgm_mean",
    "cgm_cv",
    "hypo_events_72h",
    "body_weight",
    "time_fraction",
)
FPG_ONLY_FEATURES = (
    "fasting_smbg",
    "dose_per_kg",
    "applied_change",
    "hypo_indicator",
    "body_weight",
    "time_fraction",
)


@dataclass
class DayMeasurements:
    """Measured (not true) information about one day, used by policies."""

    fasting_smbg: float
    dose_u: float
    applied_change_u: float
    cgm: np.ndarray
    hypo_events_72h: int  # CGM level-1 events in the last 72 h
    hypo_indicator: bool  # level-1 hypoglycemia measured in the last 72 h
    body_weight: float
    day: int
    horizon: int


def _scale(v: float, lo: float, hi: float) -> float:
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def daily_features(m: DayMeasurements, mode: str = "full") -> np.ndarray:
    """Feature vector ``f_t`` scaled to [0, 1]."""
    common_head = [
        _scale(m.fasting_smbg, 0.0, 400.0),
        _scale(m.dose_u / m.body_weight, 0.0, 1.0),
        _scale(m.applied_change_u, -10.0, 10.0),
    ]
    common_tail = [_scale(m.body_weight, 50.0, 150.0), _scale(m.day, 0.0, m.horizon)]
    if mode == "fpg_only":
        return np.array([*common_head, float(m.hypo_indicator), *common_tail], dtype=np.float32)
    if mode != "full":
        raise ValueError(f"unknown observation mode {mode!r}")
    cgm = m.cgm
    feats = [
        *common_head,
        time_in_range(cgm),
        time_below(cgm, 70.0),
        time_above(cgm, 180.0),
        _scale(mean_glucose(cgm), 0.0, 400.0),
        float(np.clip(coefficient_of_variation(cgm), 0.0, 1.0)),
        min(m.hypo_events_72h, 5) / 5.0,
        *common_tail,
    ]
    return np.array(feats, dtype=np.float32)


def feature_dim(mode: str) -> int:
    return len(FULL_FEATURES) if mode == "full" else len(FPG_ONLY_FEATURES)


class ObservationStack:
    """Keeps the last three daily feature vectors (oldest first)."""

    def __init__(self, mode: str = "full", days: int = STACK_DAYS) -> None:
        self.mode = mode
        self.days = days
        self._buf: deque[np.ndarray] = deque(maxlen=days)

    @property
    def dim(self) -> int:
        return feature_dim(self.mode) * self.days

    def reset(self, first: np.ndarray) -> np.ndarray:
        self._buf.clear()
        for _ in range(self.days):
            self._buf.append(first)
        return self.observation()

    def push(self, f: np.ndarray) -> np.ndarray:
        self._buf.append(f)
        return self.observation()

    def observation(self) -> np.ndarray:
        return np.concatenate(list(self._buf)).astype(np.float32)
