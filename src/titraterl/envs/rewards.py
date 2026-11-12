"""Reward formulations compared in RQ1 (proposal eqs. 11-13).

All rewards are computed from the day's 288 CGM readings, i.e. from measured
values only.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from titraterl.metrics.glycemic import risk, time_below, time_in_range

RewardFn = Callable[[np.ndarray], float]


def reward_tir(cgm: np.ndarray) -> float:
    """R1: daily time in range (70-180 mg/dL), in [0, 1]."""
    return time_in_range(cgm)


def reward_risk(cgm: np.ndarray) -> float:
    """R2: negative mean Kovatchev risk, scaled to [-1, 0]."""
    return -float(np.mean(risk(cgm))) / 100.0


def make_reward_tir_hypo(lambda1: float = 5.0, lambda2: float = 10.0) -> RewardFn:
    """R3: TIR minus explicit penalties on time below 70 and below 54 mg/dL."""

    def reward(cgm: np.ndarray) -> float:
        return time_in_range(cgm) - lambda1 * time_below(cgm, 70.0) - lambda2 * time_below(cgm, 54.0)

    reward.__name__ = f"reward_tir_hypo_l{lambda1:g}_{lambda2:g}"
    return reward


def get_reward(name: str, **kwargs: float) -> RewardFn:
    """Look up a reward by name: ``"r1"``, ``"r2"`` or ``"r3"``."""
    name = name.lower()
    if name in ("r1", "tir"):
        return reward_tir
    if name in ("r2", "risk"):
        return reward_risk
    if name in ("r3", "tir_hypo"):
        return make_reward_tir_hypo(**kwargs)
    raise ValueError(f"unknown reward {name!r}; expected r1, r2 or r3")
