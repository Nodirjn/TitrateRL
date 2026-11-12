"""Rule-based titration policies B0-B4 (proposal Section IV-C6).

All rules use measured values only: the fasting fingerstick (SMBG) taken at
07:00 and CGM hypoglycemic events. A "hypoglycemic reading below X" is a fasting
SMBG value below X or a CGM event below X (>= 15 consecutive minutes).

Rule-based policies are evaluated as published, with their own hypoglycemia
rules and without the safety shield.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from titraterl.policies.base import Policy


def _hypo(info: dict[str, Any], threshold: float) -> bool:
    return bool(info["hypo_below"][float(threshold)])


class FixedDose(Policy):
    """B0: constant 0.2 U/kg/day, no titration (upper end of the ADA start range)."""

    policy_id = "b0_fixed"

    def __init__(self, units_per_kg: float = 0.2) -> None:
        self.units_per_kg = units_per_kg

    def initial_dose(self, body_weight_kg: float) -> float:
        return self.units_per_kg * body_weight_kg

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> float:
        return 0.0


class _IntervalRule(Policy):
    """Shared logic: periodic up-titration with hypoglycemia handling.

    Args:
        interval_days: Days between titration decisions.
        window_days: Number of most recent fasting values averaged at a decision.
        target: Fasting glucose target (mg/dL); 100 by default, 130 for the ADA
            upper-goal sensitivity analysis.
        reduce_below_70: Reduce the dose by ``reduction`` after a reading < 70.
        reduction: Relative reduction (10 %, lower end of the ADA recommendation).
        block_after_72: No increase after a reading < 72 mg/dL since the last adjustment.
    """

    def __init__(
        self,
        interval_days: int,
        window_days: int,
        target: float = 100.0,
        reduce_below_70: bool = True,
        reduction: float = 0.10,
        block_after_72: bool = True,
    ) -> None:
        self.block_after_72 = block_after_72
        self.interval_days = interval_days
        self.window_days = window_days
        self.target = target
        self.reduce_below_70 = reduce_below_70
        self.reduction = reduction
        self.reset({})

    def reset(self, info: dict[str, Any]) -> None:
        self._fasting: deque[float] = deque(maxlen=max(self.window_days, 1))
        self._days_since = 0
        self._hypo72_since = False
        if info:
            self._fasting.append(float(info["fasting_smbg"]))
            self._hypo72_since = _hypo(info, 72)

    def _increase(self, mean_fasting: float) -> float:
        raise NotImplementedError

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> float:
        dose = float(info["dose_u"])
        self._fasting.append(float(info["fasting_smbg"]))
        self._days_since += 1
        self._hypo72_since = self._hypo72_since or _hypo(info, 72)
        if self.reduce_below_70 and _hypo(info, 70):
            self._days_since = 0
            self._hypo72_since = False
            return -self.reduction * dose
        if self._days_since < self.interval_days:
            return 0.0
        self._days_since = 0
        blocked = self._hypo72_since and self.block_after_72
        self._hypo72_since = False
        if blocked:
            return 0.0
        return self._increase(float(np.mean(self._fasting)))


class ATLantus(_IntervalRule):
    """B1: AT.LANTUS Algorithm 2 (patient-led, Davies et al., 2005).

    Every 3 days: +2 U if the mean fasting SMBG of the previous 3 days exceeds
    the target (100 mg/dL); the trial's discretionary 0-2 U band for 100-119
    mg/dL is implemented as +2 U. No increase after a reading < 72 mg/dL since
    the last adjustment; -10 % after a reading < 70 mg/dL.
    Documented deviation: the trial started at a dose equal to the highest
    fasting glucose in mmol/L; B1 starts at 10 U.
    """

    policy_id = "b1_atlantus"

    def __init__(self, target: float = 100.0) -> None:
        super().__init__(interval_days=3, window_days=3, target=target)

    def _increase(self, mean_fasting: float) -> float:
        return 2.0 if mean_fasting > self.target else 0.0


class TreatToTarget(Policy):
    """B2: weekly forced titration of the Treat-to-Target trial (Riddle et al., 2003).

    Every 7 days, from the mean fasting SMBG of the previous 2 days:
    >= 180: +8 U; 140-180: +6 U; 120-140: +4 U; 100-120: +2 U (target <= 100).
    No increase if a reading < 72 mg/dL occurred in the preceding week; -4 U
    (upper end of the trial's 2-4 U range) if a reading < 56 mg/dL occurred.
    """

    policy_id = "b2_treat_to_target"

    def __init__(self, target: float = 100.0) -> None:
        self.target = target
        self.reset({})

    def reset(self, info: dict[str, Any]) -> None:
        self._fasting: deque[float] = deque(maxlen=2)
        self._days_since = 0
        self._hypo72 = False
        self._hypo56 = False
        if info:
            self._fasting.append(float(info["fasting_smbg"]))

    def increase_for(self, mean_fasting: float) -> float:
        if mean_fasting >= 180:
            return 8.0
        if mean_fasting >= 140:
            return 6.0
        if mean_fasting >= 120:
            return 4.0
        if mean_fasting > self.target:
            return 2.0
        return 0.0

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> float:
        self._fasting.append(float(info["fasting_smbg"]))
        self._hypo72 = self._hypo72 or _hypo(info, 72)
        self._hypo56 = self._hypo56 or _hypo(info, 56)
        self._days_since += 1
        if self._days_since < 7:
            return 0.0
        self._days_since = 0
        hypo72, hypo56 = self._hypo72, self._hypo56
        self._hypo72 = self._hypo56 = False
        if hypo56:
            return -4.0
        if hypo72:
            return 0.0
        return self.increase_for(float(np.mean(self._fasting)))


class Insight(_IntervalRule):
    """B3: daily self-titration of the INSIGHT study (Gerstein et al., 2006).

    +1 U per day while the fasting SMBG exceeds 100 mg/dL (5.5 mmol/L).
    INSIGHT specified no dose-reduction rule, so B3 uses the ADA reduction
    (-10 % after a reading < 70 mg/dL).
    """

    policy_id = "b3_insight"

    def __init__(self, target: float = 100.0) -> None:
        super().__init__(interval_days=1, window_days=1, target=target, block_after_72=False)

    def _increase(self, mean_fasting: float) -> float:
        return 1.0 if mean_fasting > self.target else 0.0


@dataclass(frozen=True)
class ThresholdRuleParams:
    threshold: float = 110.0
    step_u: float = 2.0
    interval_days: int = 3


class ThresholdRule(_IntervalRule):
    """B4: simulator-tuned threshold rule (Should tier).

    Increase by ``step_u`` every ``interval_days`` if the mean fasting SMBG over
    the interval exceeds ``threshold``; same hypoglycemia rules as B1. The 36
    combinations of threshold {100, 110, 120, 130}, step {1, 2, 4} and interval
    {1, 3, 7} are tuned on the training cohort and selected on validation.
    B4 is not a clinical algorithm.
    """

    def __init__(self, params: ThresholdRuleParams | None = None) -> None:
        self.params = params or ThresholdRuleParams()
        p = self.params
        self.policy_id = f"b4_thr{p.threshold:g}_s{p.step_u:g}_i{p.interval_days}"
        super().__init__(interval_days=p.interval_days, window_days=p.interval_days, target=p.threshold)

    def _increase(self, mean_fasting: float) -> float:
        return self.params.step_u if mean_fasting > self.target else 0.0

    @staticmethod
    def grid() -> list[ThresholdRuleParams]:
        return [
            ThresholdRuleParams(t, s, i)
            for t in (100.0, 110.0, 120.0, 130.0)
            for s in (1.0, 2.0, 4.0)
            for i in (1, 3, 7)
        ]


def clinical_baselines(target: float = 100.0) -> list[Policy]:
    """B1-B3, the H2 test family."""
    return [ATLantus(target), TreatToTarget(target), Insight(target)]


def all_rule_policies() -> list[Policy]:
    return [FixedDose(), *clinical_baselines(), ThresholdRule()]
