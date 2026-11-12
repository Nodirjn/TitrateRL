"""``T2DTitrationEnv``: daily basal-insulin titration as a Gymnasium environment.

One environment step is one simulated day (07:00 -> 07:00); one episode is a
24-week (168-day) titration course (proposal Table V, Fig. 2).

Timeline of an episode::

    reset()  -> simulate day 0 with the initial dose d_0 (10 U by default)
                return o_0 built from day 0
    step(a_t) -> d_{t+1} = clip(d_t + a'_t)  (rate limit and dose bounds)
                 simulate day t+1, return o_{t+1}, r_{t+1}
    ...         after day T-1 the episode is truncated (T - 1 steps in total)

Environment constraints apply to every policy: the cumulative dose increase
over any 7 consecutive days is at most 8 U (largest weekly step of the
Treat-to-Target algorithm), and the dose stays in ``[0, d_max]`` with
``d_max = 1.0 U/kg x BW``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from titraterl.cohort.generator import Cohort, CohortPatient, default_cohort
from titraterl.envs.observation import DayMeasurements, ObservationStack, daily_features
from titraterl.envs.rewards import get_reward
from titraterl.metrics.glycemic import count_events
from titraterl.simulation.patient import DayResult, VirtualPatient
from titraterl.simulation.scenario import ScenarioConfig

DISCRETE_ACTIONS_U = (-4.0, -2.0, 0.0, 2.0, 4.0)
HYPO_THRESHOLDS = (54.0, 56.0, 70.0, 72.0)


@dataclass
class DayRecord:
    """Per-day log entry (daily aggregate; the 5-min trace is kept in ``result``)."""

    day: int
    dose_u: float
    proposed_u: float
    applied_u: float
    shield_active: bool
    reward: float | None
    fasting_smbg: float
    fasting_true: float
    hypo_below: dict[float, bool]
    result: DayResult = field(repr=False)


class T2DTitrationEnv(gym.Env):
    """Daily basal-insulin titration for a virtual adult with Type 2 diabetes.

    Args:
        cohort: Virtual cohort; the default 150-patient cohort when ``None``.
        split: Cohort split episodes are sampled from (``"train"``, ``"val"``,
            ``"test"``) when ``patient_ids`` is not given.
        patient_ids: Explicit list of patients to sample from.
        reward: ``"r1"`` (TIR), ``"r2"`` (risk index) or ``"r3"`` (TIR with
            hypoglycemia penalties).
        reward_kwargs: Extra arguments of the reward (``lambda1``, ``lambda2`` for R3).
        obs_mode: ``"full"`` (33 features) or ``"fpg_only"`` (18 features).
        action_mode: ``"discrete"`` (``Discrete(5)`` = -4, -2, 0, +2, +4 U) or
            ``"units"`` (continuous dose change in U, used by rule-based policies).
        horizon_days: Episode length T in days.
        rate_limit_u: Maximum cumulative increase over any 7 consecutive days.
        max_dose_u_per_kg: Upper dose bound per kg body weight.
        initial_dose_u: Starting dose ``d_0``.
        scenario: Meal scenario configuration.
        sensor_noise: Add CGM and SMBG measurement noise.
        record: Keep the full ``DayRecord`` history (needed for evaluation/logging).
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        cohort: Cohort | None = None,
        split: str = "train",
        patient_ids: list[str] | None = None,
        reward: str = "r1",
        reward_kwargs: dict[str, float] | None = None,
        obs_mode: str = "full",
        action_mode: str = "discrete",
        horizon_days: int = 168,
        rate_limit_u: float = 8.0,
        max_dose_u_per_kg: float = 1.0,
        initial_dose_u: float = 10.0,
        scenario: ScenarioConfig | None = None,
        sensor_noise: bool = True,
        record: bool = False,
    ) -> None:
        super().__init__()
        self.cohort = cohort or default_cohort()
        if patient_ids is not None:
            self.patients = [self.cohort.get(pid) for pid in patient_ids]
        else:
            self.patients = self.cohort.split(split)  # type: ignore[arg-type]
        if not self.patients:
            raise ValueError("no patients to sample from")
        self.reward_name = reward
        self.reward_fn = get_reward(reward, **(reward_kwargs or {}))
        self.obs_mode = obs_mode
        self.action_mode = action_mode
        self.horizon = int(horizon_days)
        self.rate_limit_u = float(rate_limit_u)
        self.max_dose_u_per_kg = float(max_dose_u_per_kg)
        self.default_initial_dose_u = float(initial_dose_u)
        self.scenario = scenario or ScenarioConfig()
        self.sensor_noise = sensor_noise
        self.record = record

        self._stack = ObservationStack(obs_mode)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self._stack.dim,), dtype=np.float32)
        if action_mode == "discrete":
            self.action_space: spaces.Space = spaces.Discrete(len(DISCRETE_ACTIONS_U))
        elif action_mode == "units":
            self.action_space = spaces.Box(-100.0, 100.0, shape=(1,), dtype=np.float32)
        else:
            raise ValueError(f"unknown action_mode {action_mode!r}")

        # episode state
        self.patient: VirtualPatient | None = None
        self.cohort_patient: CohortPatient | None = None
        self.dose_u = 0.0
        self.day = 0
        self.history: list[DayRecord] = []
        self._increases: deque[float] = deque(maxlen=6)
        self._hypo_l1: deque[bool] = deque(maxlen=3)
        self._hypo_l2: deque[bool] = deque(maxlen=1)
        self._events_l1: deque[int] = deque(maxlen=3)
        self._last_applied = 0.0

    # ------------------------------------------------------------------ helpers
    @property
    def d_max(self) -> float:
        assert self.patient is not None
        return self.max_dose_u_per_kg * self.patient.body_weight

    def action_to_units(self, action: Any) -> float:
        """Translate an action of the action space into a dose change (U)."""
        if self.action_mode == "discrete":
            return DISCRETE_ACTIONS_U[int(np.asarray(action).item())]
        return float(np.asarray(action, dtype=float).reshape(-1)[0])

    def hypo_level1_recent(self) -> bool:
        """Level-1 hypoglycemia measured in the last 72 h (SMBG < 70 or CGM event)."""
        return any(self._hypo_l1)

    def hypo_level2_recent(self) -> bool:
        """Level-2 hypoglycemia measured in the last 24 h (SMBG < 54 or CGM event)."""
        return any(self._hypo_l2)

    def _apply_constraints(self, delta: float) -> float:
        """Weekly rate limit and dose bounds; returns the applied change."""
        if delta > 0:
            allowed = max(0.0, self.rate_limit_u - sum(self._increases))
            delta = min(delta, allowed)
        new = float(np.clip(self.dose_u + delta, 0.0, self.d_max))
        return new - self.dose_u

    def _simulate(self) -> DayRecord:
        assert self.patient is not None
        res = self.patient.step_day(self.dose_u)
        hypo_below = {
            thr: bool(res.fasting_smbg < thr or count_events(res.glucose_cgm, thr) > 0)
            for thr in HYPO_THRESHOLDS
        }
        self._hypo_l1.append(hypo_below[70.0])
        self._hypo_l2.append(hypo_below[54.0])
        self._events_l1.append(count_events(res.glucose_cgm, 70.0))
        return DayRecord(
            res.day, self.dose_u, 0.0, 0.0, False, None, res.fasting_smbg, res.fasting_true, hypo_below, res
        )

    def _features(self, rec: DayRecord) -> np.ndarray:
        assert self.patient is not None
        m = DayMeasurements(
            fasting_smbg=rec.fasting_smbg,
            dose_u=self.dose_u,
            applied_change_u=self._last_applied,
            cgm=rec.result.glucose_cgm,
            hypo_events_72h=int(sum(self._events_l1)),
            hypo_indicator=self.hypo_level1_recent(),
            body_weight=self.patient.body_weight,
            day=rec.day,
            horizon=self.horizon,
        )
        return daily_features(m, self.obs_mode)

    def _info(self, rec: DayRecord) -> dict[str, Any]:
        assert self.cohort_patient is not None and self.patient is not None
        return {
            "patient_id": self.cohort_patient.patient_id,
            "day": rec.day,
            "dose_u": rec.dose_u,
            "proposed_u": rec.proposed_u,
            "applied_u": rec.applied_u,
            "shield_active": rec.shield_active,
            "fasting_smbg": rec.fasting_smbg,
            "fasting_true": rec.fasting_true,
            "hypo_below": rec.hypo_below,
            "body_weight": self.patient.body_weight,
            "d_max": self.d_max,
        }

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Start a new titration course.

        Options:
            patient_id: Simulate this patient (otherwise sampled uniformly).
            scenario_seed: Seed of meals and sensor noise (otherwise sampled).
            initial_dose_u: Starting dose (default ``initial_dose_u`` of the env).
        """
        super().reset(seed=seed)
        options = options or {}
        if "patient_id" in options:
            self.cohort_patient = next(p for p in self.patients if p.patient_id == options["patient_id"])
        else:
            self.cohort_patient = self.patients[int(self.np_random.integers(len(self.patients)))]
        scenario_seed = options.get("scenario_seed")
        if scenario_seed is None:
            scenario_seed = int(self.np_random.integers(2**31 - 1))
        self.scenario_seed = int(scenario_seed)
        self.patient = VirtualPatient(
            self.cohort_patient.params, self.scenario_seed, self.scenario, sensor_noise=self.sensor_noise
        )
        self.dose_u = float(
            np.clip(options.get("initial_dose_u", self.default_initial_dose_u), 0.0, self.d_max)
        )
        self._increases.clear()
        self._hypo_l1.clear()
        self._hypo_l2.clear()
        self._events_l1.clear()
        self._last_applied = 0.0
        self.history = []
        rec = self._simulate()
        self.day = rec.day
        if self.record:
            self.history.append(rec)
        obs = self._stack.reset(self._features(rec))
        return obs, self._info(rec)

    def step(self, action: Any):
        return self.step_units(self.action_to_units(action))

    def step_units(self, delta_u: float, *, shield_active: bool = False, proposed_u: float | None = None):
        """Advance one day with a dose change given in units.

        Args:
            delta_u: Requested change of the dose (U), after any shield.
            shield_active: Whether a safety shield modified the action (logged).
            proposed_u: The policy's original proposal (logged); defaults to ``delta_u``.
        """
        if self.patient is None:
            raise RuntimeError("call reset() before step()")
        applied = self._apply_constraints(float(delta_u))
        self._increases.append(max(applied, 0.0))
        self._last_applied = applied
        self.dose_u = float(self.dose_u + applied)
        rec = self._simulate()
        rec.proposed_u = float(delta_u if proposed_u is None else proposed_u)
        rec.applied_u = applied
        rec.shield_active = shield_active
        rec.reward = float(self.reward_fn(rec.result.glucose_cgm))
        self.day = rec.day
        if self.record:
            self.history.append(rec)
        obs = self._stack.push(self._features(rec))
        truncated = self.day >= self.horizon - 1
        return obs, rec.reward, False, truncated, self._info(rec)
