"""Stateful virtual patient that is simulated one day at a time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from titraterl.simulation.model import simulate_day
from titraterl.simulation.parameters import PMOL_PER_UNIT, PatientParameters, S
from titraterl.simulation.scenario import DayMeals, ScenarioConfig, clock_to_minute, generate_day
from titraterl.simulation.sensors import CGMSensor, smbg_reading

INJECTION_HOUR = 21.0
"""Bedtime injection, as in the Treat-to-Target trial."""


@dataclass
class DayResult:
    """Everything that happened during one simulated day (07:00 -> 07:00)."""

    day: int
    dose_u: float
    glucose_true: np.ndarray  # noise-free plasma glucose, 1-min (1440,)
    glucose_cgm: np.ndarray  # CGM readings, 5-min (288,)
    fasting_true: float  # noise-free plasma glucose at the end of the day (07:00)
    fasting_smbg: float  # fingerstick at the end of the day (07:00)
    meals: DayMeals

    @property
    def glucose_true_5min(self) -> np.ndarray:
        return self.glucose_true[::5]


class VirtualPatient:
    """A virtual T2D patient with its own random streams.

    Args:
        params: Patient parameters.
        scenario_seed: Seed of the scenario. Meals, CGM noise and SMBG noise are
            drawn from three independent streams spawned from this seed, so two
            policies run with the same seed see identical meals and sensor noise
            (common random numbers).
        scenario: Meal scenario configuration.
        sensor_noise: Disable to obtain noise-free measurements (validation).
        meals_enabled: Disable to simulate fasting days (validation).
    """

    def __init__(
        self,
        params: PatientParameters,
        scenario_seed: int = 0,
        scenario: ScenarioConfig | None = None,
        sensor_noise: bool = True,
        meals_enabled: bool = True,
    ) -> None:
        self.params = params
        self.p = params.vector()
        self.scenario = scenario or ScenarioConfig()
        self.meals_enabled = meals_enabled
        ss = np.random.SeedSequence(scenario_seed)
        meal_ss, cgm_ss, smbg_ss = ss.spawn(3)
        self.meal_rng = np.random.default_rng(meal_ss)
        self.smbg_rng = np.random.default_rng(smbg_ss)
        self.cgm = CGMSensor(np.random.default_rng(cgm_ss), enabled=sensor_noise)
        self.sensor_noise = sensor_noise
        self.state = params.initial_state()
        self.d_meal = 0.0
        self.day = 0
        self.injection_minute = clock_to_minute(INJECTION_HOUR)
        self.initial_smbg = smbg_reading(self.fasting_glucose, self.smbg_rng, sensor_noise)

    @property
    def body_weight(self) -> float:
        return self.params.body_weight_kg

    @property
    def fasting_glucose(self) -> float:
        """Current noise-free plasma glucose (mg/dL)."""
        return float(self.state[S.GP] / self.p[1])

    def step_day(self, dose_u: float) -> DayResult:
        """Simulate the next 24 h with a bedtime injection of ``dose_u`` units."""
        meals = generate_day(self.meal_rng, self.scenario)
        if not self.meals_enabled:
            meals = DayMeals(np.zeros_like(meals.cho_mg_min), np.zeros_like(meals.meal_start_mg), [])
        amount = max(dose_u, 0.0) * PMOL_PER_UNIT / self.body_weight
        x, g_true, g_inter, self.d_meal = simulate_day(
            self.state,
            self.p,
            meals.cho_mg_min,
            meals.meal_start_mg,
            self.d_meal,
            self.injection_minute,
            amount,
        )
        self.state = x
        cgm = self.cgm.measure_day(g_inter)
        fasting = self.fasting_glucose
        smbg = smbg_reading(fasting, self.smbg_rng, self.sensor_noise)
        result = DayResult(self.day, float(dose_u), g_true, cgm, fasting, smbg, meals)
        self.day += 1
        return result
