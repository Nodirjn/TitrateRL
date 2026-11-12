"""Daily meal scenarios (proposal Section IV-C2).

The environment day runs from 07:00 to 07:00 on the next calendar day, so the
fasting fingerstick (SMBG) taken at the end of a day reflects the night after
that day's injection. Minute 0 of a day is 07:00.

Default day:
    * breakfast 45 g, uniformly between 07:30 and 08:30
    * lunch 70 g, between 12:00 and 13:00
    * dinner 80 g, between 18:30 and 19:30
    * optional 10 g afternoon snack (probability 0.3), between 15:00 and 16:00
    * every meal amount varies by +/-20 % (uniform)

All randomness comes from a ``numpy.random.Generator`` passed in by the caller,
which makes common random numbers (CRN) across policies straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from titraterl.simulation.parameters import MINUTES_PER_DAY

DAY_START_HOUR = 7.0
"""Clock time (h) of minute 0 of an environment day."""


def clock_to_minute(hour: float) -> int:
    """Convert a clock time (h, 0-24) to the minute index of the environment day."""
    return round(((hour - DAY_START_HOUR) % 24.0) * 60.0)


@dataclass(frozen=True)
class MealSpec:
    name: str
    carbs_g: float
    window_start_h: float
    window_end_h: float
    probability: float = 1.0


@dataclass(frozen=True)
class ScenarioConfig:
    """Parameters of the stochastic meal scenario generator."""

    meals: tuple[MealSpec, ...] = (
        MealSpec("breakfast", 45.0, 7.5, 8.5),
        MealSpec("lunch", 70.0, 12.0, 13.0),
        MealSpec("snack", 10.0, 15.0, 16.0, probability=0.3),
        MealSpec("dinner", 80.0, 18.5, 19.5),
    )
    amount_variability: float = 0.20
    timing_jitter_h: float = 0.0
    skip_probability: float = 0.0
    meal_duration_min: int = 10

    @classmethod
    def robustness(cls) -> ScenarioConfig:
        """Wider timing variability and skipped meals (secondary analyses)."""
        return cls(timing_jitter_h=1.0, skip_probability=0.1)


@dataclass
class DayMeals:
    """Meals of one simulated day as minute-resolution input arrays."""

    cho_mg_min: np.ndarray  # ingestion rate per minute (mg/min)
    meal_start_mg: np.ndarray  # meal total at the first minute of each meal
    meals: list[tuple[int, float]] = field(default_factory=list)  # (minute, grams)

    @property
    def carbs_per_5min_g(self) -> np.ndarray:
        """Carbohydrates ingested in each 5-minute interval (g)."""
        return self.cho_mg_min.reshape(-1, 5).sum(axis=1) / 1000.0


def generate_day(rng: np.random.Generator, config: ScenarioConfig | None = None) -> DayMeals:
    """Draw the meals of one day."""
    config = config or ScenarioConfig()
    cho = np.zeros(MINUTES_PER_DAY)
    start = np.zeros(MINUTES_PER_DAY)
    meals: list[tuple[int, float]] = []
    for spec in config.meals:
        # Draw every random number even for skipped meals so that the random
        # stream does not depend on the outcome (keeps CRN aligned).
        u_take, u_time, u_jit, u_amt, u_skip = rng.random(5)
        if u_take >= spec.probability or u_skip < config.skip_probability:
            continue
        hour = spec.window_start_h + u_time * (spec.window_end_h - spec.window_start_h)
        hour += (2.0 * u_jit - 1.0) * config.timing_jitter_h
        grams = spec.carbs_g * (1.0 + (2.0 * u_amt - 1.0) * config.amount_variability)
        minute = clock_to_minute(hour)
        minute = min(max(minute, 0), MINUTES_PER_DAY - config.meal_duration_min)
        dur = config.meal_duration_min
        cho[minute : minute + dur] += grams * 1000.0 / dur
        start[minute] += grams * 1000.0
        meals.append((minute, grams))
    return DayMeals(cho, start, meals)


def empty_day() -> DayMeals:
    """A day without meals (used by validation tests)."""
    return DayMeals(np.zeros(MINUTES_PER_DAY), np.zeros(MINUTES_PER_DAY), [])
