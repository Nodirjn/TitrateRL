"""Missed-injection (non-adherence) scenario: V4 sensitivity and robustness analyses."""

import numpy as np

from titraterl.cohort import default_cohort
from titraterl.simulation import ScenarioConfig
from titraterl.simulation.patient import VirtualPatient


def _days(scenario, n=14, dose=20.0, seed=5):
    vp = VirtualPatient(default_cohort().patients[0].params, seed, scenario)
    return [vp.step_day(dose) for _ in range(n)]


def test_default_injects_every_day():
    assert all(d.injected for d in _days(ScenarioConfig()))


def test_missed_injections_keep_meals_and_sensor_noise():
    base = _days(ScenarioConfig())
    miss = _days(ScenarioConfig.nonadherent(0.3))
    for a, b in zip(base, miss, strict=True):
        np.testing.assert_array_equal(a.meals.cho_mg_min, b.meals.cho_mg_min)
    # before the first missed injection the trajectories are identical (same CGM noise)
    first_miss = next(i for i, d in enumerate(miss) if not d.injected)
    for a, b in zip(base[:first_miss], miss[:first_miss], strict=True):
        np.testing.assert_array_equal(a.glucose_cgm, b.glucose_cgm)


def test_missed_rate_and_effect():
    days = _days(ScenarioConfig.nonadherent(0.25), n=200)
    rate = np.mean([not d.injected for d in days])
    assert 0.15 < rate < 0.35
    never = _days(ScenarioConfig.nonadherent(1.0), n=10)
    always = _days(ScenarioConfig(), n=10)
    assert never[-1].fasting_true > always[-1].fasting_true
