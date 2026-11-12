"""Glargine absorption (V2), meal scenarios and sensor models."""

from __future__ import annotations

import numpy as np
import pytest

from titraterl.simulation.scenario import ScenarioConfig, clock_to_minute, generate_day
from titraterl.simulation.sensors import CGMSensor, smbg_reading
from titraterl.validation.checks import check_v2, glargine_profile


def test_glargine_profile_is_flat_and_reaches_steady_state():
    res = check_v2()
    assert res.passed, res.summary


def test_glargine_profile_integrates_to_one_dose():
    assert glargine_profile(days=30).sum() == pytest.approx(1.0, abs=1e-3)


def test_clock_to_minute():
    assert clock_to_minute(7.0) == 0
    assert clock_to_minute(21.0) == 840
    assert clock_to_minute(6.0) == 1380


def test_meal_scenario_is_reproducible():
    a = generate_day(np.random.default_rng(3))
    b = generate_day(np.random.default_rng(3))
    np.testing.assert_array_equal(a.cho_mg_min, b.cho_mg_min)


def test_meal_amounts_and_windows():
    for seed in range(50):
        day = generate_day(np.random.default_rng(seed))
        assert 3 <= len(day.meals) <= 4
        total = day.cho_mg_min.sum() / 1000
        assert 0.8 * 195 - 1e-6 <= total <= 1.2 * 205 + 1e-6
        minutes = sorted(m for m, _ in day.meals)
        assert clock_to_minute(7.5) <= minutes[0] <= clock_to_minute(8.5)


def test_skipped_meals_keep_random_stream_aligned():
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    generate_day(rng_a, ScenarioConfig(skip_probability=1.0))
    generate_day(rng_b, ScenarioConfig())
    assert rng_a.random() == rng_b.random()


def test_cgm_range_and_shape():
    cgm = CGMSensor(np.random.default_rng(0))
    out = cgm.measure_day(np.full(1440, 120.0))
    assert out.shape == (288,)
    assert out.min() >= 39 and out.max() <= 400
    assert abs(np.mean(out) - 120) < 30


def test_cgm_noise_can_be_disabled():
    cgm = CGMSensor(np.random.default_rng(0), enabled=False)
    np.testing.assert_allclose(cgm.measure_day(np.full(1440, 150.0)), 150.0)


@pytest.mark.parametrize("g", [80.0, 200.0])
def test_smbg_meets_iso_15197_share(g):
    rng = np.random.default_rng(0)
    readings = np.array([smbg_reading(g, rng) for _ in range(20_000)])
    limit = 15.0 if g < 100 else 0.15 * g
    assert 0.94 <= np.mean(np.abs(readings - g) <= limit) <= 0.97
