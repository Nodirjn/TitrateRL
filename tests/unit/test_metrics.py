"""Glycemic metrics against hand-computed examples."""

from __future__ import annotations

import numpy as np
import pytest

from titraterl.metrics import glycemic as m
from titraterl.metrics.episode import time_to_target


def test_time_in_range_bounds_inclusive():
    assert m.time_in_range([70, 180, 69.9, 180.1]) == 0.5


def test_time_below_and_above():
    g = [50, 60, 100, 200, 260]
    assert m.time_below(g, 70) == pytest.approx(0.4)
    assert m.time_below(g, 54) == pytest.approx(0.2)
    assert m.time_above(g, 180) == pytest.approx(0.4)
    assert m.time_above(g, 250) == pytest.approx(0.2)


def test_mean_cv_gmi():
    g = [100, 200]
    assert m.mean_glucose(g) == 150
    assert m.coefficient_of_variation(g) == pytest.approx(50 / 150)
    assert m.gmi(g) == pytest.approx(3.31 + 0.02392 * 150)


def test_risk_is_zero_near_112_and_symmetric_scale():
    assert m.risk([112.5])[0] < 0.01
    assert m.risk([40])[0] > 20 and m.risk([400])[0] > 20
    assert m.lbgi([200]) == 0.0 and m.hbgi([60]) == 0.0


def test_event_requires_15_minutes():
    g = np.full(30, 100.0)
    g[5:7] = 60  # 10 min -> no event
    assert m.count_events(g, 70) == 0
    g[5:8] = 60  # 15 min -> event
    assert m.count_events(g, 70) == 1


def test_event_ends_after_15_minutes_above():
    g = np.full(40, 100.0)
    g[0:3] = 60
    g[4:7] = 60  # recovery of only 5 min -> same event
    assert m.count_events(g, 70) == 1
    g[10:13] = 60  # recovery >= 15 min -> new event
    assert m.count_events(g, 70) == 2


def test_time_to_target():
    assert time_to_target([150, 120, 110, 100, 90]) == 1.0
    assert np.isnan(time_to_target([150, 140, 135]))


def test_empty_trace_raises():
    with pytest.raises(ValueError):
        m.time_in_range([])
