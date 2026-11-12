"""Rule-based policies reproduce the published titration steps (F4)."""

from __future__ import annotations

import pytest

from titraterl.policies import ATLantus, FixedDose, Insight, ThresholdRule, TreatToTarget


def info(fasting: float, dose: float = 20.0, hypo: tuple[float, ...] = ()) -> dict:
    return {
        "fasting_smbg": fasting,
        "dose_u": dose,
        "hypo_below": {t: t in hypo for t in (54.0, 56.0, 70.0, 72.0)},
    }


def run(policy, fastings, dose=20.0, hypo_days=None):
    hypo_days = hypo_days or {}
    policy.reset(info(fastings[0], dose))
    return [policy.act(None, info(f, dose, hypo_days.get(i, ()))) for i, f in enumerate(fastings[1:])]


def test_b0_constant_dose():
    p = FixedDose()
    assert p.initial_dose(90) == pytest.approx(18.0)
    assert run(p, [200] * 10) == [0.0] * 9


def test_b1_plus_two_every_three_days_above_100():
    acts = run(ATLantus(), [150] * 10)
    assert acts == [0, 0, 2, 0, 0, 2, 0, 0, 2]


def test_b1_no_change_at_target():
    assert set(run(ATLantus(), [95] * 10)) == {0.0}


def test_b1_reduces_10_percent_below_70():
    acts = run(ATLantus(), [150, 150, 65], dose=30.0, hypo_days={1: (70.0, 72.0)})
    assert acts[1] == pytest.approx(-3.0)


def test_b1_blocks_increase_after_reading_below_72():
    acts = run(ATLantus(), [150] * 4, hypo_days={0: (72.0,)})
    assert acts == [0, 0, 0]


@pytest.mark.parametrize("fasting,step", [(200, 8), (150, 6), (130, 4), (110, 2), (100, 0)])
def test_b2_weekly_steps(fasting, step):
    acts = run(TreatToTarget(), [fasting] * 8)
    assert acts[:6] == [0] * 6 and acts[6] == step


def test_b2_decreases_4u_after_reading_below_56():
    acts = run(TreatToTarget(), [150] * 8, hypo_days={3: (56.0, 70.0, 72.0)})
    assert acts[6] == -4.0


def test_b3_daily_plus_one():
    assert run(Insight(), [130] * 5) == [1, 1, 1, 1]


def test_b3_ada_reduction():
    acts = run(Insight(), [130, 130, 60], dose=40.0, hypo_days={1: (70.0, 72.0)})
    assert acts[1] == pytest.approx(-4.0)


def test_b4_grid_has_36_rules():
    assert len(ThresholdRule.grid()) == 36
