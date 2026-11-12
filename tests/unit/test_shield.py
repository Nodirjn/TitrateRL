"""Safety shield rules and wrapper."""

from __future__ import annotations

import pytest

from titraterl.envs import SafetyShield, T2DTitrationEnv, shield_filter


def test_no_hypo_passes_action_through():
    assert shield_filter(2.0, 20.0, False, False) == (2.0, False)


def test_level1_blocks_increases_only():
    assert shield_filter(4.0, 20.0, True, False) == (0.0, True)
    assert shield_filter(-2.0, 20.0, True, False) == (-2.0, False)


def test_level2_reduces_dose_by_20_percent():
    delta, active = shield_filter(4.0, 30.0, True, True)
    assert delta == pytest.approx(-6.0) and active


def test_wrapper_logs_interventions(cohort):
    env = SafetyShield(T2DTitrationEnv(cohort=cohort, record=True))
    env.reset(seed=0)
    base = env.unwrapped
    base._hypo_l1.append(True)  # simulate a measured level-1 event
    _, _, _, _, info = env.step(4)
    assert info["shield_active"] and info["applied_u"] == 0.0 and info["proposed_u"] == 4.0
    assert env.interventions == 1


def test_wrapper_requires_titration_env():
    import gymnasium as gym

    with pytest.raises(TypeError):
        SafetyShield(gym.make("CartPole-v1"))
