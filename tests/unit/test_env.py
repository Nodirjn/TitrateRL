"""Gymnasium environment: API, constraints, determinism, speed (F2)."""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import titraterl
from titraterl.envs import DISCRETE_ACTIONS_U, T2DTitrationEnv


@pytest.fixture
def env(cohort):
    return T2DTitrationEnv(cohort=cohort, split="train", record=True)


def test_registered_with_gymnasium():
    env = gym.make(titraterl.ENV_ID)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (33,)


def test_check_env(cohort):
    check_env(T2DTitrationEnv(cohort=cohort, horizon_days=10), skip_render_check=True)


@pytest.mark.parametrize("mode,dim", [("full", 33), ("fpg_only", 18)])
def test_observation_dimensions(cohort, mode, dim):
    env = T2DTitrationEnv(cohort=cohort, obs_mode=mode)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (dim,) and env.observation_space.contains(obs)


def test_episode_length_and_truncation(cohort):
    env = T2DTitrationEnv(cohort=cohort, horizon_days=168)
    env.reset(seed=0)
    steps = 0
    truncated = False
    while not truncated:
        _, _, terminated, truncated, info = env.step(2)
        assert not terminated
        steps += 1
    assert steps == 167 and info["day"] == 167


def test_weekly_rate_limit_caps_increases(env):
    env.reset(seed=0, options={"initial_dose_u": 10.0})
    doses = [env.dose_u]
    for _ in range(14):
        env.step(4)  # +4 U requested every day
        doses.append(env.dose_u)
    d = np.array(doses)
    for i in range(len(d) - 7):
        assert d[i + 7] - d[i] <= 8.0 + 1e-9
    assert d[-1] > d[0]


def test_dose_bounds(env):
    env.reset(seed=0, options={"initial_dose_u": 2.0})
    env.step(0)  # -4 U
    assert env.dose_u == 0.0
    env.reset(seed=0, options={"initial_dose_u": 1e6})
    assert env.dose_u == pytest.approx(env.d_max)


def test_same_seed_same_trajectory(cohort):
    def run():
        env = T2DTitrationEnv(cohort=cohort)
        obs, _ = env.reset(seed=5)
        out = [obs]
        for a in (3, 3, 4, 2, 1):
            out.append(env.step(a)[0])
        return np.array(out)

    np.testing.assert_array_equal(run(), run())


def test_discrete_actions():
    assert DISCRETE_ACTIONS_U == (-4.0, -2.0, 0.0, 2.0, 4.0)


def test_units_action_mode(cohort):
    env = T2DTitrationEnv(cohort=cohort, action_mode="units")
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["applied_u"] == pytest.approx(1.0)


def test_rewards_are_bounded(cohort):
    for name in ("r1", "r2", "r3"):
        env = T2DTitrationEnv(cohort=cohort, reward=name)
        env.reset(seed=0)
        r = env.step(2)[1]
        assert -100 <= r <= 1.0


def test_episode_runs_fast(cohort):
    env = T2DTitrationEnv(cohort=cohort)
    env.reset(seed=0)
    env.step(2)  # JIT warm-up
    env.reset(seed=1)
    t0 = time.perf_counter()
    truncated = False
    while not truncated:
        truncated = env.step(2)[3]
    assert time.perf_counter() - t0 < 2.0  # F2 acceptance criterion
