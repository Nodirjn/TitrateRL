"""Gymnasium environment, rewards, observations and the safety shield."""

from titraterl.envs.rewards import get_reward
from titraterl.envs.titration_env import DISCRETE_ACTIONS_U, T2DTitrationEnv
from titraterl.envs.wrappers import SafetyShield, ShieldConfig, shield_filter


def make_env(shield: bool = False, **kwargs) -> T2DTitrationEnv | SafetyShield:
    """Create a titration environment, optionally wrapped with the safety shield."""
    env = T2DTitrationEnv(**kwargs)
    return SafetyShield(env) if shield else env


__all__ = [
    "DISCRETE_ACTIONS_U",
    "SafetyShield",
    "ShieldConfig",
    "T2DTitrationEnv",
    "get_reward",
    "make_env",
    "shield_filter",
]
