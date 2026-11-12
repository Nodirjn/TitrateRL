"""Safety shield (proposal Section IV-C7).

The shield is a rule-based action filter for the RL agents, inspired by
shielding in safe RL (Alshiekh et al., AAAI 2018). Using measured values only:

(a) level-1 hypoglycemia (fasting SMBG < 70 mg/dL or a CGM level-1 event) in the
    last 72 h -> dose increases are blocked (``a' <= 0``);
(b) level-2 hypoglycemia (SMBG < 54 mg/dL or a CGM level-2 event) in the last
    24 h -> the dose is reduced by 20 %;
(c) the dose never exceeds ``d_max`` (also enforced by the environment).

The shield gives **no formal safety guarantee**: insulin that has already been
injected keeps acting, so hypoglycemia is also penalized and reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from titraterl.envs.titration_env import T2DTitrationEnv


@dataclass(frozen=True)
class ShieldConfig:
    level2_reduction: float = 0.20


def shield_filter(
    delta_u: float,
    dose_u: float,
    level1_recent: bool,
    level2_recent: bool,
    config: ShieldConfig | None = None,
) -> tuple[float, bool]:
    """Pure shield rule. Returns ``(filtered_change, shield_intervened)``."""
    config = config or ShieldConfig()
    if level2_recent:
        new = -config.level2_reduction * dose_u
        return new, new != delta_u
    if level1_recent and delta_u > 0:
        return 0.0, True
    return delta_u, False


class SafetyShield(gym.Wrapper):
    """Gymnasium wrapper applying :func:`shield_filter` to every action.

    Must wrap a :class:`T2DTitrationEnv` directly (or through wrappers that do
    not change actions), because it calls ``step_units`` of the base env.
    """

    def __init__(self, env: gym.Env, config: ShieldConfig | None = None) -> None:
        super().__init__(env)
        if not isinstance(env.unwrapped, T2DTitrationEnv):
            raise TypeError("SafetyShield requires a T2DTitrationEnv")
        self.config = config or ShieldConfig()
        self.interventions = 0
        self.steps = 0

    def reset(self, **kwargs: Any):
        self.interventions = 0
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        base: T2DTitrationEnv = self.env.unwrapped  # type: ignore[assignment]
        proposed = base.action_to_units(action)
        filtered, active = shield_filter(
            proposed, base.dose_u, base.hypo_level1_recent(), base.hypo_level2_recent(), self.config
        )
        self.steps += 1
        self.interventions += int(active)
        obs, reward, terminated, truncated, info = base.step_units(
            filtered, shield_active=active, proposed_u=proposed
        )
        info["shield_rate"] = self.interventions / self.steps
        return obs, reward, terminated, truncated, info
