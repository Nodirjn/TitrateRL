"""Adapter that exposes a trained Stable-Baselines3 model as a :class:`Policy`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from titraterl.policies.base import Policy


class SB3Policy(Policy):
    """A trained PPO or DQN agent.

    Args:
        model: A loaded SB3 model (``PPO`` or ``DQN``).
        policy_id: Identifier, e.g. ``"ppo_r3_shield_s2"``.
        obs_mode: Observation mode the model was trained with.
        shield: Whether the safety shield is applied during evaluation.
        deterministic: Use the greedy action.
    """

    action_mode = "discrete"

    def __init__(
        self,
        model: Any,
        policy_id: str,
        obs_mode: str = "full",
        shield: bool = False,
        deterministic: bool = True,
    ) -> None:
        self.model = model
        self.policy_id = policy_id
        self.obs_mode = obs_mode
        self.shield = shield
        self.deterministic = deterministic

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(np.asarray(action).item())

    @classmethod
    def load(cls, path: str | Path, algo: str = "ppo", **kwargs: Any) -> SB3Policy:
        from stable_baselines3 import DQN, PPO

        algo_cls = {"ppo": PPO, "dqn": DQN}[algo.lower()]
        model = algo_cls.load(str(path), device="cpu")
        kwargs.setdefault("policy_id", Path(path).stem)
        return cls(model, **kwargs)
