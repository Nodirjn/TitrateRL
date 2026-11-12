"""Common ``Policy`` interface shared by rule-based baselines and RL agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Policy(ABC):
    """A titration policy.

    Attributes:
        policy_id: Short identifier used in logs, e.g. ``"b1_atlantus"``.
        action_mode: ``"units"`` (returns a dose change in U) or ``"discrete"``
            (returns an index into ``DISCRETE_ACTIONS_U``).
        obs_mode: Observation the policy expects (``"full"`` or ``"fpg_only"``).
        shield: Whether the safety shield filters this policy's actions.
    """

    policy_id: str = "policy"
    action_mode: str = "units"
    obs_mode: str = "full"
    shield: bool = False

    def initial_dose(self, body_weight_kg: float) -> float:
        """Dose of the first day (U). Default 10 U, as recommended by the ADA."""
        return 10.0

    def reset(self, info: dict[str, Any]) -> None:  # noqa: B027 - optional hook
        """Called once per episode with the info dict returned by ``env.reset``."""

    @abstractmethod
    def act(self, obs: np.ndarray, info: dict[str, Any]) -> Any:
        """Return the action for the next day given the latest observation and info."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.policy_id!r})"
