"""TitrateRL: simulation-based reinforcement learning for personalized basal
insulin titration in Type 2 diabetes.

In silico research software only. Not a medical device; not for dosing real people.
"""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

__version__ = "0.1.0.dev0"

ENV_ID = "TitrateRL/T2D-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="titraterl.envs.titration_env:T2DTitrationEnv",
    )

__all__ = ["ENV_ID", "__version__"]
