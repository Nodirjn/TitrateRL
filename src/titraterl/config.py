"""YAML experiment configuration.

Every experiment is fully defined by a version-controlled YAML file in
``configs/``. Unknown keys raise an error so that typos cannot silently change
an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from titraterl.cohort.generator import CohortConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def _strict(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return data


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99
    net_arch: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"


@dataclass
class DQNConfig:
    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    batch_size: int = 64
    target_update_interval: int = 1_000
    train_freq: int = 4
    learning_starts: int = 1_000
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    gamma: float = 0.99
    net_arch: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"


@dataclass
class TrainConfig:
    run_name: str = "ppo_r1"
    algo: str = "ppo"
    reward: str = "r1"
    reward_kwargs: dict[str, float] = field(default_factory=dict)
    obs_mode: str = "full"
    shield: bool = False
    total_timesteps: int = 1_000_000
    n_envs: int = 8
    seed: int = 0
    eval_freq: int = 50_000
    selection_scenarios_per_patient: int = 2
    selection_base_seed: int = 101
    output_dir: str = "runs"
    cohort: dict[str, Any] = field(default_factory=dict)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        data = dict(_strict(cls, data))
        if "ppo" in data:
            data["ppo"] = PPOConfig(**_strict(PPOConfig, data["ppo"] or {}))
        if "dqn" in data:
            data["dqn"] = DQNConfig(**_strict(DQNConfig, data["dqn"] or {}))
        cfg = cls(**data)
        if cfg.algo not in ("ppo", "dqn"):
            raise ValueError("algo must be 'ppo' or 'dqn'")
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> TrainConfig:
        data = load_yaml(path)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.from_dict(data)

    def cohort_config(self) -> CohortConfig:
        return CohortConfig.from_dict(self.cohort) if self.cohort else CohortConfig()

    @property
    def run_id(self) -> str:
        return f"{self.run_name}_s{self.seed}"


@dataclass
class EvalConfig:
    split: str = "test"
    n_scenarios: int = 10
    base_seed: int = 7
    rule_policies: list[str] = field(
        default_factory=lambda: ["b0_fixed", "b1_atlantus", "b2_treat_to_target", "b3_insight"]
    )
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    primary_policy: str | None = None
    output_dir: str = "results"
    log_trajectories: bool = False
    cohort: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        return cls(**_strict(cls, load_yaml(path)))
