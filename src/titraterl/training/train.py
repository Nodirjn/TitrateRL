"""Training entry point for PPO and DQN agents (proposal Algorithm 1, TRAIN)."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from titraterl import __version__
from titraterl.cohort.generator import Cohort, generate_cohort
from titraterl.config import TrainConfig
from titraterl.envs import make_env
from titraterl.training.callbacks import SelectionCallback

ACTIVATIONS = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _env_factory(cfg: TrainConfig, cohort: Cohort, rank: int):
    def _init() -> gym.Env:
        env = make_env(
            shield=cfg.shield,
            cohort=cohort,
            split="train",
            reward=cfg.reward,
            reward_kwargs=cfg.reward_kwargs,
            obs_mode=cfg.obs_mode,
        )
        env.reset(seed=cfg.seed * 1000 + rank)
        return Monitor(env)

    return _init


def build_model(cfg: TrainConfig, env: Any, tensorboard_log: str | None) -> PPO | DQN:
    if cfg.algo == "ppo":
        h = cfg.ppo
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=h.learning_rate,
            n_steps=h.n_steps,
            batch_size=h.batch_size,
            n_epochs=h.n_epochs,
            clip_range=h.clip_range,
            gae_lambda=h.gae_lambda,
            gamma=h.gamma,
            seed=cfg.seed,
            device="cpu",
            policy_kwargs={
                "net_arch": {"pi": h.net_arch, "vf": h.net_arch},
                "activation_fn": ACTIVATIONS[h.activation],
            },
            tensorboard_log=tensorboard_log,
            verbose=0,
        )
    h = cfg.dqn
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=h.learning_rate,
        buffer_size=h.buffer_size,
        batch_size=h.batch_size,
        target_update_interval=h.target_update_interval,
        train_freq=h.train_freq,
        learning_starts=h.learning_starts,
        exploration_fraction=h.exploration_fraction,
        exploration_final_eps=h.exploration_final_eps,
        gamma=h.gamma,
        seed=cfg.seed,
        device="cpu",
        policy_kwargs={"net_arch": h.net_arch, "activation_fn": ACTIVATIONS[h.activation]},
        tensorboard_log=tensorboard_log,
        verbose=0,
    )


def train(cfg: TrainConfig, cohort: Cohort | None = None) -> Path:
    """Train one (algorithm, reward, observation, shield, seed) configuration.

    Returns:
        The run directory containing ``best_model.zip``, ``final_model.zip``,
        ``selection.json`` and ``run.json`` (config, versions, commit hash).
    """
    cohort = cohort or generate_cohort(cfg.cohort_config())
    out = Path(cfg.output_dir) / cfg.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "titraterl_version": __version__,
                "commit": git_commit(),
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
            indent=2,
            default=str,
        )
    )

    try:
        import tensorboard  # noqa: F401

        tb: str | None = str(out / "tb")
    except ImportError:
        tb = None
    vec = DummyVecEnv([_env_factory(cfg, cohort, i) for i in range(cfg.n_envs)])
    model = build_model(cfg, vec, tb)
    callback = SelectionCallback(
        cohort,
        cfg.eval_freq,
        cfg.n_envs,
        out,
        cfg.obs_mode,
        cfg.shield,
        cfg.reward,
        cfg.reward_kwargs,
        cfg.selection_scenarios_per_patient,
        cfg.selection_base_seed,
    )
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback, progress_bar=False)
    model.save(out / "final_model")
    vec.close()
    return out
