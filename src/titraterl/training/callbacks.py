"""Validation-based checkpoint selection (proposal IV-C5).

Every ``eval_freq`` transitions the current policy is evaluated on the selection
scenarios (two per validation patient). The retained checkpoint is the one with
the highest median validation TIR among checkpoints for which the consensus
targets TBR<70 < 4 % and TBR<54 < 1 % are met in at least 90 % of validation
patients; if no checkpoint qualifies, the one with the lowest median TBR<70 is
kept. SB3's reward-based best-model rule is not used, so the criterion is
identical for all rewards.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from titraterl.cohort.generator import Cohort
from titraterl.evaluation.runner import evaluate, per_patient
from titraterl.policies.sb3 import SB3Policy


@dataclass
class SelectionScore:
    timesteps: int
    median_tir: float
    median_tbr70: float
    share_meeting_targets: float

    @property
    def qualifies(self) -> bool:
        return self.share_meeting_targets >= 0.90


def is_better(new: SelectionScore, best: SelectionScore | None) -> bool:
    """Checkpoint rule: qualifying beats non-qualifying; then TIR, else lowest TBR."""
    if best is None:
        return True
    if new.qualifies != best.qualifies:
        return new.qualifies
    if new.qualifies:
        return new.median_tir > best.median_tir
    return new.median_tbr70 < best.median_tbr70


class SelectionCallback(BaseCallback):
    def __init__(
        self,
        cohort: Cohort,
        eval_freq: int,
        n_envs: int,
        out_dir: Path,
        obs_mode: str,
        shield: bool,
        reward: str,
        reward_kwargs: dict,
        scenarios_per_patient: int = 2,
        base_seed: int = 101,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.cohort = cohort
        self.freq_calls = max(eval_freq // n_envs, 1)
        self.out_dir = Path(out_dir)
        self.obs_mode = obs_mode
        self.shield = shield
        self.env_kwargs = {"reward": reward, "reward_kwargs": reward_kwargs}
        self.scenarios = scenarios_per_patient
        self.base_seed = base_seed
        self.best: SelectionScore | None = None
        self.history: list[SelectionScore] = []

    def _evaluate(self) -> SelectionScore:
        policy = SB3Policy(self.model, "selection", obs_mode=self.obs_mode, shield=self.shield)
        df, _ = evaluate(
            [policy], self.cohort, self.cohort.split("val"), self.scenarios, self.base_seed, self.env_kwargs
        )
        pp = per_patient(df)
        meets = (pp["full_tbr70"] < 0.04) & (pp["full_tbr54"] < 0.01)
        return SelectionScore(
            int(self.num_timesteps),
            float(np.median(pp["full_tir"])),
            float(np.median(pp["full_tbr70"])),
            float(meets.mean()),
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.freq_calls != 0:
            return True
        score = self._evaluate()
        self.history.append(score)
        self.logger.record("selection/median_tir", score.median_tir)
        self.logger.record("selection/median_tbr70", score.median_tbr70)
        self.logger.record("selection/share_meeting_targets", score.share_meeting_targets)
        if is_better(score, self.best):
            self.best = score
            self.model.save(self.out_dir / "best_model")
        if self.verbose:
            print(
                f"[select] t={score.timesteps} TIR={score.median_tir:.3f} "
                f"TBR70={score.median_tbr70:.4f} ok={score.share_meeting_targets:.0%}"
            )
        return True

    def _on_training_end(self) -> None:
        if self.best is None:  # training shorter than eval_freq: evaluate once
            score = self._evaluate()
            self.history.append(score)
            self.best = score
            self.model.save(self.out_dir / "best_model")
        (self.out_dir / "selection.json").write_text(
            json.dumps(
                {
                    "best": asdict(self.best),
                    "history": [asdict(s) for s in self.history],
                },
                indent=2,
            )
        )
