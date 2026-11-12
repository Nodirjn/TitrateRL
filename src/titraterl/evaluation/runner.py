"""Policy evaluation with common random numbers (CRN).

Every policy is simulated on the same patients and the same scenario seeds, so
meals and sensor noise are identical across policies and differences come from
the policy alone.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from titraterl.cohort.generator import Cohort, CohortPatient
from titraterl.envs.titration_env import DayRecord, T2DTitrationEnv
from titraterl.envs.wrappers import SafetyShield
from titraterl.metrics.episode import episode_metrics
from titraterl.policies.base import Policy


def scenario_seed(base_seed: int, patient_id: str, index: int) -> int:
    """Deterministic scenario seed for (patient, scenario index)."""
    pid_num = int("".join(ch for ch in patient_id if ch.isdigit()) or 0)
    ss = np.random.SeedSequence([base_seed, pid_num, index])
    return int(ss.generate_state(1)[0])


@dataclass
class EpisodeResult:
    policy_id: str
    patient_id: str
    scenario_index: int
    scenario_seed: int
    records: list[DayRecord] = field(repr=False)
    metrics: dict[str, float]
    wall_time_s: float


def run_episode(
    policy: Policy,
    cohort: Cohort,
    patient: CohortPatient,
    seed: int,
    scenario_index: int = 0,
    env_kwargs: dict[str, Any] | None = None,
) -> EpisodeResult:
    """Simulate one 24-week course of ``policy`` for ``patient``."""
    kwargs = dict(env_kwargs or {})
    kwargs.setdefault("obs_mode", policy.obs_mode)
    env = T2DTitrationEnv(
        cohort=cohort, patient_ids=[patient.patient_id], action_mode=policy.action_mode, record=True, **kwargs
    )
    runner_env: T2DTitrationEnv | SafetyShield = SafetyShield(env) if policy.shield else env
    t0 = time.perf_counter()
    obs, info = runner_env.reset(
        options={
            "patient_id": patient.patient_id,
            "scenario_seed": seed,
            "initial_dose_u": policy.initial_dose(patient.params.body_weight_kg),
        }
    )
    policy.reset(info)
    truncated = terminated = False
    while not (terminated or truncated):
        action = policy.act(obs, info)
        if policy.action_mode == "units":
            action = np.array([action], dtype=np.float32)
        obs, _, terminated, truncated, info = runner_env.step(action)
    metrics = episode_metrics(env.history, patient.params.body_weight_kg)
    return EpisodeResult(
        policy.policy_id,
        patient.patient_id,
        scenario_index,
        seed,
        env.history,
        metrics,
        time.perf_counter() - t0,
    )


def evaluate(
    policies: Iterable[Policy],
    cohort: Cohort,
    patients: Sequence[CohortPatient],
    n_scenarios: int = 10,
    base_seed: int = 7,
    env_kwargs: dict[str, Any] | None = None,
    keep_records: bool = False,
    progress: bool = False,
) -> tuple[pd.DataFrame, list[EpisodeResult]]:
    """Evaluate policies on ``patients`` x ``n_scenarios`` with CRN.

    Returns:
        A data frame with one row per (policy, patient, scenario) and every
        episode metric, and the list of episode results (with day records only
        if ``keep_records``).
    """
    rows: list[dict[str, Any]] = []
    results: list[EpisodeResult] = []
    policies = list(policies)
    total = len(policies) * len(patients) * n_scenarios
    done = 0
    for policy in policies:
        for patient in patients:
            for k in range(n_scenarios):
                seed = scenario_seed(base_seed, patient.patient_id, k)
                res = run_episode(policy, cohort, patient, seed, k, env_kwargs)
                rows.append(
                    {
                        "policy_id": res.policy_id,
                        "patient_id": res.patient_id,
                        "split": patient.split,
                        "si_tertile": patient.si_tertile,
                        "rho_si": patient.params.rho_si,
                        "scenario_index": k,
                        "scenario_seed": seed,
                        **res.metrics,
                    }
                )
                if not keep_records:
                    res.records = []
                results.append(res)
                done += 1
                if progress:
                    print(
                        f"\r[{done}/{total}] {policy.policy_id:<24} {patient.patient_id}", end="", flush=True
                    )
    if progress:
        print()
    return pd.DataFrame(rows), results


def per_patient(df: pd.DataFrame, group: str = "policy_id") -> pd.DataFrame:
    """Average over scenarios so that every patient contributes one value per policy."""
    keys = [group, "patient_id", "split", "si_tertile", "rho_si"]
    numeric = df.select_dtypes("number").columns.difference(
        ["scenario_index", "scenario_seed", "si_tertile", "rho_si"]
    )
    return df.groupby(keys, as_index=False)[list(numeric)].mean()
