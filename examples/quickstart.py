"""Quickstart: simulate the clinical baselines and a random agent on one virtual patient.

Run:  python examples/quickstart.py
In silico only - not medical advice.
"""

from __future__ import annotations

import gymnasium as gym

import titraterl  # noqa: F401  (registers TitrateRL/T2D-v0)
from titraterl.cohort import default_cohort
from titraterl.evaluation import run_episode
from titraterl.policies import ATLantus, FixedDose, Insight, TreatToTarget


def main() -> None:
    cohort = default_cohort()
    patient = cohort.split("test")[3]
    print(
        f"Patient {patient.patient_id}: {patient.params.body_weight_kg:.0f} kg, "
        f"pre-insulin fasting glucose {patient.params.fpg_preinsulin:.0f} mg/dL\n"
    )

    print(f"{'policy':<22}{'TIR wk21-24':>12}{'TBR<70':>9}{'final dose':>12}")
    for policy in (FixedDose(), ATLantus(), TreatToTarget(), Insight()):
        m = run_episode(policy, cohort, patient, seed=0).metrics
        print(
            f"{policy.policy_id:<22}{100 * m['maintenance_tir']:>11.1f}%"
            f"{100 * m['full_tbr70']:>8.2f}%{m['final_dose_u']:>10.1f} U"
        )

    env = gym.make("TitrateRL/T2D-v0", reward="r3")
    env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        done = terminated or truncated
    print(f"\nRandom agent on {info['patient_id']}: return {total:.1f}, final dose {info['dose_u']:.0f} U")


if __name__ == "__main__":
    main()
