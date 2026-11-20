# Full-protocol models (Week 12)

Selected checkpoints (`best_model.zip`, chosen on the 30 validation patients) of the full training
protocol: PPO with rewards R1, R2, R3 (phase 1, shield off) and PPO with R* = R1 plus the safety
shield (phase 2), each with training seeds 0-4, 10^6 transitions per run.

Each folder holds `best_model.zip`, `selection.json` (validation scores of every checkpoint) and
`run.json` (configuration and timing). The reward selection is in
`docs/results/reward_selection_full.json`. In silico research only, not for dosing real people.
