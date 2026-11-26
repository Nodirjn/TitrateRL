# Pilot agents (Week 11)

Trained PPO agents used for the Week 11 system testing and evaluation and for the dashboard demo.

| Agent | Reward | Shield | Training days | Seed | Selected checkpoint |
|---|---|---|---|---|---|
| `ppo_r1_s0` | R1 (time in range) | no | 3 × 10⁵ | 0 | see `selection.json` |
| `ppo_r2_s0` | R2 (risk index) | no | 3 × 10⁵ | 0 | see `selection.json` |
| `ppo_r3_s0` | R3 (TIR − 5·TBR<70 − 10·TBR<54) | no | 3 × 10⁵ | 0 | see `selection.json` |
| `ppo_r3_shield_s0` | R3 | yes | 3 × 10⁵ | 0 | exploratory |
| `ppo_rstar_shield_s0` | R★ = R1 | yes | 3 × 10⁵ | 0 | **primary policy** |

Each folder holds `best_model.zip` (Stable-Baselines3 PPO), `run.json` (full configuration,
package version, commit hash) and `selection.json` (validation history and the retained
checkpoint).

**These are pilot agents.** They were trained with 30 % of the planned budget (10⁶ days) and one
seed instead of five. The full protocol is run in Weeks 11–12 (`configs/train/*.yaml`, seeds 0–4).

Reproduce:

```bash
titraterl train --config configs/train/ppo_r1.yaml --seed 0 --timesteps 300000
titraterl train --config configs/train/ppo_rstar_shield.yaml --seed 0 --timesteps 300000
```

Load in Python:

```python
from titraterl.policies import SB3Policy
agent = SB3Policy.load("models/pilot/ppo_rstar_shield_s0/best_model.zip",
                       policy_id="ppo_rstar_shield_s0", shield=True)
```

In silico research artifacts only; not for dosing real people.
