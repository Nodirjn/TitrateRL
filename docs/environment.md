# Environment: `TitrateRL/T2D-v0`

`titraterl.envs.T2DTitrationEnv` is a standard Gymnasium environment (proposal Table V).

```python
import gymnasium as gym
import titraterl

env = gym.make("TitrateRL/T2D-v0", reward="r3", obs_mode="full", split="train")
```

## Constructor arguments

| Argument | Default | Meaning |
|---|---|---|
| `cohort` | default 150-patient cohort | `titraterl.cohort.Cohort` |
| `split` | `"train"` | Patients sampled at `reset` (`train`, `val`, `test`) |
| `patient_ids` | `None` | Explicit patient list (overrides `split`) |
| `reward` | `"r1"` | `"r1"`, `"r2"` or `"r3"` |
| `reward_kwargs` | `{}` | `lambda1`, `lambda2` for R3 (default 5, 10) |
| `obs_mode` | `"full"` | `"full"` (33) or `"fpg_only"` (18) |
| `action_mode` | `"discrete"` | `"discrete"` (`Discrete(5)`) or `"units"` (`Box`, U) |
| `horizon_days` | 168 | Episode length T |
| `rate_limit_u` | 8.0 | Max cumulative increase over 7 days |
| `max_dose_u_per_kg` | 1.0 | Upper dose bound |
| `initial_dose_u` | 10.0 | `d_0` |
| `sensor_noise` | `True` | CGM and SMBG noise |
| `record` | `False` | Keep `DayRecord` history (needed for metrics and logs) |

`reset(options=...)` accepts `patient_id`, `scenario_seed` and `initial_dose_u`.

## Timeline

```
reset()   simulate day 0 with d_0           -> o_0
step(a_0) d_1 = clip(d_0 + a'_0), day 1     -> o_1, r_1
...
step(a_166)                    day 167     -> truncated = True   (167 steps)
```

A day runs from 07:00 to 07:00: meals during the day, injection at 21:00, fasting fingerstick at
07:00 the next morning (the end of the day).

## Observation

Daily feature vector `f_t` (all scaled to [0, 1]); the observation stacks `[f_{t-2}, f_{t-1}, f_t]`.

| # | Feature | Scaling |
|---|---|---|
| 0 | Fasting SMBG | / 400 mg/dL |
| 1 | Dose per kg | U/kg, clipped to [0, 1] |
| 2 | Applied change `a'_{t-1}` | (a' + 10) / 20 |
| 3–5 | CGM TIR, TBR<70, TAR>180 of the day | fractions |
| 6 | CGM mean | / 400 |
| 7 | CGM coefficient of variation | clipped to [0, 1] |
| 8 | CGM level-1 events in the last 72 h | min(n, 5) / 5 |
| 9 | Body weight | (BW − 50) / 100 |
| 10 | Time `t / T` | |

FPG-only variant (`obs_mode="fpg_only"`): features 0, 1, 2, a binary level-1 hypoglycemia
indicator for the last 72 h, body weight and time; 6 × 3 = 18 values. It sees exactly the inputs
of the clinical algorithms.

## Actions and constraints

Discrete actions `{0, 1, 2, 3, 4}` map to dose changes `{−4, −2, 0, +2, +4}` U.
For every policy the environment then applies:

* **weekly rate limit:** positive changes are truncated so that the increases over any 7
  consecutive days sum to at most 8 U (largest weekly step of Treat-to-Target);
* **dose bounds:** `0 ≤ d ≤ 1.0 U/kg × BW`.

## Rewards

Computed from the day's 288 CGM readings:

| Name | Formula |
|---|---|
| R1 | `TIR_t` |
| R2 | `−(1/K) Σ ρ(G_k) / 100`, with `ρ(G) = 10 [1.509 ((ln G)^1.084 − 5.381)]²` |
| R3 | `TIR_t − λ1·TBR<70,t − λ2·TBR<54,t` (λ1 = 5, λ2 = 10) |

## Safety shield

`SafetyShield` (Gymnasium wrapper; RL agents only):

1. level-1 hypoglycemia measured in the last 72 h (fasting SMBG < 70 mg/dL or a CGM event of ≥ 15
   minutes < 70 mg/dL) → no dose increase;
2. level-2 hypoglycemia in the last 24 h (< 54 mg/dL) → dose −20 %;
3. dose ≤ `d_max` (also enforced by the environment).

The shield gives no formal guarantee. The `info` dict reports `proposed_u`, `applied_u`,
`shield_active` and the running `shield_rate`.

## `info` dict

`patient_id`, `day`, `dose_u`, `proposed_u`, `applied_u`, `shield_active`, `fasting_smbg`,
`fasting_true`, `hypo_below` (dict: threshold → measured reading below it today), `body_weight`,
`d_max`. Rule-based policies use only measured fields.
