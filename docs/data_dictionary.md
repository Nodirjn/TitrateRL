# Data dictionary

Simulation results are stored as Parquet (proposal Table VIII). Parquet does not enforce keys or
value ranges, so `titraterl.evaluation.logging` validates every table before it is written
(`validate_patient_table`, `validate_trajectory_table`).

## Table `patient` (one row per virtual patient)

Written by `titraterl cohort` → `data/cohort.parquet`.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `patient_id` | VARCHAR(8) | PRIMARY KEY | e.g. `T2D-017` |
| `split` | ENUM | train / val / test | cohort partition |
| `rho_si`, `rho_beta` | FLOAT | > 0 | insulin-sensitivity and beta-cell factors |
| `si_tertile` | SMALLINT | 1–3 | insulin-sensitivity stratum (RQ3) |
| `body_weight_kg` | FLOAT | 50–150 | body weight |
| `fpg_preinsulin` | FLOAT | 140–300 | fasting glucose `Gb` before insulin (mg/dL) |
| `params_json` | JSON | NOT NULL | nominal parameters and full ODE parameter vector |

## Table `trajectory` (one row per 5-minute sample)

Written by the evaluation pipeline when `log_trajectories: true`.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `run_id` | UUID | PK with `patient_id`, `day`, `minute` | policy × seed × scenario run |
| `patient_id` | VARCHAR(8) | FK → `patient` | simulated patient |
| `policy_id` | VARCHAR(32) | NOT NULL | e.g. `ppo_r3_shield_s2`, `b1_atlantus` |
| `day`, `minute` | SMALLINT | 0–167; 0–1435 step 5 | minute 0 = 07:00 |
| `glucose_true` | FLOAT | ≥ 0 | noise-free plasma glucose (evaluation) |
| `glucose_cgm` | FLOAT | 39–400 | CGM reading (observation, reward, shield) |
| `glucose_smbg` | FLOAT | NULL except at day end | fasting fingerstick |
| `cho_g` | FLOAT | ≥ 0 | carbohydrate ingested in the interval |
| `basal_dose_u` | FLOAT | 0–`d_max` | dose in effect for the day |
| `action_u`, `applied_u` | FLOAT | NULL except at day end | proposed and applied change |
| `shield_active` | BOOL | NOT NULL | shield intervention that day |
| `reward` | FLOAT | NULL except at day end | daily reward |

## Episode table `episodes.parquet`

One row per (policy, patient, scenario) with identifiers (`policy_id`, `patient_id`, `split`,
`si_tertile`, `rho_si`, `scenario_index`, `scenario_seed`) and every metric of
[evaluation.md](evaluation.md).

## Run metadata `runs/<run_id>/run.json`

Full training configuration, package version, git commit hash, Python and PyTorch versions.
`selection.json` holds the validation history and the retained checkpoint.
