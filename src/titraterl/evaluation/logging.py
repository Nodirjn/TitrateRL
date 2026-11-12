"""Parquet log store with schema validation (proposal Table VIII).

Parquet does not enforce keys or value ranges, so every table is checked by a
validator before it is written. Two tables exist:

* ``patient``: one row per virtual patient.
* ``trajectory``: one row per 5-minute sample of an evaluated episode.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from titraterl.envs.titration_env import DayRecord


class SchemaError(ValueError):
    """A table violates the data dictionary."""


PATIENT_COLUMNS = {
    "patient_id": "object",
    "split": "object",
    "rho_si": "float64",
    "rho_beta": "float64",
    "si_tertile": "int64",
    "body_weight_kg": "float64",
    "fpg_preinsulin": "float64",
    "params_json": "object",
}

TRAJECTORY_COLUMNS = {
    "run_id": "object",
    "patient_id": "object",
    "policy_id": "object",
    "day": "int16",
    "minute": "int16",
    "glucose_true": "float64",
    "glucose_cgm": "float64",
    "glucose_smbg": "float64",
    "cho_g": "float64",
    "basal_dose_u": "float64",
    "action_u": "float64",
    "applied_u": "float64",
    "shield_active": "bool",
    "reward": "float64",
}


def _require(df: pd.DataFrame, columns: dict[str, str]) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise SchemaError(f"missing columns: {sorted(missing)}")


def validate_patient_table(df: pd.DataFrame) -> None:
    _require(df, PATIENT_COLUMNS)
    if df["patient_id"].duplicated().any():
        raise SchemaError("patient_id must be unique (PRIMARY KEY)")
    if (df["patient_id"].str.len() > 8).any():
        raise SchemaError("patient_id longer than 8 characters")
    if not df["split"].isin(["train", "val", "test"]).all():
        raise SchemaError("split must be train/val/test")
    if ((df["rho_si"] <= 0) | (df["rho_beta"] <= 0)).any():
        raise SchemaError("rho_si and rho_beta must be > 0")
    if not df["si_tertile"].isin([1, 2, 3]).all():
        raise SchemaError("si_tertile must be 1-3")
    if not df["body_weight_kg"].between(50, 150).all():
        raise SchemaError("body_weight_kg must be in [50, 150]")
    if not df["fpg_preinsulin"].between(140, 300).all():
        raise SchemaError("fpg_preinsulin must be in [140, 300]")
    if df["params_json"].isna().any():
        raise SchemaError("params_json is NOT NULL")


def validate_trajectory_table(df: pd.DataFrame, d_max: float | None = None) -> None:
    _require(df, TRAJECTORY_COLUMNS)
    if df.duplicated(["run_id", "patient_id", "day", "minute"]).any():
        raise SchemaError("(run_id, patient_id, day, minute) must be unique")
    if not df["day"].between(0, 167).all():
        raise SchemaError("day must be in 0-167")
    if not (df["minute"].between(0, 1435).all() and (df["minute"] % 5 == 0).all()):
        raise SchemaError("minute must be in 0-1435, step 5")
    if (df["glucose_true"] < 0).any():
        raise SchemaError("glucose_true must be >= 0")
    if not df["glucose_cgm"].between(39, 400).all():
        raise SchemaError("glucose_cgm must be in 39-400")
    if (df["cho_g"] < 0).any():
        raise SchemaError("cho_g must be >= 0")
    if (df["basal_dose_u"] < 0).any() or (d_max is not None and (df["basal_dose_u"] > d_max + 1e-9).any()):
        raise SchemaError("basal_dose_u out of [0, d_max]")
    if df["shield_active"].isna().any():
        raise SchemaError("shield_active is NOT NULL")


def trajectory_frame(
    records: Sequence[DayRecord], patient_id: str, policy_id: str, run_id: str | None = None
) -> pd.DataFrame:
    """Convert day records of one episode into the ``trajectory`` table."""
    run_id = run_id or str(uuid.uuid4())
    frames = []
    for r in records:
        n = r.result.glucose_cgm.size
        last = np.zeros(n, dtype=bool)
        last[-1] = True
        smbg = np.full(n, np.nan)
        smbg[-1] = r.fasting_smbg  # taken at 07:00 at the end of the day
        frames.append(
            pd.DataFrame(
                {
                    "run_id": run_id,
                    "patient_id": patient_id,
                    "policy_id": policy_id,
                    "day": np.int16(r.day),
                    "minute": (np.arange(n) * 5).astype(np.int16),
                    "glucose_true": r.result.glucose_true_5min,
                    "glucose_cgm": r.result.glucose_cgm,
                    "glucose_smbg": smbg,
                    "cho_g": r.result.meals.carbs_per_5min_g,
                    "basal_dose_u": r.dose_u,
                    "action_u": np.where(last, r.proposed_u, np.nan),
                    "applied_u": np.where(last, r.applied_u, np.nan),
                    "shield_active": r.shield_active,
                    "reward": np.where(last, np.nan if r.reward is None else r.reward, np.nan),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def write_trajectory(df: pd.DataFrame, path: str | Path, d_max: float | None = None) -> Path:
    """Validate and append a trajectory table as one Parquet file."""
    validate_trajectory_table(df, d_max)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
