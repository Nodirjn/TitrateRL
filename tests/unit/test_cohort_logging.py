"""Cohort generation (F1) and the Parquet log schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titraterl.cohort import Cohort, CohortConfig, generate_cohort
from titraterl.evaluation.logging import (
    SchemaError,
    trajectory_frame,
    validate_patient_table,
    validate_trajectory_table,
)


def test_split_sizes_and_stratification(cohort):
    assert len(cohort) == 150
    sizes = {s: len(cohort.split(s)) for s in ("train", "val", "test")}
    assert sizes == {"train": 90, "val": 30, "test": 30}
    for s in ("val", "test"):
        tert = [p.si_tertile for p in cohort.split(s)]
        assert all(tert.count(t) == 10 for t in (1, 2, 3))


def test_cohort_is_reproducible():
    a = generate_cohort(CohortConfig(seed=5)).to_frame()
    b = generate_cohort(CohortConfig(seed=5)).to_frame()
    pd.testing.assert_frame_equal(a, b)


def test_parameter_ranges(cohort):
    df = cohort.to_frame()
    assert df.fpg_preinsulin.between(140, 300).all()
    assert df.body_weight_kg.between(50, 150).all()
    validate_patient_table(df)


def test_save_load_roundtrip(cohort, tmp_path):
    path = cohort.save(tmp_path / "cohort.parquet")
    loaded = Cohort.load(path)
    np.testing.assert_allclose(loaded.patients[7].params.vector(), cohort.patients[7].params.vector())


def test_patient_schema_rejects_duplicates(cohort):
    df = cohort.to_frame()
    df.loc[1, "patient_id"] = df.loc[0, "patient_id"]
    with pytest.raises(SchemaError):
        validate_patient_table(df)


def test_trajectory_schema(cohort):
    from titraterl.evaluation.runner import run_episode
    from titraterl.policies import ATLantus

    pt = cohort.split("test")[0]
    res = run_episode(ATLantus(), cohort, pt, seed=0, env_kwargs={"horizon_days": 5})
    df = trajectory_frame(res.records, pt.patient_id, "b1_atlantus")
    assert len(df) == 5 * 288
    validate_trajectory_table(df, d_max=pt.params.body_weight_kg)
    df.loc[0, "glucose_cgm"] = 500
    with pytest.raises(SchemaError):
        validate_trajectory_table(df)
