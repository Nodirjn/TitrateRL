"""End-to-end: evaluation with CRN, CLI and a smoke training run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from titraterl.cli import main
from titraterl.evaluation.runner import evaluate, per_patient, run_episode
from titraterl.policies import ATLantus, FixedDose, TreatToTarget

ROOT = Path(__file__).resolve().parents[2]


def test_crn_same_meals_for_every_policy(cohort):
    pt = cohort.split("test")[0]
    a = run_episode(ATLantus(), cohort, pt, seed=3, env_kwargs={"horizon_days": 10})
    b = run_episode(FixedDose(), cohort, pt, seed=3, env_kwargs={"horizon_days": 10})
    for ra, rb in zip(a.records, b.records):
        np.testing.assert_array_equal(ra.result.meals.cho_mg_min, rb.result.meals.cho_mg_min)


def test_titration_beats_no_titration(cohort):
    pts = cohort.split("val")[:3]
    df, _ = evaluate([FixedDose(), TreatToTarget()], cohort, pts, n_scenarios=1)
    pp = per_patient(df).groupby("policy_id").maintenance_tir.mean()
    assert pp["b2_treat_to_target"] > pp["b0_fixed"]


def test_cli_simulate(capsys):
    assert main(["simulate", "--policy", "b3_insight", "--days", "14"]) == 0
    assert "full_tir" in capsys.readouterr().out


def test_cli_cohort(tmp_path):
    assert main(["cohort", "--out", str(tmp_path / "c.parquet")]) == 0
    assert (tmp_path / "c.parquet").exists()


@pytest.mark.slow
def test_smoke_training(tmp_path):
    from titraterl.config import TrainConfig
    from titraterl.training.train import train

    cfg = TrainConfig.from_yaml(ROOT / "configs/train/smoke.yaml", output_dir=str(tmp_path))
    out = train(cfg)
    assert (out / "best_model.zip").exists() and (out / "selection.json").exists()


@pytest.mark.slow
def test_dashboard_renders():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=240)
    at.run()
    assert not at.exception
    assert any("In silico" in w.value for w in at.warning)


def test_evaluation_pipeline(tmp_path, cohort):
    from titraterl.config import EvalConfig
    from titraterl.evaluation.pipeline import run_pipeline

    cfg = EvalConfig(
        split="val", n_scenarios=1, rule_policies=["b0_fixed", "b3_insight"], output_dir=str(tmp_path)
    )
    small = type(cohort)(cohort.split("val")[:3])
    out = run_pipeline(cfg, cohort=small, progress=False)
    for name in ("episodes.parquet", "per_patient.csv", "summary.csv", "figures/maintenance_tir.png"):
        assert (out / name).exists()
