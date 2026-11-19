"""Pipeline analyses (CI table, H1, H3, report) and system-test helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from titraterl.config import EvalConfig
from titraterl.evaluation.pipeline import ci_table, h1_analysis, h2_analysis, h3_analysis, write_report
from titraterl.validation.system_test import SystemTestResult, _max_weekly_increase, to_markdown


def synthetic_per_patient(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rho = rng.lognormal(0, 0.35, n)
    tert = pd.qcut(rho, 3, labels=[1, 2, 3]).astype(int)
    rows = []
    base = rng.uniform(0.5, 0.6, n)
    effects = {
        "b1_atlantus": 0.0,
        "b2_treat_to_target": 0.01,
        "b3_insight": 0.005,
        "ppo_r1": 0.05,
        "ppo_r3": 0.04,
        "ppo_shield": 0.08,
    }
    for pid, eff in effects.items():
        for i in range(n):
            gain = eff * (1.5 - 0.5 * rho[i]) if pid.startswith("ppo") else eff
            rows.append(
                {
                    "policy_id": pid,
                    "patient_id": f"T2D-{i:03d}",
                    "split": "test",
                    "si_tertile": int(tert[i]),
                    "rho_si": float(rho[i]),
                    "maintenance_tir": base[i] + gain + rng.normal(0, 0.005),
                    "full_tbr70": 0.02 if pid == "ppo_r1" else 0.005 + rng.uniform(0, 0.002),
                    "full_tbr54": 0.001,
                    "final_dose_u_per_kg": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_ci_table_brackets_median():
    t = ci_table(synthetic_per_patient())
    assert set(t.policy_id) >= {"ppo_shield", "b1_atlantus"}
    assert (t.maintenance_tir_ci_low <= t.maintenance_tir_median).all()
    assert (t.maintenance_tir_median <= t.maintenance_tir_ci_high).all()


def test_h1_detects_lower_hypoglycemia_for_r3():
    res = h1_analysis(synthetic_per_patient(), "ppo_r1", "ppo_r3")
    assert res["supported"] and res["p_value"] < 0.05


def test_h3_negative_correlation():
    res = h3_analysis(synthetic_per_patient(), "ppo_shield")
    assert res["reference"] == "b2_treat_to_target"
    assert res["spearman_rho"] < 0 and res["supported"]
    assert set(res["by_si_tertile"]) == {1, 2, 3}


def test_report_contains_all_sections(tmp_path):
    pp = synthetic_per_patient()
    path = write_report(
        tmp_path,
        ci_table(pp),
        h1_analysis(pp, "ppo_r1", "ppo_r3"),
        h2_analysis(pp, "ppo_shield"),
        h3_analysis(pp, "ppo_shield"),
        EvalConfig(n_scenarios=10),
    )
    text = path.read_text()
    assert "## H1" in text and "## H2" in text and "## H3" in text and "not medical advice" in text


def test_max_weekly_increase():
    doses = [10] + [12] * 3 + [14, 16, 18, 20, 20, 20]
    assert _max_weekly_increase(doses) == 10.0


def test_system_test_markdown():
    md = to_markdown(
        [
            SystemTestResult("ST-01", "F1", "x", "ok", "ok", True),
            SystemTestResult("ST-02", "F2", "y", "bad", "ok", False),
            SystemTestResult("ST-03", "F3", "z", "-", "ok", None),
        ]
    )
    assert "1 passed, 1 failed, 1 skipped" in md and "**FAIL**" in md
