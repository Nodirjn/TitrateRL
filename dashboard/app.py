"""TitrateRL interactive dashboard (MVP feature F6).

Run with ``titraterl dashboard`` or ``streamlit run dashboard/app.py``.

The dashboard accepts only virtual-patient identifiers and offers no input for
real glucose values or doses. Every output is labelled as a simulation result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from titraterl.cohort import default_cohort
from titraterl.evaluation.runner import run_episode
from titraterl.policies import RULE_POLICIES, Policy, SB3Policy

DISCLAIMER = (
    "**In silico simulation only.** TitrateRL is research software. It is not a medical "
    "device and must not be used to dose insulin for real people."
)
RUNS_DIR = Path("runs")


def discover_checkpoints() -> dict[str, dict]:
    """Trained agents saved by ``titraterl train`` (runs/<run_id>/best_model.zip)."""
    found = {}
    for run_json in sorted(RUNS_DIR.glob("*/run.json")):
        model = run_json.parent / "best_model.zip"
        if model.exists():
            cfg = json.loads(run_json.read_text())["config"]
            found[run_json.parent.name] = {
                "path": model,
                "algo": cfg["algo"],
                "obs_mode": cfg["obs_mode"],
                "shield": cfg["shield"],
            }
    return found


def make_policy(name: str, checkpoints: dict[str, dict]) -> Policy:
    if name in RULE_POLICIES:
        return RULE_POLICIES[name]()
    ck = checkpoints[name]
    return SB3Policy.load(
        ck["path"], algo=ck["algo"], policy_id=name, obs_mode=ck["obs_mode"], shield=ck["shield"]
    )


@st.cache_data(show_spinner=False)
def simulate(policy_name: str, patient_id: str, seed: int) -> tuple[pd.DataFrame, dict]:
    cohort = default_cohort()
    policy = make_policy(policy_name, discover_checkpoints())
    res = run_episode(policy, cohort, cohort.get(patient_id), seed)
    rows = []
    for r in res.records:
        t = r.day + np.arange(r.result.glucose_cgm.size) * 5 / 1440
        rows.append(
            pd.DataFrame(
                {
                    "day": t,
                    "cgm": r.result.glucose_cgm,
                    "true": r.result.glucose_true_5min,
                    "dose_u": r.dose_u,
                }
            )
        )
    trace = pd.concat(rows, ignore_index=True)
    trace["policy"] = policy_name
    return trace, res.metrics


def main() -> None:
    st.set_page_config(page_title="TitrateRL dashboard", layout="wide")
    st.title("TitrateRL - basal insulin titration (in silico)")
    st.warning(DISCLAIMER)

    cohort = default_cohort()
    checkpoints = discover_checkpoints()
    test_ids = [p.patient_id for p in cohort.split("test")]

    with st.sidebar:
        st.header("Simulation")
        patient_id = st.selectbox("Virtual test patient", test_ids)
        options = list(RULE_POLICIES) + list(checkpoints)
        chosen = st.multiselect(
            "Policies (max 3)", options, default=["b1_atlantus", "b2_treat_to_target"], max_selections=3
        )
        seed = st.number_input("Scenario seed", min_value=0, max_value=2**31 - 1, value=0, step=1)
        week = st.slider("Week to display (glucose trace)", 1, 24, 24)
        if not checkpoints:
            st.caption("No trained agents found in runs/. Train one with `titraterl train`.")

    cp = cohort.get(patient_id)
    st.caption(
        f"{patient_id}: body weight {cp.params.body_weight_kg:.0f} kg, pre-insulin fasting "
        f"glucose {cp.params.fpg_preinsulin:.0f} mg/dL, insulin-sensitivity tertile "
        f"{cp.si_tertile} (hidden from the policies)"
    )
    if not chosen:
        st.info("Select at least one policy.")
        return

    traces, metrics = [], {}
    with st.spinner("Simulating 24 weeks..."):
        for name in chosen:
            tr, m = simulate(name, patient_id, int(seed))
            traces.append(tr)
            metrics[name] = m
    data = pd.concat(traces, ignore_index=True)

    lo, hi = (week - 1) * 7, week * 7
    window = data[(data.day >= lo) & (data.day < hi)]
    st.subheader(f"CGM glucose, week {week} (mg/dL)")
    st.line_chart(window.pivot_table(index="day", columns="policy", values="cgm"))
    st.subheader("Daily basal dose (U)")
    daily = data.assign(d=data.day.astype(int)).groupby(["d", "policy"]).dose_u.first().unstack()
    st.line_chart(daily)

    st.subheader("Outcome metrics (noise-free glucose)")
    show = [
        "maintenance_tir",
        "full_tir",
        "full_tbr70",
        "full_tbr54",
        "full_tar180",
        "full_gmi",
        "nocturnal_tbr70",
        "time_to_target_days",
        "final_dose_u_per_kg",
        "dose_changes_per_week",
        "shield_rate",
    ]
    table = pd.DataFrame(metrics).loc[[k for k in show if k in next(iter(metrics.values()))]]
    st.dataframe(table.style.format("{:.3f}"))

    st.download_button(
        "Export traces (CSV)",
        data.to_csv(index=False).encode(),
        file_name=f"titraterl_{patient_id}_seed{seed}.csv",
        mime="text/csv",
    )
    st.caption("Simulation result - " + DISCLAIMER.replace("**", ""))


if __name__ == "__main__":
    main()
