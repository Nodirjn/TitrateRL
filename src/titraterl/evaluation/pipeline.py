"""One-command evaluation pipeline (MVP feature F5).

``titraterl evaluate --config configs/eval/test.yaml`` regenerates every metric
table, statistical test and figure from the configuration file:

* ``episodes.parquet``   one row per (policy, patient, scenario)
* ``per_patient.csv``    averaged over scenarios
* ``summary.csv``        median and IQR per policy
* ``h2.json``            H2 decision rule against B1-B3 (if a primary policy is set)
* ``figures/*.png``      TIR / TBR box plots and dose trajectories
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from titraterl.cohort.generator import Cohort, CohortConfig, generate_cohort
from titraterl.config import EvalConfig
from titraterl.evaluation.logging import trajectory_frame, write_trajectory
from titraterl.evaluation.runner import evaluate, per_patient
from titraterl.evaluation.statistics import evaluate_h2
from titraterl.policies import RULE_POLICIES, Policy, SB3Policy

H2_BASELINES = ("b1_atlantus", "b2_treat_to_target", "b3_insight")


def build_policies(cfg: EvalConfig) -> list[Policy]:
    policies: list[Policy] = [RULE_POLICIES[name]() for name in cfg.rule_policies]
    for ck in cfg.checkpoints:
        policies.append(
            SB3Policy.load(
                ck["path"],
                algo=ck.get("algo", "ppo"),
                policy_id=ck["policy_id"],
                obs_mode=ck.get("obs_mode", "full"),
                shield=bool(ck.get("shield", False)),
            )
        )
    return policies


def summarize_policies(pp: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "maintenance_tir",
        "full_tbr70",
        "full_tbr54",
        "full_tar180",
        "full_mean",
        "full_gmi",
        "nocturnal_tbr70",
        "fasting_in_target",
        "time_to_target_days",
        "final_dose_u_per_kg",
        "dose_changes_per_week",
        "shield_rate",
    ]
    cols = [c for c in cols if c in pp.columns]
    g = pp.groupby("policy_id")[cols]
    return pd.concat({"median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75)}, axis=1)


def h2_analysis(pp: pd.DataFrame, primary: str) -> list[dict]:
    wide_tir = pp.pivot(index="patient_id", columns="policy_id", values="maintenance_tir")
    wide_tbr = pp.pivot(index="patient_id", columns="policy_id", values="full_tbr70")
    baselines = [b for b in H2_BASELINES if b in wide_tir.columns]
    res = evaluate_h2(
        {b: wide_tir[primary].to_numpy() for b in baselines},
        {b: wide_tir[b].to_numpy() for b in baselines},
        {b: wide_tbr[primary].to_numpy() for b in baselines},
        {b: wide_tbr[b].to_numpy() for b in baselines},
    )
    return [r.as_dict() for r in res]


def make_figures(pp: pd.DataFrame, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    paths = []
    order = sorted(pp["policy_id"].unique())
    for col, label in (
        ("maintenance_tir", "Maintenance TIR (%)"),
        ("full_tbr70", "TBR < 70 mg/dL, full period (%)"),
    ):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        data = [100 * pp.loc[pp.policy_id == p, col].to_numpy() for p in order]
        ax.boxplot(data, tick_labels=order, showfliers=False)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title("In silico results - not medical advice", fontsize=9)
        fig.tight_layout()
        path = out / f"{col}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def run_pipeline(cfg: EvalConfig, cohort: Cohort | None = None, progress: bool = True) -> Path:
    cohort = cohort or generate_cohort(CohortConfig.from_dict(cfg.cohort) if cfg.cohort else CohortConfig())
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policies = build_policies(cfg)
    patients = cohort.split(cfg.split)  # type: ignore[arg-type]
    df, results = evaluate(
        policies,
        cohort,
        patients,
        cfg.n_scenarios,
        cfg.base_seed,
        keep_records=cfg.log_trajectories,
        progress=progress,
    )
    df.to_parquet(out / "episodes.parquet", index=False)
    pp = per_patient(df)
    pp.to_csv(out / "per_patient.csv", index=False)
    summarize_policies(pp).to_csv(out / "summary.csv")
    if cfg.log_trajectories:
        for res in results:
            frame = trajectory_frame(res.records, res.patient_id, res.policy_id)
            bw = cohort.get(res.patient_id).params.body_weight_kg
            write_trajectory(
                frame,
                out / "trajectories" / f"{res.policy_id}_{res.patient_id}_{res.scenario_index}.parquet",
                d_max=bw,
            )
    if cfg.primary_policy:
        (out / "h2.json").write_text(json.dumps(h2_analysis(pp, cfg.primary_policy), indent=2))
    make_figures(pp, out / "figures")
    (out / "eval_config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=str))
    return out


def tune_threshold_rule(
    cohort: Cohort, n_scenarios: int = 1, base_seed: int = 11, progress: bool = False
) -> pd.DataFrame:
    """B4: evaluate the 36 threshold rules on training patients, rank on validation."""
    from titraterl.policies.rules import ThresholdRule

    rules = [ThresholdRule(p) for p in ThresholdRule.grid()]
    rows = []
    for split in ("train", "val"):
        df, _ = evaluate(rules, cohort, cohort.split(split), n_scenarios, base_seed, progress=progress)
        pp = per_patient(df)
        g = pp.groupby("policy_id")
        meets = g.apply(
            lambda d: float(((d.full_tbr70 < 0.04) & (d.full_tbr54 < 0.01)).mean()), include_groups=False
        )
        rows.append(
            pd.DataFrame(
                {
                    "split": split,
                    "median_tir": g["full_tir"].median(),
                    "median_tbr70": g["full_tbr70"].median(),
                    "share_meeting_targets": meets,
                }
            )
        )
    table = pd.concat(rows).reset_index()
    val = table[table.split == "val"].copy()
    val["qualifies"] = val["share_meeting_targets"] >= 0.9
    val = val.sort_values(["qualifies", "median_tir"], ascending=[False, False])
    table.attrs["selected"] = str(val.iloc[0]["policy_id"]) if len(val) else ""
    return table


__all__ = [
    "build_policies",
    "h2_analysis",
    "make_figures",
    "run_pipeline",
    "summarize_policies",
    "tune_threshold_rule",
]
