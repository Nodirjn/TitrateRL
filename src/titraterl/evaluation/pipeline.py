"""One-command evaluation pipeline (MVP feature F5).

``titraterl evaluate --config configs/eval/test.yaml`` regenerates every metric
table, statistical test and figure from the configuration file:

* ``episodes.parquet``   one row per (policy, patient, scenario)
* ``per_patient.csv``    averaged over scenarios
* ``summary.csv``        median and IQR per policy
* ``ci_table.csv``       median and 95 % bootstrap CI per policy, share meeting targets
* ``h1.json``            H1 test (if ``h1_policies`` is set)
* ``h2.json``            H2 decision rule against B1-B3 (if a primary policy is set)
* ``h3.json``            H3 correlation and tertile summary (if a primary policy is set)
* ``report.md``          human-readable summary of all of the above
* ``figures/*.png``      TIR / TBR box plots
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from titraterl.cohort.generator import Cohort, CohortConfig, generate_cohort
from titraterl.config import EvalConfig
from titraterl.evaluation.logging import trajectory_frame, write_trajectory
from titraterl.evaluation.runner import evaluate, per_patient, pool_training_seeds
from titraterl.evaluation.statistics import bootstrap_ci, evaluate_h1, evaluate_h2, evaluate_h3
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


def ci_table(pp: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Median and 95 % bootstrap CI (patients resampled) of the primary endpoints per policy."""
    rows = []
    for pid, d in pp.groupby("policy_id"):
        row: dict[str, Any] = {"policy_id": pid, "n_patients": len(d)}
        for col in ("maintenance_tir", "full_tbr70", "full_tbr54", "final_dose_u_per_kg"):
            v = d[col].to_numpy()
            lo, hi = bootstrap_ci(v, np.median, n_boot=2000, seed=seed)
            row[f"{col}_median"] = float(np.median(v))
            row[f"{col}_ci_low"], row[f"{col}_ci_high"] = lo, hi
        row["share_meeting_targets"] = float(((d.full_tbr70 < 0.04) & (d.full_tbr54 < 0.01)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def h1_analysis(pp: pd.DataFrame, r1: str, r3: str) -> dict:
    tir = pp.pivot(index="patient_id", columns="policy_id", values="maintenance_tir")
    tbr = pp.pivot(index="patient_id", columns="policy_id", values="full_tbr70")
    res = evaluate_h1(tbr[r3].to_numpy(), tbr[r1].to_numpy(), tir[r3].to_numpy(), tir[r1].to_numpy())
    return {"r1": r1, "r3": r3, **res}


def h3_analysis(pp: pd.DataFrame, primary: str, reference: str | None = None) -> dict:
    tir = pp.pivot(index="patient_id", columns="policy_id", values="maintenance_tir")
    if reference is None:
        candidates = [b for b in H2_BASELINES if b in tir.columns and b != primary]
        reference = max(candidates, key=lambda b: float(tir[b].median()))
    meta = pp.drop_duplicates("patient_id").set_index("patient_id")
    diff = (tir[primary] - tir[reference]).rename("tir_diff")
    rho = meta.loc[diff.index, "rho_si"].to_numpy()
    res = evaluate_h3(rho, diff.to_numpy())
    tert = meta.loc[diff.index, "si_tertile"]
    by_tertile = {
        int(t): {"n": int((tert == t).sum()), "median_tir_diff_pp": float(100 * diff[tert == t].median())}
        for t in sorted(tert.unique())
    }
    return {"primary": primary, "reference": reference, **res, "by_si_tertile": by_tertile}


def write_report(
    out: Path, cis: pd.DataFrame, h1: dict | None, h2: list[dict] | None, h3: dict | None, cfg: EvalConfig
) -> Path:
    def pct(v: float) -> str:
        return f"{100 * v:.1f}"

    lines = [
        "# Evaluation report",
        "",
        f"Split: **{cfg.split}**, {int(cis.n_patients.max())} patients x {cfg.n_scenarios} scenarios "
        "(common random numbers). Endpoints from noise-free glucose. In silico results - not medical advice.",
        "",
        "| Policy | Maintenance TIR % (95% CI) | TBR<70 % (95% CI) | TBR<54 % "
        "| Final dose U/kg | Meets targets |",
        "|---|---|---|---|---|---|",
    ]
    for r in cis.sort_values("maintenance_tir_median", ascending=False).itertuples():
        lines.append(
            f"| {r.policy_id} | {pct(r.maintenance_tir_median)} ({pct(r.maintenance_tir_ci_low)}-"
            f"{pct(r.maintenance_tir_ci_high)}) | {pct(r.full_tbr70_median)} ({pct(r.full_tbr70_ci_low)}-"
            f"{pct(r.full_tbr70_ci_high)}) | {pct(r.full_tbr54_median)} | "
            f"{r.final_dose_u_per_kg_median:.2f} | "
            f"{pct(r.share_meeting_targets)}% |"
        )
    if h1:
        lines += [
            "",
            "## H1 (RQ1)",
            "",
            f"{h1['r3']} vs {h1['r1']}: one-sided Wilcoxon p = {h1['p_value']:.4f}; "
            f"median TIR loss {h1['median_tir_loss_pp']:.1f} pp; supported: **{h1['supported']}**",
        ]
    if h2:
        lines += [
            "",
            "## H2 (RQ2)",
            "",
            "| Baseline | HL TIR gain (pp) | p (Holm) | mean TBR diff (pp) | upper CI (pp) | supported |",
            "|---|---|---|---|---|---|",
        ]
        for r in h2:
            lines.append(
                f"| {r['baseline']} | {r['tir_gain_hl_pp']:.1f} | {r['tir_p_holm']:.4f} | "
                f"{r['tbr_diff_mean_pp']:.2f} | {r['tbr_diff_ci_upper_pp']:.2f} | **{r['supported']}** |"
            )
    if h3:
        lines += [
            "",
            "## H3 (RQ3)",
            "",
            f"{h3['primary']} vs {h3['reference']}: Spearman rho = {h3['spearman_rho']:.3f}, "
            f"one-sided p = {h3['p_value']:.4f}; supported: **{h3['supported']}**",
            "",
            "| SI tertile | n | median TIR difference (pp) |",
            "|---|---|---|",
        ]
        for t, v in h3["by_si_tertile"].items():
            lines.append(f"| {t} | {v['n']} | {v['median_tir_diff_pp']:.1f} |")
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _jsonable(o: Any) -> Any:
    """JSON fallback for numpy scalars."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def run_analyses(cfg: EvalConfig, pp: pd.DataFrame, out: Path) -> Path:
    """CI table, H1-H3, report and figures from the per-patient table."""
    cis = ci_table(pp)
    cis.to_csv(out / "ci_table.csv", index=False)
    h1 = h2 = h3 = None
    if len(cfg.h1_policies) == 2:
        h1 = h1_analysis(pp, *cfg.h1_policies)
        (out / "h1.json").write_text(json.dumps(h1, indent=2, default=_jsonable))
    if cfg.primary_policy:
        h2 = h2_analysis(pp, cfg.primary_policy)
        (out / "h2.json").write_text(json.dumps(h2, indent=2, default=_jsonable))
        h3 = h3_analysis(pp, cfg.primary_policy, cfg.h3_reference)
        (out / "h3.json").write_text(json.dumps(h3, indent=2, default=_jsonable))
    make_figures(pp, out / "figures")
    return write_report(out, cis, h1, h2, h3, cfg)


def _evaluate_chunk(cfg_dict: dict, patient_ids: list[str]) -> pd.DataFrame:
    """Worker: rebuild cohort and policies in the subprocess and evaluate some patients."""
    cfg = EvalConfig(**cfg_dict)
    cohort = generate_cohort(CohortConfig.from_dict(cfg.cohort) if cfg.cohort else CohortConfig())
    patients = [cohort.get(pid) for pid in patient_ids]
    df, _ = evaluate(
        build_policies(cfg), cohort, patients, cfg.n_scenarios, cfg.base_seed, env_kwargs=cfg.env_kwargs()
    )
    return df


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
    patients = cohort.split(cfg.split)  # type: ignore[arg-type]
    if cfg.n_jobs > 1 and not cfg.log_trajectories:
        ids = [p.patient_id for p in patients]
        chunks = [ids[i :: cfg.n_jobs] for i in range(cfg.n_jobs)]
        with ProcessPoolExecutor(max_workers=cfg.n_jobs) as pool:
            frames = list(pool.map(_evaluate_chunk, [asdict(cfg)] * len(chunks), chunks))
        df = pd.concat(frames, ignore_index=True)
        results = []
    else:
        df, results = evaluate(
            build_policies(cfg),
            cohort,
            patients,
            cfg.n_scenarios,
            cfg.base_seed,
            env_kwargs=cfg.env_kwargs(),
            keep_records=cfg.log_trajectories,
            progress=progress,
        )
    df.to_parquet(out / "episodes.parquet", index=False)
    pp = per_patient(df)
    if cfg.pool_seeds:
        pp.to_csv(out / "per_patient_by_seed.csv", index=False)
        summarize_policies(pp).to_csv(out / "summary_by_seed.csv")
        pp = per_patient(pool_training_seeds(df))
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
    run_analyses(cfg, pp, out)
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
    "ci_table",
    "h1_analysis",
    "h2_analysis",
    "h3_analysis",
    "make_figures",
    "run_analyses",
    "run_pipeline",
    "summarize_policies",
    "tune_threshold_rule",
    "write_report",
]
