"""Command-line interface: ``titraterl <command>``.

Commands:
    cohort      generate the virtual cohort and write it to Parquet
    validate    run simulator checks V1-V5 (decision gate G1)
    simulate    simulate one rule-based policy for one patient (quick look)
    train       train a PPO/DQN agent from a YAML configuration
    evaluate    run the evaluation pipeline from a YAML configuration
    tune-b4     tune the simulator-tuned threshold rule B4
    dashboard   launch the Streamlit dashboard
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DISCLAIMER = "In silico research software - not medical advice, not for dosing real people."


def _cohort_from(path: str | None):
    from titraterl.cohort.generator import CohortConfig, generate_cohort
    from titraterl.config import load_yaml

    cfg = CohortConfig.from_dict(load_yaml(path)) if path else CohortConfig()
    return generate_cohort(cfg)


def cmd_cohort(args: argparse.Namespace) -> int:
    cohort = _cohort_from(args.config)
    path = cohort.save(args.out)
    df = cohort.to_frame()
    print(f"wrote {len(df)} patients to {path}")
    print(df.groupby("split").size().to_string())
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from titraterl.validation.checks import as_dicts, gate_g1, run_all, to_markdown

    results = run_all(_cohort_from(args.cohort), quick=args.quick)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(as_dicts(results), indent=2))
    md = to_markdown(results)
    (out / "validation.md").write_text(md + "\n")
    print(md)
    return 0 if (gate_g1(results) or not args.strict) else 1


def cmd_simulate(args: argparse.Namespace) -> int:
    from titraterl.evaluation.runner import run_episode
    from titraterl.policies import RULE_POLICIES

    cohort = _cohort_from(args.cohort)
    patient = cohort.get(args.patient) if args.patient else cohort.split("test")[0]
    policy = RULE_POLICIES[args.policy]()
    res = run_episode(policy, cohort, patient, args.seed, env_kwargs={"horizon_days": args.days})
    keys = [
        "maintenance_tir",
        "full_tir",
        "full_tbr70",
        "full_tbr54",
        "final_dose_u",
        "final_dose_u_per_kg",
        "time_to_target_days",
    ]
    print(f"{policy.policy_id} on {patient.patient_id} ({res.wall_time_s:.2f} s)")
    for k in keys:
        if k in res.metrics:
            print(f"  {k:<22} {res.metrics[k]:.4f}")
    print(DISCLAIMER)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from titraterl.config import TrainConfig
    from titraterl.training.train import train

    cfg = TrainConfig.from_yaml(
        args.config, seed=args.seed, total_timesteps=args.timesteps, output_dir=args.output_dir
    )
    out = train(cfg)
    print(f"run saved to {out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from titraterl.config import EvalConfig
    from titraterl.evaluation.pipeline import run_pipeline

    cfg = EvalConfig.from_yaml(args.config)
    if args.n_scenarios:
        cfg.n_scenarios = args.n_scenarios
    if args.output_dir:
        cfg.output_dir = args.output_dir
    out = run_pipeline(cfg)
    print(f"results written to {out}")
    print(DISCLAIMER)
    return 0


def cmd_tune_b4(args: argparse.Namespace) -> int:
    from titraterl.evaluation.pipeline import tune_threshold_rule

    table = tune_threshold_rule(_cohort_from(args.cohort), progress=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"selected: {table.attrs['selected']} (table: {args.out})")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    app = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="titraterl", description=__doc__.split("\n")[0], epilog=DISCLAIMER)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("cohort", help="generate the virtual cohort")
    p.add_argument("--config", default=None, help="cohort YAML (default: built-in)")
    p.add_argument("--out", default="data/cohort.parquet")
    p.set_defaults(func=cmd_cohort)

    p = sub.add_parser("validate", help="run simulator checks V1-V5 (gate G1)")
    p.add_argument("--cohort", default=None)
    p.add_argument("--out", default="reports")
    p.add_argument("--quick", action="store_true", help="fewer patients (CI)")
    p.add_argument("--strict", action="store_true", help="exit 1 if G1 fails")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("simulate", help="simulate one rule-based policy")
    p.add_argument(
        "--policy",
        default="b1_atlantus",
        choices=["b0_fixed", "b1_atlantus", "b2_treat_to_target", "b3_insight", "b4_threshold"],
    )
    p.add_argument("--patient", default=None, help="patient id, e.g. T2D-017")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--days", type=int, default=168)
    p.add_argument("--cohort", default=None)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("train", help="train an agent from a YAML config")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="run the evaluation pipeline")
    p.add_argument("--config", required=True)
    p.add_argument("--n-scenarios", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("tune-b4", help="tune the threshold rule B4")
    p.add_argument("--cohort", default=None)
    p.add_argument("--out", default="results/b4_tuning.csv")
    p.set_defaults(func=cmd_tune_b4)

    p = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    p.set_defaults(func=cmd_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
