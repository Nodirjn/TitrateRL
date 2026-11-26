"""System test suite (Week 11, System Testing and Evaluation).

``titraterl system-test`` runs end-to-end checks of the integrated system against
the MVP acceptance criteria (proposal Table IX) and writes a report with one row
per test case: identifier, requirement, measured value, criterion and verdict.

Unit and integration tests (``pytest``) verify components; the system tests
below verify the assembled system as a user or an evaluator would exercise it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from titraterl.cohort.generator import Cohort, default_cohort


@dataclass
class SystemTestResult:
    test_id: str
    requirement: str
    description: str
    measured: str
    criterion: str
    passed: bool | None  # None = skipped
    seconds: float = 0.0


def _timed(fn: Callable[[], tuple[str, bool | None]]) -> tuple[str, bool | None, float]:
    t0 = time.perf_counter()
    try:
        measured, raw = fn()
        ok = None if raw is None else bool(raw)
    except Exception as exc:  # a crashing test is a failed test
        measured, ok = f"error: {type(exc).__name__}: {exc}", False
    return measured, ok, time.perf_counter() - t0


# ----------------------------------------------------------------------------- tests
def st_unit_tests(root: Path) -> tuple[str, bool]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", "not slow", "-p", "no:cacheprovider"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    last = [ln for ln in proc.stdout.strip().splitlines() if "passed" in ln or "failed" in ln]
    return (last[-1].strip("= ") if last else proc.stdout[-200:]), proc.returncode == 0


def st_cohort(cohort: Cohort) -> tuple[str, bool]:
    from titraterl.cohort.generator import generate_cohort

    again = generate_cohort(cohort.config) if cohort.config else default_cohort()
    same = cohort.to_frame().equals(again.to_frame())
    sizes = {s: len(cohort.split(s)) for s in ("train", "val", "test")}
    ok = same and sizes == {"train": 90, "val": 30, "test": 30}
    return f"{len(cohort)} patients {sizes}; regenerated identically: {same}", ok


def st_gymnasium_api() -> tuple[str, bool]:
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env

    import titraterl

    env = gym.make(titraterl.ENV_ID)
    check_env(env.unwrapped, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    return f"{titraterl.ENV_ID} registered; check_env passed; obs shape {obs.shape}", obs.shape == (33,)


def st_episode_runtime(cohort: Cohort) -> tuple[str, bool]:
    from titraterl.envs import T2DTitrationEnv

    env = T2DTitrationEnv(cohort=cohort)
    env.reset(seed=0)
    env.step(2)
    times = []
    for seed in range(3):
        env.reset(seed=seed)
        t0 = time.perf_counter()
        done = False
        while not done:
            done = env.step(2)[3]
        times.append(time.perf_counter() - t0)
    worst = max(times)
    return f"168-day episode: mean {np.mean(times):.3f} s, worst {worst:.3f} s", worst < 2.0


def st_dashboard_rerun(cohort: Cohort) -> tuple[str, bool]:
    from titraterl.evaluation.runner import run_episode
    from titraterl.policies import ATLantus, Insight, TreatToTarget

    patient = cohort.split("test")[0]
    t0 = time.perf_counter()
    for policy in (ATLantus(), TreatToTarget(), Insight()):
        run_episode(policy, cohort, patient, seed=123)
    dt = time.perf_counter() - t0
    return f"3 policies x 24 weeks re-run in {dt:.2f} s", dt < 5.0


def st_determinism(cohort: Cohort) -> tuple[str, bool]:
    from titraterl.evaluation.runner import run_episode
    from titraterl.policies import TreatToTarget

    patient = cohort.split("test")[1]
    a = run_episode(TreatToTarget(), cohort, patient, seed=9).metrics
    b = run_episode(TreatToTarget(), cohort, patient, seed=9).metrics
    same = all(np.isclose(a[k], b[k], equal_nan=True) for k in a)
    return f"identical metrics for repeated seed: {same}", same


def st_crn(cohort: Cohort) -> tuple[str, bool]:
    from titraterl.evaluation.runner import run_episode
    from titraterl.policies import FixedDose, Insight

    patient = cohort.split("test")[2]
    a = run_episode(FixedDose(), cohort, patient, seed=4, env_kwargs={"horizon_days": 14}).records
    b = run_episode(Insight(), cohort, patient, seed=4, env_kwargs={"horizon_days": 14}).records
    meals = all(np.array_equal(x.result.meals.cho_mg_min, y.result.meals.cho_mg_min) for x, y in zip(a, b))
    return f"identical meals across policies over 14 days: {meals}", meals


def _max_weekly_increase(doses: list[float]) -> float:
    d = np.asarray(doses)
    inc = np.clip(np.diff(d), 0, None)
    if inc.size < 7:
        return float(inc.sum())
    return float(np.max(np.convolve(inc, np.ones(7), mode="valid")))


def st_safety_constraints(cohort: Cohort, runs_dir: Path) -> tuple[str, bool]:
    """Rate limit and dose bound hold for every policy, including trained agents."""
    from titraterl.evaluation.runner import run_episode
    from titraterl.policies import ATLantus, Insight, TreatToTarget

    policies = [ATLantus(), TreatToTarget(), Insight(), *load_trained_policies(runs_dir)]
    worst_inc, worst_ratio, n = 0.0, 0.0, 0
    for policy in policies:
        for patient in cohort.split("test")[:5]:
            res = run_episode(policy, cohort, patient, seed=17)
            doses = [r.dose_u for r in res.records]
            worst_inc = max(worst_inc, _max_weekly_increase(doses))
            worst_ratio = max(worst_ratio, max(doses) / patient.params.body_weight_kg)
            n += 1
    ok = worst_inc <= 8.0 + 1e-9 and worst_ratio <= 1.0 + 1e-9
    return (
        f"{n} episodes ({len(policies)} policies): max 7-day increase {worst_inc:.1f} U, "
        f"max dose {worst_ratio:.2f} U/kg",
        ok,
    )


def st_shield(cohort: Cohort) -> tuple[str, bool]:
    from titraterl.envs import SafetyShield, T2DTitrationEnv

    env = SafetyShield(T2DTitrationEnv(cohort=cohort))
    env.reset(seed=0)
    base = env.unwrapped
    base._hypo_l1.append(True)
    info1 = env.step(4)[4]
    base._hypo_l2.append(True)
    dose_before = base.dose_u
    info2 = env.step(4)[4]
    ok = info1["applied_u"] == 0.0 and np.isclose(info2["applied_u"], -0.2 * dose_before)
    return (
        f"level-1: +4 U -> {info1['applied_u']:+.1f} U; level-2: +4 U -> {info2['applied_u']:+.2f} U "
        f"(dose {dose_before:.1f} U)",
        ok,
    )


def load_trained_policies(runs_dir: Path) -> list[Any]:
    from titraterl.policies import SB3Policy

    out = []
    models = Path(__file__).resolve().parents[3] / "models"
    seen: set[str] = set()
    dirs = (models / "full", Path(runs_dir), models / "pilot")  # full-protocol agents first
    run_files = [p for d in dirs for p in sorted(d.glob("*/run.json"))]
    for run_json in run_files:
        if run_json.parent.name in seen:
            continue
        seen.add(run_json.parent.name)
        model = run_json.parent / "best_model.zip"
        if model.exists():
            cfg = json.loads(run_json.read_text())["config"]
            out.append(
                SB3Policy.load(
                    model,
                    algo=cfg["algo"],
                    policy_id=run_json.parent.name,
                    obs_mode=cfg["obs_mode"],
                    shield=cfg["shield"],
                )
            )
    return out


def st_trained_agents(cohort: Cohort, runs_dir: Path) -> tuple[str, bool | None]:
    from titraterl.evaluation.runner import run_episode

    policies = load_trained_policies(runs_dir)
    if not policies:
        return "no trained agents in runs/", None
    patient = cohort.split("test")[0]
    names = []
    for p in policies:
        m = run_episode(p, cohort, patient, seed=0).metrics
        names.append(f"{p.policy_id} (TIR {100 * m['maintenance_tir']:.0f}%)")
    return f"{len(policies)} agents load and complete an episode: " + ", ".join(names), True


def st_pipeline(cohort: Cohort, tmp: Path) -> tuple[str, bool]:
    from titraterl.config import EvalConfig
    from titraterl.evaluation.pipeline import run_pipeline

    small = Cohort(cohort.split("val")[:3])
    cfg = EvalConfig(
        split="val",
        n_scenarios=1,
        rule_policies=["b0_fixed", "b1_atlantus", "b3_insight"],
        output_dir=str(tmp),
        primary_policy="b3_insight",
    )
    out = run_pipeline(cfg, cohort=small, progress=False)
    files = ["episodes.parquet", "per_patient.csv", "ci_table.csv", "h2.json", "report.md"]
    missing = [f for f in files if not (out / f).exists()]
    return f"pipeline outputs present: {not missing}" + (
        f" (missing {missing})" if missing else ""
    ), not missing


def st_validation_gate(cohort: Cohort) -> tuple[str, bool]:
    from titraterl.validation.checks import check_v1, check_v2, check_v4

    v1, v2, v4 = check_v1(cohort, n_patients=3), check_v2(), check_v4(cohort, n_patients=30)
    return (
        f"V1 {'pass' if v1.passed else 'FAIL'}, V2 {'pass' if v2.passed else 'FAIL'}, "
        f"V4 {'pass' if v4.passed else 'FAIL'} ({v4.summary})",
        bool(v1.passed and v2.passed and v4.passed),
    )


# ----------------------------------------------------------------------------- runner
def run_system_tests(
    root: Path, runs_dir: Path, tmp: Path, include_unit: bool = True
) -> list[SystemTestResult]:
    cohort = default_cohort()
    plan: list[tuple[str, str, str, str, Callable[[], tuple[str, bool | None]]]] = [
        ("ST-01", "all", "Unit and integration test suite", "all tests pass", lambda: st_unit_tests(root)),
        (
            "ST-02",
            "F1",
            "Cohort generated reproducibly, 90/30/30 split",
            "identical; sizes",
            lambda: st_cohort(cohort),
        ),
        (
            "ST-03",
            "F1",
            "Simulator checks of gate G1 (V1, V2, V4)",
            "all pass",
            lambda: st_validation_gate(cohort),
        ),
        (
            "ST-04",
            "F2, F7",
            "Environment registered and passes Gymnasium check",
            "check_env passes",
            st_gymnasium_api,
        ),
        ("ST-05", "F2", "Runtime of one 168-day episode", "< 2 s", lambda: st_episode_runtime(cohort)),
        ("ST-06", "F2", "Same seed gives the same trajectory", "identical", lambda: st_determinism(cohort)),
        ("ST-07", "F5", "Common random numbers across policies", "identical meals", lambda: st_crn(cohort)),
        (
            "ST-08",
            "F2",
            "Rate limit and dose bound for every policy",
            "<= 8 U / 7 d; <= 1 U/kg",
            lambda: st_safety_constraints(cohort, runs_dir),
        ),
        (
            "ST-09",
            "F2",
            "Safety shield blocks increases and reduces dose",
            "0 U; -20 %",
            lambda: st_shield(cohort),
        ),
        (
            "ST-10",
            "F3",
            "Trained agents load and run end to end",
            "all complete",
            lambda: st_trained_agents(cohort, runs_dir),
        ),
        (
            "ST-11",
            "F5",
            "Evaluation pipeline produces all outputs",
            "all files",
            lambda: st_pipeline(cohort, tmp),
        ),
        (
            "ST-12",
            "F6",
            "Dashboard 24-week re-run for 3 policies",
            "< 5 s",
            lambda: st_dashboard_rerun(cohort),
        ),
    ]
    if not include_unit:
        plan = plan[1:]
    results = []
    for tid, req, desc, crit, fn in plan:
        measured, ok, dt = _timed(fn)
        results.append(SystemTestResult(tid, req, desc, measured, crit, ok, round(dt, 2)))
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        print(f"[{status}] {tid} {desc}: {measured}", flush=True)
    return results


def to_markdown(results: list[SystemTestResult]) -> str:
    n_pass = sum(r.passed is True for r in results)
    n_fail = sum(r.passed is False for r in results)
    lines = [
        "# System test report",
        "",
        f"**{n_pass} passed, {n_fail} failed, {len(results) - n_pass - n_fail} skipped** "
        f"of {len(results)} system tests.",
        "",
        "| ID | Req. | Test | Measured | Criterion | Result | Time (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        status = "SKIP" if r.passed is None else ("PASS" if r.passed else "**FAIL**")
        lines.append(
            f"| {r.test_id} | {r.requirement} | {r.description} | {r.measured} | {r.criterion} | "
            f"{status} | {r.seconds} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(results: list[SystemTestResult], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "system_test.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    path = out / "system_test.md"
    path.write_text(to_markdown(results))
    return path
