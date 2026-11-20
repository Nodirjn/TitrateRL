"""Run the full training protocol as a resumable job queue.

Phase 1 (RQ1): PPO with R1, R2, R3, shield off, seeds 0-4.
Phase 2 (RQ2): PPO with the selected reward R* and the shield, seeds 0-4.

Runs whose ``final_model.zip`` already exists are skipped, so the script can be stopped and
restarted at any time. ``--budget-min`` stops starting new runs after that many minutes, which
keeps every started run inside a fixed time window (useful on shared or time-limited machines).

Examples::

    python scripts/run_protocol.py --phase 1 --lanes 2
    python scripts/run_protocol.py --phase 2 --lanes 2 --budget-min 70
    python scripts/run_protocol.py --status
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2, 3, 4)
PHASES = {
    1: ("ppo_r1", "ppo_r2", "ppo_r3"),
    2: ("ppo_rstar_shield",),
}


def jobs(phase: int) -> list[tuple[str, int]]:
    # seed-major order: after each round all configurations have the same number of seeds
    return [(cfg, s) for s in SEEDS for cfg in PHASES[phase]]


def run_dir(out: Path, cfg: str, seed: int) -> Path:
    return out / f"{cfg}_s{seed}"


def done(out: Path, cfg: str, seed: int) -> bool:
    return (run_dir(out, cfg, seed) / "final_model.zip").exists()


def status(out: Path) -> None:
    for phase in PHASES:
        for cfg, seed in jobs(phase):
            mark = "done" if done(out, cfg, seed) else "-"
            print(f"phase {phase}  {cfg}_s{seed:<2} {mark}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--phase", type=int, choices=sorted(PHASES))
    ap.add_argument("--lanes", type=int, default=2, help="runs in parallel (one CPU core each)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "full")
    ap.add_argument("--timesteps", type=int, default=None, help="override total_timesteps")
    ap.add_argument(
        "--budget-min", type=float, default=None, help="do not start new runs after this many minutes"
    )
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        status(args.out)
        return 0
    if args.phase is None:
        ap.error("--phase is required unless --status is given")

    args.out.mkdir(parents=True, exist_ok=True)
    logs = args.out / "logs"
    logs.mkdir(exist_ok=True)
    start = time.monotonic()
    queue = [j for j in jobs(args.phase) if not done(args.out, *j)]
    print(f"phase {args.phase}: {len(queue)} run(s) to do, {args.lanes} lane(s)", flush=True)

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")

    def worker(cfg: str, seed: int) -> int:
        if args.budget_min is not None and (time.monotonic() - start) / 60 > args.budget_min:
            print(f"skip {cfg}_s{seed}: time budget reached", flush=True)
            return 0
        cmd = [
            sys.executable,
            "-m",
            "titraterl.cli",
            "train",
            "--config",
            str(ROOT / "configs" / "train" / f"{cfg}.yaml"),
            "--seed",
            str(seed),
            "--output-dir",
            str(args.out),
        ]
        if args.timesteps:
            cmd += ["--timesteps", str(args.timesteps)]
        t0 = time.monotonic()
        print(f"start {cfg}_s{seed}", flush=True)
        with open(logs / f"{cfg}_s{seed}.log", "w") as fh:
            rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        print(f"end   {cfg}_s{seed}: exit {rc}, {(time.monotonic() - t0) / 60:.1f} min", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.lanes) as pool:
        codes = list(pool.map(lambda j: worker(*j), queue))
    return max(codes, default=0)


if __name__ == "__main__":
    raise SystemExit(main())
