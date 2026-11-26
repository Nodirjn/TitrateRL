"""Pre-registered reward selection (phase 1 -> R*) over all training seeds.

Rule (proposal IV-D3): among rewards for which at least 90 % of the validation patients meet
the consensus hypoglycemia targets, choose the one with the highest median validation TIR.
With several seeds, a reward's value is the median over seeds of the retained checkpoint.

    python scripts/select_reward.py --runs runs/full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

REWARDS = ("r1", "r2", "r3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs/full"))
    ap.add_argument("--min-share", type=float, default=0.90)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    table = {}
    for r in REWARDS:
        bests = [
            json.loads(p.read_text())["best"] for p in sorted(args.runs.glob(f"ppo_{r}_s*/selection.json"))
        ]
        if not bests:
            continue
        table[r] = {
            "n_seeds": len(bests),
            "median_tir": median(b["median_tir"] for b in bests),
            "share_meeting_targets": median(b["share_meeting_targets"] for b in bests),
            "per_seed_tir": [round(b["median_tir"], 4) for b in bests],
        }
    eligible = {r: v for r, v in table.items() if v["share_meeting_targets"] >= args.min_share}
    selected = max(eligible, key=lambda r: eligible[r]["median_tir"]) if eligible else None
    result = {
        "rule": f"max median TIR s.t. share >= {args.min_share}",
        "rewards": table,
        "selected": selected,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
