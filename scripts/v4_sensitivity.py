"""V4 sensitivity analysis: which deviation explains the gap to the Treat-to-Target trial?

The main cohort is frozen (agents are trained on it). This script varies, one at a time and
jointly, two factors the simulator idealizes and reports the V4 outcomes of Treat-to-Target
on the 90 training patients:

* insulin sensitivity: median and spread (sigma of log) of rho_si (default 2.2, 0.35);
* adherence: probability that a daily injection is missed (default 0).

    python scripts/v4_sensitivity.py --out docs/validation_v4_sensitivity.csv
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import pandas as pd

from titraterl.cohort import generate_cohort
from titraterl.cohort.generator import CohortConfig, LogNormal
from titraterl.simulation import ScenarioConfig
from titraterl.validation.checks import check_v4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--medians", type=float, nargs="+", default=[2.2])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.35, 0.5, 0.7])
    ap.add_argument("--missed", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    ap.add_argument("--n-patients", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("reports/v4_sensitivity.csv"))
    args = ap.parse_args()

    base = CohortConfig()
    rows = []
    for median, sigma, p in itertools.product(args.medians, args.sigmas, args.missed):
        cfg = replace(base, rho_si=LogNormal(median, sigma))
        res = check_v4(generate_cohort(cfg), args.n_patients, ScenarioConfig.nonadherent(p))
        d = res.details
        rows.append(
            {
                "median_rho_si": median,
                "sigma_rho_si": sigma,
                "missed_injection_prob": p,
                "fpg_week12": round(d["fpg_week12"], 1),
                "fpg_week24": round(d["fpg_week24"], 1),
                "final_dose_u_per_kg": round(d["final_dose_u_per_kg"], 2),
                "share_fpg_le_100": round(d["share_fpg_le_100"], 2),
                "v4_pass": res.passed,
            }
        )
        print(rows[-1], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
