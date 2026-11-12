"""Pre-specified statistical analysis (proposal Sections II-E and IV-C9).

* H1: one-sided paired Wilcoxon signed-rank test (R3 vs R1 on TBR<70).
* H2: per clinical algorithm, (i) Holm-adjusted one-sided paired Wilcoxon test
  of the maintenance-phase TIR difference, (ii) Hodges-Lehmann estimate >= 5 pp,
  (iii) upper limit of the 95 % bootstrap CI of the mean paired TBR<70
  difference below the 1 pp non-inferiority margin.
* H3: one-sided Spearman correlation between ``rho_si`` and the TIR difference.

Every patient contributes one paired value (averaged over scenarios and seeds);
bootstrap confidence intervals resample patients.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


def hodges_lehmann(diff: Sequence[float]) -> float:
    """Hodges-Lehmann estimator: median of the Walsh averages of paired differences."""
    d = np.asarray(diff, dtype=float)
    i, j = np.triu_indices(d.size)
    return float(np.median((d[i] + d[j]) / 2.0))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI that resamples the units (patients)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot = np.apply_along_axis(statistic, 1, v[idx])
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def wilcoxon_greater(diff: Sequence[float]) -> float:
    """One-sided paired Wilcoxon signed-rank p-value for H_a: median(diff) > 0."""
    d = np.asarray(diff, dtype=float)
    if np.allclose(d, 0.0):
        return 1.0
    return float(stats.wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


@dataclass
class H2Result:
    baseline: str
    n: int
    tir_gain_hl_pp: float
    tir_p_raw: float
    tir_p_holm: float
    tbr_diff_mean_pp: float
    tbr_diff_ci_upper_pp: float
    superior: bool
    clinically_meaningful: bool
    non_inferior: bool

    @property
    def supported(self) -> bool:
        return self.superior and self.clinically_meaningful and self.non_inferior

    def as_dict(self) -> dict:
        d = asdict(self)
        d["supported"] = self.supported
        return d


def evaluate_h2(
    primary_tir: dict[str, np.ndarray],
    baseline_tir: dict[str, np.ndarray],
    primary_tbr: dict[str, np.ndarray],
    baseline_tbr: dict[str, np.ndarray],
    alpha: float = 0.05,
    min_gain_pp: float = 5.0,
    ni_margin_pp: float = 1.0,
    seed: int = 0,
) -> list[H2Result]:
    """H2 decision rule against every baseline (inputs are fractions, paired by patient).

    Args:
        primary_tir: ``{baseline: per-patient maintenance TIR of the primary policy}``.
            The primary values are repeated per key so arrays stay aligned with
            the baseline arrays.
    """
    names = list(baseline_tir)
    raw = [wilcoxon_greater(primary_tir[b] - baseline_tir[b]) for b in names]
    adj = holm(raw)
    out = []
    for b, p_raw, p_adj in zip(names, raw, adj, strict=True):
        d_tir = 100 * (primary_tir[b] - baseline_tir[b])
        d_tbr = 100 * (primary_tbr[b] - baseline_tbr[b])
        hl = hodges_lehmann(d_tir)
        _, ci_hi = bootstrap_ci(d_tbr, seed=seed)
        out.append(
            H2Result(
                b,
                d_tir.size,
                hl,
                p_raw,
                float(p_adj),
                float(np.mean(d_tbr)),
                ci_hi,
                p_adj < alpha,
                hl >= min_gain_pp,
                ci_hi < ni_margin_pp,
            )
        )
    return out


def evaluate_h1(
    tbr_r3: np.ndarray,
    tbr_r1: np.ndarray,
    tir_r3: np.ndarray,
    tir_r1: np.ndarray,
    alpha: float = 0.05,
    max_tir_loss_pp: float = 5.0,
) -> dict[str, float | bool]:
    """H1: R3 has lower TBR<70 than R1 (one-sided Wilcoxon) and median TIR at most 5 pp lower."""
    p = wilcoxon_greater(np.asarray(tbr_r1) - np.asarray(tbr_r3))
    tir_loss = 100 * (np.median(tir_r1) - np.median(tir_r3))
    return {
        "p_value": p,
        "median_tir_loss_pp": float(tir_loss),
        "supported": bool(p < alpha and tir_loss <= max_tir_loss_pp),
    }


def evaluate_h3(rho_si: np.ndarray, tir_diff: np.ndarray, alpha: float = 0.05) -> dict[str, float | bool]:
    """H3 (part i): negative Spearman correlation between rho_si and the TIR benefit."""
    res = stats.spearmanr(rho_si, tir_diff, alternative="less")
    return {
        "spearman_rho": float(res.statistic),
        "p_value": float(res.pvalue),
        "supported": bool(res.pvalue < alpha),
    }
