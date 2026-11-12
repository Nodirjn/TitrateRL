"""Statistical helpers."""

from __future__ import annotations

import numpy as np
import pytest

from titraterl.evaluation.statistics import (
    bootstrap_ci,
    evaluate_h2,
    hodges_lehmann,
    holm,
    wilcoxon_greater,
)


def test_hodges_lehmann_simple():
    assert hodges_lehmann([1, 2, 3]) == pytest.approx(2.0)


def test_holm_matches_manual():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_wilcoxon_detects_positive_shift():
    rng = np.random.default_rng(0)
    assert wilcoxon_greater(rng.normal(1.0, 1.0, 30)) < 0.01
    assert wilcoxon_greater(np.zeros(10)) == 1.0


def test_bootstrap_ci_contains_mean():
    v = np.random.default_rng(1).normal(5, 1, 50)
    lo, hi = bootstrap_ci(v, n_boot=2000)
    assert lo < v.mean() < hi


def test_h2_decision_rule():
    rng = np.random.default_rng(2)
    base = rng.uniform(0.5, 0.6, 30)
    prim = base + 0.08 + rng.normal(0, 0.01, 30)
    tbr_b = rng.uniform(0.0, 0.01, 30)
    res = evaluate_h2({"b1": prim}, {"b1": base}, {"b1": tbr_b}, {"b1": tbr_b})[0]
    assert res.supported and res.tir_gain_hl_pp > 5
