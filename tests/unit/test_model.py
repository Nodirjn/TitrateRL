"""V1: integrator, equilibrium, periodic steady state and dose-response tests."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from titraterl.simulation import model
from titraterl.simulation.model import simulate_day
from titraterl.simulation.parameters import (
    MINUTES_PER_DAY,
    PMOL_PER_UNIT,
    BasalDerivationError,
    P,
    PatientParameters,
    S,
    derive_basal,
)
from titraterl.validation.checks import fasting_after_constant_dose, integrator_error

ZERO = np.zeros(MINUTES_PER_DAY)


def test_numba_index_constants_match_layout():
    for name, value in vars(model).items():
        if name.startswith("P_"):
            assert getattr(P, name[2:]) == value, name
        if name.startswith("S_"):
            assert getattr(S, name[2:]) == value, name


def test_rk4_matches_solve_ivp():
    assert integrator_error(minutes=600) < 0.5  # mg/dL


@pytest.mark.parametrize("gb", [140.0, 196.0, 300.0])
def test_patient_is_in_equilibrium_without_insulin(gb):
    pt = PatientParameters("EQ", 80.0, gb, rho_si=0.7, rho_beta=1.3)
    x0 = pt.initial_state()
    x_end, g, _, _ = simulate_day(x0, pt.vector(), ZERO, ZERO, 0.0, -1, 0.0)
    assert np.max(np.abs(g - gb)) < 1e-6
    assert np.max(np.abs(x_end - x0)) < 1e-6


def test_every_cohort_patient_is_in_equilibrium(cohort):
    for cp in cohort.patients[::10]:
        _, g, _, _ = simulate_day(cp.params.initial_state(), cp.params.vector(), ZERO, ZERO, 0.0, -1, 0.0)
        assert np.max(np.abs(g - cp.params.fpg_preinsulin)) < 1e-6


def test_periodic_steady_state_under_constant_dose(nominal_patient):
    f = fasting_after_constant_dose(nominal_patient, 20.0, days=30)
    assert abs(f[-1] - f[-2]) < 0.1


def test_dose_response_is_monotone(nominal_patient):
    values = [fasting_after_constant_dose(nominal_patient, d, days=25)[-1] for d in (0, 10, 20, 40, 60)]
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(values))
    assert values[-1] < values[0] - 20  # insulin has a material effect


def test_injection_raises_plasma_insulin(nominal_patient):
    p = nominal_patient.vector()
    x0 = nominal_patient.initial_state()
    x_end, _, _, _ = simulate_day(x0, p, ZERO, ZERO, 0.0, 0, 20 * PMOL_PER_UNIT / 90.0)
    assert x_end[S.IP] > x0[S.IP]


def test_meal_raises_glucose(nominal_patient):
    p = nominal_patient.vector()
    cho = ZERO.copy()
    start = ZERO.copy()
    cho[60:70] = 7500.0
    start[60] = 75000.0
    _, g, _, _ = simulate_day(nominal_patient.initial_state(), p, cho, start, 0.0, -1, 0.0)
    assert g.max() > nominal_patient.fpg_preinsulin + 50


def test_basal_derivation_rejects_impossible_patient():
    nominal = PatientParameters("x", 80, 196).nominal.replace(EGPb=0.5)
    pt = PatientParameters("BAD", 80.0, 196.0, nominal=nominal)
    with pytest.raises(BasalDerivationError):
        derive_basal(pt)
