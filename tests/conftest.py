"""Shared fixtures."""

from __future__ import annotations

import pytest

from titraterl.cohort import default_cohort
from titraterl.simulation.parameters import PatientParameters


@pytest.fixture(scope="session")
def cohort():
    return default_cohort()


@pytest.fixture
def nominal_patient() -> PatientParameters:
    return PatientParameters("TEST-1", body_weight_kg=90.0, fpg_preinsulin=196.0)
