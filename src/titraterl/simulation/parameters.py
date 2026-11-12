"""Model parameters of the Type 2 diabetes (T2D) virtual patient.

The physiological model is the meal simulation model of Dalla Man, Rizza and
Cobelli (IEEE TBME 54(10), 2007) with its published Type 2 parameter set,
extended in this project with subcutaneous long-acting (glargine) insulin.

Two kinds of parameters exist:

* **Nominal** parameters (:data:`NOMINAL_T2D`) are shared by all patients.
* **Derived basal** parameters (``kp1``, ``Vm0``, ``m6``, ``Sb`` ...) are
  re-computed for every patient so that the patient is in equilibrium at its
  own pre-insulin fasting glucose ``Gb`` (see :func:`derive_basal`).

Every value used by the integrator is stored in one flat ``float64`` array so
that the right-hand side can be compiled with Numba. :class:`P` holds the
array indices; :class:`S` holds the state indices.

.. note::
   The nominal T2D values follow Table I of Dalla Man et al. (2007). They must be
   re-checked against the paper before milestone M1 (see ``docs/model.md``).
   The glargine absorption constants are a project choice identified to satisfy
   validation check V2 (flat profile, steady state by the 4th dose).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
PMOL_PER_UNIT = 6000.0
"""1 international unit (U) of insulin = 6000 pmol (same convention as simglucose)."""

MINUTES_PER_DAY = 1440


# ---------------------------------------------------------------------------
# State vector layout (15 states, Appendix A of the proposal)
# ---------------------------------------------------------------------------
class S:
    """Indices of the 15-component state vector ``x``."""

    GP = 0  # glucose mass in plasma (mg/kg)
    GT = 1  # glucose mass in slowly equilibrating tissues (mg/kg)
    GS = 2  # interstitial (subcutaneous) glucose read by the CGM (mg/kg)
    IL = 3  # insulin mass in the liver (pmol/kg)
    IP = 4  # insulin mass in plasma (pmol/kg)
    X = 5  # insulin action on glucose utilization (pmol/L)
    I1 = 6  # first delay compartment of the insulin signal (pmol/L)
    ID = 7  # delayed insulin signal acting on production (pmol/L)
    IPO = 8  # insulin in the portal vein (pmol/kg)
    Y = 9  # static beta-cell secretion component (pmol/kg/min)
    QSTO1 = 10  # glucose in the stomach, solid phase (mg)
    QSTO2 = 11  # glucose in the stomach, liquid phase (mg)
    QGUT = 12  # glucose in the intestine (mg)
    ISC1 = 13  # subcutaneous insulin, depot / precipitate (pmol/kg)
    ISC2 = 14  # subcutaneous insulin, soluble compartment (pmol/kg)

    N = 15
    NAMES = (
        "Gp",
        "Gt",
        "Gs",
        "Il",
        "Ip",
        "X",
        "I1",
        "Id",
        "Ipo",
        "Y",
        "Qsto1",
        "Qsto2",
        "Qgut",
        "Isc1",
        "Isc2",
    )


# ---------------------------------------------------------------------------
# Parameter vector layout
# ---------------------------------------------------------------------------
class P:
    """Indices of the flat parameter vector passed to the compiled model."""

    BW = 0
    VG = 1
    K1 = 2
    K2 = 3
    VI = 4
    M1 = 5
    M2 = 6
    M4 = 7
    M5 = 8
    M6 = 9
    HEB = 10
    KMAX = 11
    KMIN = 12
    KABS = 13
    B = 14
    D = 15
    F = 16
    KP1 = 17
    KP2 = 18
    KP3 = 19
    KP4 = 20
    KI = 21
    FCNS = 22
    VM0 = 23
    VMX = 24
    KM0 = 25
    P2U = 26
    K = 27
    ALPHA = 28
    BETA = 29
    GAMMA = 30
    KE1 = 31
    KE2 = 32
    KD = 33
    KA1 = 34
    KA2 = 35
    KSC = 36
    GB = 37
    IB = 38
    SB = 39
    SECRETION_ON = 40  # 1.0 = T2D endogenous secretion, 0.0 = switched off (T1D mode)
    N = 41


@dataclass(frozen=True)
class NominalParameters:
    """Nominal (population) parameters. Units follow Dalla Man et al. (2007)."""

    # Glucose kinetics
    VG: float = 1.49  # dL/kg
    k1: float = 0.042  # 1/min
    k2: float = 0.071  # 1/min
    # Insulin kinetics
    VI: float = 0.04  # L/kg
    m1: float = 0.379  # 1/min
    m2: float = 0.673  # 1/min
    m4: float = 0.269  # 1/min
    m5: float = 0.0526  # min*kg/pmol
    HEb: float = 0.6  # basal hepatic insulin extraction (-)
    # Rate of appearance (gut)
    kmax: float = 0.0465  # 1/min (kgri = kmax)
    kmin: float = 0.0076  # 1/min
    kabs: float = 0.023  # 1/min
    b: float = 0.68  # -
    d: float = 0.09  # - (called c in the paper)
    f: float = 0.90  # -
    # Endogenous glucose production
    kp2: float = 0.0007  # 1/min
    kp3: float = 0.005  # mg/kg/min per pmol/L
    kp4: float = 0.0786  # mg/kg/min per pmol/kg
    ki: float = 0.0066  # 1/min
    # Utilization
    Fcns: float = 1.0  # mg/kg/min
    Vmx: float = 0.034  # mg/kg/min per pmol/L
    Km0: float = 466.21  # mg/kg
    p2U: float = 0.084  # 1/min
    # Secretion
    K: float = 0.99  # pmol/kg per mg/dL
    alpha: float = 0.013  # 1/min
    beta: float = 0.05  # pmol/kg/min per mg/dL
    gamma: float = 0.5  # 1/min
    # Renal excretion
    ke1: float = 0.0007  # 1/min
    ke2: float = 269.0  # mg/kg
    # Subcutaneous glargine absorption (project identification, check V2)
    kd: float = 4.1e-4  # 1/min, dissolution of the depot (t1/2 ~ 28 h)
    ka1: float = 0.0  # 1/min, direct absorption from the depot
    ka2: float = 0.015  # 1/min, absorption of soluble insulin
    # Interstitial glucose
    ksc: float = 0.1  # 1/min
    # Nominal T2D basal values kept equal for all patients (proposal IV-C2)
    Ib: float = 50.0  # basal plasma insulin (pmol/L)
    EGPb: float = 2.0  # basal endogenous glucose production (mg/kg/min)

    def replace(self, **changes: float) -> NominalParameters:
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values.update(changes)
        return NominalParameters(**values)


NOMINAL_T2D = NominalParameters()


@dataclass
class PatientParameters:
    """Parameters of one virtual patient.

    Attributes:
        patient_id: Unique identifier, e.g. ``"T2D-017"``.
        body_weight_kg: Body weight ``BW`` (kg).
        fpg_preinsulin: Pre-insulin fasting plasma glucose ``Gb`` (mg/dL).
        rho_si: Insulin-sensitivity factor scaling ``Vmx`` and ``kp3``.
        rho_beta: Beta-cell factor scaling ``beta`` and ``K``.
        nominal: Population parameters.
    """

    patient_id: str
    body_weight_kg: float
    fpg_preinsulin: float
    rho_si: float = 1.0
    rho_beta: float = 1.0
    nominal: NominalParameters = field(default_factory=lambda: NOMINAL_T2D)
    secretion_on: bool = True

    def vector(self) -> np.ndarray:
        """Return the flat parameter vector with derived basal values."""
        return derive_basal(self)

    def initial_state(self) -> np.ndarray:
        """Return the basal (insulin-naive, fasting) equilibrium state."""
        return basal_state(self.vector())


class BasalDerivationError(ValueError):
    """Raised when a parameter combination has no physiological steady state."""


def derive_basal(patient: PatientParameters) -> np.ndarray:
    """Build the parameter vector and solve the basal steady state.

    The steady state of equations (1)-(10) without meals and exogenous insulin
    fixes ``Sb``, ``m6``, ``Gtb``, ``Vm0`` and ``kp1`` (Dalla Man et al., 2007).
    """
    n = patient.nominal
    p = np.zeros(P.N, dtype=np.float64)
    p[P.BW] = patient.body_weight_kg
    p[P.VG] = n.VG
    p[P.K1] = n.k1
    p[P.K2] = n.k2
    p[P.VI] = n.VI
    p[P.M1] = n.m1
    p[P.M2] = n.m2
    p[P.M4] = n.m4
    p[P.M5] = n.m5
    p[P.HEB] = n.HEb
    p[P.KMAX] = n.kmax
    p[P.KMIN] = n.kmin
    p[P.KABS] = n.kabs
    p[P.B] = n.b
    p[P.D] = n.d
    p[P.F] = n.f
    p[P.KP2] = n.kp2
    p[P.KP3] = n.kp3 * patient.rho_si
    p[P.KP4] = n.kp4
    p[P.KI] = n.ki
    p[P.FCNS] = n.Fcns
    p[P.VMX] = n.Vmx * patient.rho_si
    p[P.KM0] = n.Km0
    p[P.P2U] = n.p2U
    p[P.K] = n.K * patient.rho_beta
    p[P.ALPHA] = n.alpha
    p[P.BETA] = n.beta * patient.rho_beta
    p[P.GAMMA] = n.gamma
    p[P.KE1] = n.ke1
    p[P.KE2] = n.ke2
    p[P.KD] = n.kd
    p[P.KA1] = n.ka1
    p[P.KA2] = n.ka2
    p[P.KSC] = n.ksc
    p[P.GB] = patient.fpg_preinsulin
    p[P.IB] = n.Ib
    p[P.SECRETION_ON] = 1.0 if patient.secretion_on else 0.0

    # --- insulin subsystem at basal -------------------------------------
    m3b = n.HEb * n.m1 / (1.0 - n.HEb)
    ipb = n.Ib * n.VI
    ilb = (n.m2 + n.m4) * ipb / n.m1
    sb = (n.m1 + m3b) * ilb - n.m2 * ipb  # pmol/kg/min
    if sb <= 0:
        raise BasalDerivationError("non-positive basal secretion")
    p[P.SB] = sb
    p[P.M6] = n.HEb + n.m5 * sb
    ipob = sb / n.gamma

    # --- glucose subsystem at basal -------------------------------------
    gpb = patient.fpg_preinsulin * n.VG
    eb = n.ke1 * max(0.0, gpb - n.ke2)
    gtb = (n.Fcns + eb - n.EGPb + n.k1 * gpb) / n.k2
    uidb = n.EGPb - n.Fcns - eb
    if gtb <= 0 or uidb <= 0:
        raise BasalDerivationError(
            f"no steady state for Gb={patient.fpg_preinsulin:.1f} mg/dL (Gtb={gtb:.2f}, Uidb={uidb:.3f})"
        )
    p[P.VM0] = uidb * (n.Km0 + gtb) / gtb
    p[P.KP1] = n.EGPb + p[P.KP2] * gpb + p[P.KP3] * n.Ib + p[P.KP4] * ipob
    return p


def basal_state(p: np.ndarray) -> np.ndarray:
    """Equilibrium state for a parameter vector produced by :func:`derive_basal`."""
    x = np.zeros(S.N, dtype=np.float64)
    m1, m2, m4, heb = p[P.M1], p[P.M2], p[P.M4], p[P.HEB]
    gpb = p[P.GB] * p[P.VG]
    eb = p[P.KE1] * max(0.0, gpb - p[P.KE2])
    egpb = p[P.KP1] - p[P.KP2] * gpb - p[P.KP3] * p[P.IB] - p[P.KP4] * p[P.SB] / p[P.GAMMA]
    gtb = (p[P.FCNS] + eb - egpb + p[P.K1] * gpb) / p[P.K2]
    ipb = p[P.IB] * p[P.VI]
    m3b = heb * m1 / (1.0 - heb)
    x[S.GP] = gpb
    x[S.GT] = gtb
    x[S.GS] = gpb
    x[S.IP] = ipb
    x[S.IL] = (m2 + m4) * ipb / m1
    del m3b
    x[S.X] = 0.0
    x[S.I1] = p[P.IB]
    x[S.ID] = p[P.IB]
    x[S.IPO] = p[P.SB] / p[P.GAMMA]
    x[S.Y] = 0.0
    return x
