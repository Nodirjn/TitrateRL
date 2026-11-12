"""Right-hand side of the 15-state T2D model and the fixed-step RK4 integrator.

Equations (1)-(10) and (A1)-(A9) of the proposal. All functions are compiled
with Numba (``nopython`` mode) because one 24-week episode needs about
242 000 integration steps.
"""

from __future__ import annotations

import math

import numpy as np

from titraterl.simulation.parameters import MINUTES_PER_DAY
from titraterl.utils.jit import njit

# Index constants as plain module globals so that Numba can freeze them.
P_ALPHA = 28
P_B = 14
P_BETA = 29
P_BW = 0
P_D = 15
P_F = 16
P_FCNS = 22
P_GAMMA = 30
P_GB = 37
P_HEB = 10
P_IB = 38
P_K = 27
P_K1 = 2
P_K2 = 3
P_KA1 = 34
P_KA2 = 35
P_KABS = 13
P_KD = 33
P_KE1 = 31
P_KE2 = 32
P_KI = 21
P_KM0 = 25
P_KMAX = 11
P_KMIN = 12
P_KP1 = 17
P_KP2 = 18
P_KP3 = 19
P_KP4 = 20
P_KSC = 36
P_M1 = 5
P_M2 = 6
P_M4 = 7
P_M5 = 8
P_M6 = 9
P_P2U = 26
P_SB = 39
P_SECRETION_ON = 40
P_VG = 1
P_VI = 4
P_VM0 = 23
P_VMX = 24
S_GP = 0
S_GS = 2
S_GT = 1
S_I1 = 6
S_ID = 7
S_IL = 3
S_IP = 4
S_IPO = 8
S_ISC1 = 13
S_ISC2 = 14
S_QGUT = 12
S_QSTO1 = 10
S_QSTO2 = 11
S_X = 5
S_Y = 9

# States that are masses and therefore cannot be negative (X, I1, Id, Y can).
_NONNEGATIVE = np.array(
    [True, True, True, True, True, False, False, False, True, False, True, True, True, True, True]
)


@njit
def derivatives(x: np.ndarray, p: np.ndarray, cho_mg_min: float, d_meal_mg: float) -> np.ndarray:
    """Return ``dx/dt`` (per minute).

    Args:
        x: State vector (15,), see :class:`~titraterl.simulation.parameters.S`.
        p: Parameter vector (41,), see :class:`~titraterl.simulation.parameters.P`.
        cho_mg_min: Carbohydrate ingestion rate during this step (mg/min).
        d_meal_mg: Amount ``D`` of the last meal used by the gastric emptying law (mg).
    """
    dx = np.zeros(15)
    bw = p[P_BW]
    vg = p[P_VG]

    gp = max(x[S_GP], 0.0)
    gt = max(x[S_GT], 0.0)
    ip = max(x[S_IP], 0.0)
    il = max(x[S_IL], 0.0)

    # ---- gut: three-compartment model (A1)-(A6) -------------------------
    qsto = x[S_QSTO1] + x[S_QSTO2]
    if d_meal_mg > 0.0:
        aa = 5.0 / (2.0 * d_meal_mg * (1.0 - p[P_B]))
        cc = 5.0 / (2.0 * d_meal_mg * p[P_D])
        kempt = p[P_KMIN] + (p[P_KMAX] - p[P_KMIN]) / 2.0 * (
            math.tanh(aa * (qsto - p[P_B] * d_meal_mg)) - math.tanh(cc * (qsto - p[P_D] * d_meal_mg)) + 2.0
        )
    else:
        kempt = p[P_KMAX]
    dx[S_QSTO1] = -p[P_KMAX] * x[S_QSTO1] + cho_mg_min
    dx[S_QSTO2] = -kempt * x[S_QSTO2] + p[P_KMAX] * x[S_QSTO1]
    dx[S_QGUT] = -p[P_KABS] * x[S_QGUT] + kempt * x[S_QSTO2]
    ra = p[P_F] * p[P_KABS] * x[S_QGUT] / bw

    # ---- insulin signals --------------------------------------------------
    i_plasma = ip / p[P_VI]
    ipo = max(x[S_IPO], 0.0)
    secretion = p[P_SECRETION_ON] * p[P_GAMMA] * ipo  # S(t), eq. (7)

    # ---- endogenous glucose production, eq. (5) ----------------------------
    egp = p[P_KP1] - p[P_KP2] * gp - p[P_KP3] * x[S_ID] - p[P_KP4] * ipo * p[P_SECRETION_ON]
    egp = max(egp, 0.0)

    # ---- utilization and renal excretion, eqs. (6), (A7) -------------------
    uii = p[P_FCNS]
    vm = max(p[P_VM0] + p[P_VMX] * x[S_X], 0.0)
    uid = vm * gt / (p[P_KM0] + gt)
    renal = p[P_KE1] * (gp - p[P_KE2]) if gp > p[P_KE2] else 0.0

    # ---- glucose kinetics, eqs. (1)-(2) -------------------------------------
    dgp = egp + ra - uii - renal - p[P_K1] * gp + p[P_K2] * gt
    dx[S_GP] = dgp
    dx[S_GT] = -uid + p[P_K1] * gp - p[P_K2] * gt

    # ---- subcutaneous glargine, eq. (10) -------------------------------------
    dx[S_ISC1] = -(p[P_KD] + p[P_KA1]) * x[S_ISC1]
    dx[S_ISC2] = p[P_KD] * x[S_ISC1] - p[P_KA2] * x[S_ISC2]
    ra_i = p[P_KA1] * x[S_ISC1] + p[P_KA2] * x[S_ISC2]

    # ---- hepatic extraction and insulin kinetics, eqs. (3), (4), (9) --------
    if p[P_SECRETION_ON] > 0.5:
        he = -p[P_M5] * secretion + p[P_M6]
        he = min(max(he, 0.0), 0.95)
    else:
        he = p[P_HEB]
    m3 = he * p[P_M1] / (1.0 - he)
    dx[S_IL] = -(p[P_M1] + m3) * il + p[P_M2] * ip + secretion
    dx[S_IP] = -(p[P_M2] + p[P_M4]) * ip + p[P_M1] * il + ra_i

    # ---- insulin action, eqs. (6), (A8) -------------------------------------
    dx[S_X] = -p[P_P2U] * x[S_X] + p[P_P2U] * (i_plasma - p[P_IB])
    dx[S_I1] = -p[P_KI] * (x[S_I1] - i_plasma)
    dx[S_ID] = -p[P_KI] * (x[S_ID] - x[S_I1])

    # ---- beta-cell secretion, eqs. (7)-(8) ----------------------------------
    if p[P_SECRETION_ON] > 0.5:
        g = gp / vg
        gdot = dgp / vg
        h = p[P_GB]
        sb = p[P_SB]
        if p[P_BETA] * (g - h) >= -sb:
            dx[S_Y] = -p[P_ALPHA] * (x[S_Y] - p[P_BETA] * (g - h))
        else:
            dx[S_Y] = -p[P_ALPHA] * x[S_Y] - p[P_ALPHA] * sb
        spo = x[S_Y] + p[P_K] * max(gdot, 0.0) + sb
        spo = max(spo, 0.0)
        dx[S_IPO] = -p[P_GAMMA] * ipo + spo

    # ---- interstitial glucose, eq. (A9) ---------------------------------------
    dx[S_GS] = -p[P_KSC] * x[S_GS] + p[P_KSC] * gp
    return dx


@njit
def rk4_step(x: np.ndarray, p: np.ndarray, cho: float, d_meal: float, h: float) -> np.ndarray:
    """One classical fourth-order Runge-Kutta step of length ``h`` minutes."""
    k1 = derivatives(x, p, cho, d_meal)
    k2 = derivatives(x + 0.5 * h * k1, p, cho, d_meal)
    k3 = derivatives(x + 0.5 * h * k2, p, cho, d_meal)
    k4 = derivatives(x + h * k3, p, cho, d_meal)
    out = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Masses cannot become negative; clip round-off at zero.
    for i in range(15):
        if _NONNEGATIVE[i] and out[i] < 0.0:
            out[i] = 0.0
    return out


@njit
def simulate_day(
    x0: np.ndarray,
    p: np.ndarray,
    cho_mg_min: np.ndarray,
    meal_start_mg: np.ndarray,
    d_meal_mg: float,
    injection_minute: int,
    injection_pmol_kg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Integrate one day (1440 one-minute RK4 steps).

    Args:
        x0: State at minute 0 of the day.
        p: Parameter vector.
        cho_mg_min: Carbohydrate ingestion rate for every minute (1440,).
        meal_start_mg: Total amount of a meal at its first minute, 0 elsewhere (1440,).
        d_meal_mg: ``D`` carried over from the previous day.
        injection_minute: Minute of the basal injection (impulse into ``Isc1``).
        injection_pmol_kg: Injected amount (pmol/kg); 0 for no injection.

    Returns:
        ``(x_end, plasma_glucose, interstitial_glucose, d_meal_mg)`` where the two
        glucose arrays hold the value at the start of every minute (mg/dL).
    """
    n = MINUTES_PER_DAY
    g_plasma = np.empty(n)
    g_inter = np.empty(n)
    x = x0.copy()
    d = d_meal_mg
    vg = p[P_VG]
    for k in range(n):
        if k == injection_minute and injection_pmol_kg > 0.0:
            x[S_ISC1] += injection_pmol_kg
        if meal_start_mg[k] > 0.0:
            d = x[S_QSTO1] + x[S_QSTO2] + meal_start_mg[k]
        g_plasma[k] = x[S_GP] / vg
        g_inter[k] = x[S_GS] / vg
        x = rk4_step(x, p, cho_mg_min[k], d, 1.0)
    return x, g_plasma, g_inter, d
