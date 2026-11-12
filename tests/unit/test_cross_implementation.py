"""Cross-implementation test against simglucose (V1).

With endogenous secretion switched off and simglucose parameters, the shared
subsystems (glucose, insulin, gut, subcutaneous insulin, CGM compartment) must
reproduce simglucose's right-hand side. simglucose's ``model`` function is
loaded from its source file so that its package ``__init__`` (which needs the
legacy ``gym`` package) is not imported. The test is skipped when simglucose is
not installed.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from titraterl.simulation.model import derivatives
from titraterl.simulation.parameters import P, S

spec = importlib.util.find_spec("simglucose")
pytestmark = pytest.mark.skipif(spec is None, reason="simglucose not installed")


def _load_simglucose_model():
    pkg = Path(spec.submodule_search_locations[0])
    src = (pkg / "patient" / "t1dpatient.py").read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "T1DPatient")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "model")
    fn.decorator_list = []
    ns = {"np": np, "logger": logging.getLogger("simglucose")}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "t1dpatient", "exec"), ns)
    params = pd.read_csv(pkg / "params" / "vpatient_params.csv")
    return ns["model"], params


SG_TO_OURS = {
    0: S.QSTO1,
    1: S.QSTO2,
    2: S.QGUT,
    3: S.GP,
    4: S.GT,
    5: S.IP,
    6: S.X,
    7: S.I1,
    8: S.ID,
    9: S.IL,
    10: S.ISC1,
    11: S.ISC2,
    12: S.GS,
}


def _our_params(r) -> np.ndarray:
    p = np.zeros(P.N)
    mapping = {
        P.BW: "BW",
        P.VG: "Vg",
        P.K1: "k1",
        P.K2: "k2",
        P.VI: "Vi",
        P.M1: "m1",
        P.M2: "m2",
        P.M4: "m4",
        P.M5: "m5",
        P.HEB: "HEb",
        P.KMAX: "kmax",
        P.KMIN: "kmin",
        P.KABS: "kabs",
        P.B: "b",
        P.D: "d",
        P.F: "f",
        P.KP1: "kp1",
        P.KP2: "kp2",
        P.KP3: "kp3",
        P.KI: "ki",
        P.FCNS: "Fsnc",
        P.VM0: "Vm0",
        P.VMX: "Vmx",
        P.KM0: "Km0",
        P.P2U: "p2u",
        P.KE1: "ke1",
        P.KE2: "ke2",
        P.KD: "kd",
        P.KA1: "ka1",
        P.KA2: "ka2",
        P.KSC: "ksc",
        P.GB: "Gb",
        P.IB: "Ib",
    }
    for idx, name in mapping.items():
        p[idx] = r[name]
    p[P.SECRETION_ON] = 0.0
    return p


@pytest.mark.parametrize("name", ["adult#001", "adolescent#003", "child#005"])
@pytest.mark.parametrize("cho_g_min", [0.0, 5.0])
def test_rhs_matches_simglucose(name, cho_g_min):
    sg_model, table = _load_simglucose_model()
    r = table.set_index("Name").loc[name]
    sg_params = SimpleNamespace(**r.to_dict())
    x_sg = np.array([r[f"x0_{i:2d}"] for i in range(1, 14)], dtype=float)
    rng = np.random.default_rng(0)
    x_sg = x_sg * (1 + 0.2 * rng.random(13))  # off-equilibrium state
    x_sg[0] += 20000.0
    last_qsto, last_food = 15000.0, 50.0
    basal_u_min = r["u2ss"] * r["BW"] / 6000.0
    dx_sg = sg_model(
        0.0, x_sg, SimpleNamespace(CHO=cho_g_min, insulin=basal_u_min), sg_params, last_qsto, last_food
    )

    x = np.zeros(S.N)
    for i_sg, i_ours in SG_TO_OURS.items():
        x[i_ours] = x_sg[i_sg]
    p = _our_params(r)
    dx = derivatives(x, p, cho_g_min * 1000.0, last_qsto + last_food * 1000.0)
    dx[S.ISC1] += r["u2ss"]  # continuous basal infusion (pmol/kg/min) used by simglucose
    for i_sg, i_ours in SG_TO_OURS.items():
        assert dx[i_ours] == pytest.approx(dx_sg[i_sg], rel=1e-9, abs=1e-9), (name, i_sg)
