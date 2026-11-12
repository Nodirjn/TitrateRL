"""Virtual cohort generator (proposal Section IV-C2).

Each of the N = 150 virtual patients receives

* a pre-insulin fasting glucose ``Gb ~ N(196, 48)`` truncated to [140, 300] mg/dL,
  approximating the baseline of the Treat-to-Target trial;
* a body weight from a truncated normal distribution on [50, 150] kg;
* log-normal factors ``rho_si`` (scales ``Vmx`` and ``kp3``) and ``rho_beta``
  (scales ``beta`` and ``K``).

The cohort is split 90/30/30 into train/validation/test, stratified by the
``rho_si`` tertile. Everything is reproducible from a single seed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from titraterl.simulation.parameters import (
    NOMINAL_T2D,
    BasalDerivationError,
    NominalParameters,
    PatientParameters,
    derive_basal,
)

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class TruncNormal:
    mean: float
    sd: float
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        while True:
            v = rng.normal(self.mean, self.sd)
            if self.low <= v <= self.high:
                return float(v)


@dataclass(frozen=True)
class LogNormal:
    median: float
    sigma: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(self.median * np.exp(self.sigma * rng.standard_normal()))


@dataclass(frozen=True)
class CohortConfig:
    """Distributions of the virtual cohort.

    The defaults are the result of the first calibration pass (check V4,
    ``docs/validation.md``): a ``rho_si`` median of 2.2 brings the final
    Treat-to-Target dose into the trial range (0.4-0.6 U/kg) and keeps every
    patient below the 1 U/kg dose bound, while the week-24 fasting glucose is
    still lower than in the trial. Refined during the Week 4-5 calibration.
    """

    n_patients: int = 150
    n_train: int = 90
    n_val: int = 30
    n_test: int = 30
    seed: int = 2026
    fpg: TruncNormal = TruncNormal(196.0, 48.0, 140.0, 300.0)
    body_weight: TruncNormal = TruncNormal(90.0, 18.0, 50.0, 150.0)
    rho_si: LogNormal = LogNormal(2.2, 0.35)
    rho_beta: LogNormal = LogNormal(1.0, 0.30)
    nominal: NominalParameters = field(default_factory=lambda: NOMINAL_T2D)

    @classmethod
    def from_dict(cls, d: dict) -> CohortConfig:
        d = dict(d)
        for key, typ in (
            ("fpg", TruncNormal),
            ("body_weight", TruncNormal),
            ("rho_si", LogNormal),
            ("rho_beta", LogNormal),
        ):
            if key in d and isinstance(d[key], dict):
                d[key] = typ(**d[key])
        if "nominal" in d and isinstance(d["nominal"], dict):
            d["nominal"] = NOMINAL_T2D.replace(**d["nominal"])
        return cls(**d)


@dataclass
class CohortPatient:
    params: PatientParameters
    split: Split
    si_tertile: int

    @property
    def patient_id(self) -> str:
        return self.params.patient_id


class Cohort:
    """A list of virtual patients with split helpers."""

    def __init__(self, patients: list[CohortPatient], config: CohortConfig | None = None) -> None:
        self.patients = patients
        self.config = config

    def __len__(self) -> int:
        return len(self.patients)

    def __iter__(self):
        return iter(self.patients)

    def split(self, name: Split) -> list[CohortPatient]:
        return [p for p in self.patients if p.split == name]

    def get(self, patient_id: str) -> CohortPatient:
        for p in self.patients:
            if p.patient_id == patient_id:
                return p
        raise KeyError(patient_id)

    # ------------------------------------------------------------------ I/O
    def to_frame(self) -> pd.DataFrame:
        """Table ``patient`` of the data dictionary (docs/data_dictionary.md)."""
        rows = []
        for cp in self.patients:
            pp = cp.params
            rows.append(
                {
                    "patient_id": pp.patient_id,
                    "split": cp.split,
                    "rho_si": pp.rho_si,
                    "rho_beta": pp.rho_beta,
                    "si_tertile": cp.si_tertile,
                    "body_weight_kg": pp.body_weight_kg,
                    "fpg_preinsulin": pp.fpg_preinsulin,
                    "params_json": json.dumps(
                        {
                            "nominal": asdict(pp.nominal),
                            "vector": [float(v) for v in derive_basal(pp)],
                        }
                    ),
                }
            )
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> Path:
        from titraterl.evaluation.logging import validate_patient_table

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_frame()
        validate_patient_table(df)
        df.to_parquet(path, index=False)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Cohort:
        df = pd.read_parquet(path)
        patients = []
        for r in df.itertuples(index=False):
            nominal = NOMINAL_T2D.replace(**json.loads(r.params_json)["nominal"])
            pp = PatientParameters(
                r.patient_id,
                float(r.body_weight_kg),
                float(r.fpg_preinsulin),
                float(r.rho_si),
                float(r.rho_beta),
                nominal,
            )
            patients.append(CohortPatient(pp, r.split, int(r.si_tertile)))
        return cls(patients)


def generate_cohort(config: CohortConfig | None = None) -> Cohort:
    """Generate the virtual cohort deterministically from ``config.seed``."""
    config = config or CohortConfig()
    if config.n_train + config.n_val + config.n_test != config.n_patients:
        raise ValueError("train + val + test must equal n_patients")
    rng = np.random.default_rng(config.seed)
    params: list[PatientParameters] = []
    while len(params) < config.n_patients:
        pid = f"T2D-{len(params) + 1:03d}"
        pp = PatientParameters(
            patient_id=pid,
            body_weight_kg=config.body_weight.sample(rng),
            fpg_preinsulin=config.fpg.sample(rng),
            rho_si=config.rho_si.sample(rng),
            rho_beta=config.rho_beta.sample(rng),
            nominal=config.nominal,
        )
        try:
            derive_basal(pp)
        except BasalDerivationError:
            continue  # resample: no physiological steady state
        params.append(pp)

    # Tertiles of rho_si over the whole cohort
    rho = np.array([p.rho_si for p in params])
    cut1, cut2 = np.quantile(rho, [1 / 3, 2 / 3])
    tertile = np.where(rho <= cut1, 1, np.where(rho <= cut2, 2, 3))

    # Stratified split: shuffle inside each tertile, then deal patients
    split = np.empty(len(params), dtype=object)
    fractions = np.array([config.n_train, config.n_val, config.n_test]) / config.n_patients
    for t in (1, 2, 3):
        idx = np.flatnonzero(tertile == t)
        rng.shuffle(idx)
        n_tr = round(fractions[0] * len(idx))
        n_va = round(fractions[1] * len(idx))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr : n_tr + n_va]] = "val"
        split[idx[n_tr + n_va :]] = "test"
    patients = [CohortPatient(p, s, int(t)) for p, s, t in zip(params, split, tertile, strict=True)]
    return Cohort(patients, config)


_DEFAULT: Cohort | None = None


def default_cohort() -> Cohort:
    """The default 150-patient cohort (cached; seed 2026)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = generate_cohort()
    return _DEFAULT
