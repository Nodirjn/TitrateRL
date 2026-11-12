"""Measurement models: continuous glucose monitor (CGM) and fingerstick (SMBG).

CGM:
    Sensor error model of Breton and Kovatchev (J Diabetes Sci Technol, 2008)
    as implemented in simglucose: an AR(1) process on a 15-minute grid
    (partial autocorrelation 0.7) mapped through a Johnson SU transform and
    interpolated to the 5-minute CGM grid. Readings are clipped to the sensor
    range 39-400 mg/dL.

SMBG:
    Zero-mean Gaussian error with SD 7.5 mg/dL below 100 mg/dL and 7.5 % above,
    so that about 95 % of readings fall within the ISO 15197:2013 limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CGM_PERIOD_MIN = 5
CGM_SAMPLES_PER_DAY = 288


@dataclass(frozen=True)
class CGMNoiseParams:
    pacf: float = 0.7
    gamma: float = -0.5444
    lam: float = 15.9574
    delta: float = 1.6898
    xi: float = -5.47
    min_mg_dl: float = 39.0
    max_mg_dl: float = 400.0


class CGMSensor:
    """Stateful CGM error generator (the AR state carries over between days)."""

    def __init__(
        self, rng: np.random.Generator, params: CGMNoiseParams | None = None, enabled: bool = True
    ) -> None:
        self.rng = rng
        self.params = params or CGMNoiseParams()
        self.enabled = enabled
        self._e = float(rng.standard_normal())
        self._last_eps = self._johnson(self._e)

    def _johnson(self, e: float | np.ndarray) -> float | np.ndarray:
        p = self.params
        return p.xi + p.lam * np.sinh((e - p.gamma) / p.delta)

    def day_errors(self) -> np.ndarray:
        """Errors for the 288 CGM samples of one day (mg/dL)."""
        n15 = CGM_SAMPLES_PER_DAY // 3  # 96 points on the 15-min grid
        eps = np.empty(n15 + 1)
        eps[0] = self._last_eps
        for k in range(1, n15 + 1):
            self._e = self.params.pacf * (self._e + self.rng.standard_normal())
            eps[k] = self._johnson(self._e)
        self._last_eps = eps[-1]
        t15 = np.arange(n15 + 1) * 15.0
        t5 = np.arange(CGM_SAMPLES_PER_DAY) * 5.0
        return np.interp(t5, t15, eps)

    def measure_day(self, interstitial_mg_dl: np.ndarray) -> np.ndarray:
        """CGM readings every 5 min from the 1-min interstitial glucose trace."""
        g = interstitial_mg_dl[::CGM_PERIOD_MIN]
        err = self.day_errors()  # always drawn, keeps the random stream aligned
        if self.enabled:
            g = g + err
        return np.clip(g, self.params.min_mg_dl, self.params.max_mg_dl)


def smbg_reading(true_mg_dl: float, rng: np.random.Generator, enabled: bool = True) -> float:
    """One fingerstick reading."""
    z = rng.standard_normal()
    if not enabled:
        return float(true_mg_dl)
    sd = 7.5 if true_mg_dl < 100.0 else 0.075 * true_mg_dl
    return float(max(20.0, true_mg_dl + sd * z))
