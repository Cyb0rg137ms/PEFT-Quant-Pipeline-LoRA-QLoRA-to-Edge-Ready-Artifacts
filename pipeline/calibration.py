"""Back-compat shim for :mod:`peftquant.calibration`."""

from peftquant.calibration import (
    ActivationCalibrator,
    LayerCalibrationStats,
    compute_sqnr,
)

__all__ = ["ActivationCalibrator", "LayerCalibrationStats", "compute_sqnr"]
