"""Minimize beam RMS size on a single camera diagnostic.

Classes
-------
MinRMSIntensityEvaluator
    Minimize the intensity-weighted RMS beam width.
"""

from __future__ import annotations

from geecs_scanner.optimization.base_evaluator import BaseEvaluator


class MinRMSIntensityEvaluator(BaseEvaluator):
    """Minimize intensity-weighted RMS width; kept positive for MINIMIZE direction."""

    def compute_objective(self, scalars, bin_number):
        """Quadrature sum of x and y intensity-weighted RMS widths."""
        dev = self.primary_device
        x_rms = scalars[f"{dev}_x_rms"]
        y_rms = scalars[f"{dev}_y_rms"]

        return x_rms**2 + y_rms**2

    def compute_observables(self, scalars, bin_number):
        dev = self.primary_device
        x_rms = scalars[f"{dev}_x_rms"]
        y_rms = scalars[f"{dev}_y_rms"]

        return {
            "x_rms_px": x_rms,
            "y_rms_px": y_rms,
            "image_peak_value": scalars[
                f"{dev}_image_peak_value"
            ],  # not sure if needed
            "rms_radius_px": (x_rms**2 + y_rms**2) ** 0.5,
        }
