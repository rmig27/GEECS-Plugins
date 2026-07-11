# Rachel test

"""Minimize beam FWHM and center beam on a single camera diagnostic.

Classes
-------
BeamSizeAndCenterEvaluator
    Minimize ``w1 * sqrt[(x - x0)² + (y - y0)²] + w2 * ((x_fwhm * cal)² + (y_fwhm * cal)²)`` on whichever camera
    diagnostic is listed first in ``analyzers``.
"""

from __future__ import annotations

from geecs_scanner.optimization.base_evaluator import BaseEvaluator


class BeamSizeAndCenterEvaluator(BaseEvaluator):
    """Minimize quadrature sum of FWHMs (in physical units) and distance from center (x0, y0)."""

    def __init__(self, calibration: float = 24.4e-3, **kwargs):
        super().__init__(**kwargs)
        self.calibration = calibration
        self.target_x = target_x
        self.target_y = target_y
        self.weight_size = weight_size
        self.weight_position = weight_position

    def get_beam_param(self, scalars, dev):
        x_pos = scalars[f"{dev}_x_centroid"]
        y_pos = scalars[f"{dev}_y_centroid"]
        x_fwhm = scalars[f"{dev}_x_fwhm"]
        y_fwhm = scalars[f"{dev}_y_fwhm"]

        dx_px = x_pos - self.target_x
        dy_px = y_pos - self.target_y

        dx_units = dx_px * self.calibration
        dy_units = dy_px * self.calibration
        x_fwhm_units = x_fwhm * self.calibration
        y_fwhm_units = y_fwhm * self.calibration

        return {
            "dx_units": dx_units,
            "dy_units": dy_units,
            "x_fwhm_units": x_fwhm_units,
            "y_fwhm_units": y_fwhm_units,
            "dx_px": dx_px,
            "dy_px": dy_px,
            "x_fwhm_px": x_fwhm,
            "y_fwhm_px": y_fwhm,
        }

    def compute_objective(self, scalars, bin_number):
        """Quadrature sum of calibrated FWHMs."""
        p = self.get_beam_param(scalars, self.primary_device)
        dist_sq = p["dx_units"] ** 2 + p["dy_units"] ** 2
        size_sq = p["x_fwhm_units"] ** 2 + p["y_fwhm_units"] ** 2
        return (self.weight_position * dist_sq) + (self.self.weight_size * size_sq)

    def compute_observables(self, scalars, bin_number):
        p = self.get_beam_param(scalars, self.primary_device)
        dist_sq = p["dx_units"] ** 2 + p["dy_units"] ** 2
        size_sq = p["x_fwhm_units"] ** 2 + p["y_fwhm_units"] ** 2
        return {
            "centroid_x_px": p["x_px"],
            "centroid_y_px": p["y_px"],
            "x_fwhm_px": p["x_fwhm_px"],
            "y_fwhm_px": p["y_fwhm_px"],
            "dist_from_target_units": (dist_sq) ** 0.5,
            "size_fwhm_quad_units": (size_sq) ** 0.5,
            "objective_val": (self.self.weight_position * dist_sq)
            + (self.self.weight_position * size_sq),
        }
