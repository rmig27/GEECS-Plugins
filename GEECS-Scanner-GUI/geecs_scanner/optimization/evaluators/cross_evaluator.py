"""Minimize custom objective function for symmetric cross.

Classes
-------
SymmetricCrossEvaluator
    Custom objective function to minimize.
"""

from __future__ import annotations
import numpy as np
from geecs_scanner.optimization.base_evaluator import BaseEvaluator


class SymmetricCrossEvaluator(BaseEvaluator):
    """
    Optimizes for a high-intensity, symmetric cross shape on a diagnostic camera.

    Minimizes function:
    Objective = below.
    """

    def __init__(self, calibration: float = 24.4e-3, **kwargs):
        super().__init__(**kwargs)
        self.calibration = calibration

    def compute_objective(self, scalars, bin_number):
        """Objective to minimize."""
        dev = self.primary_device
        x_cross = scalars[f"{dev}_x_peak_location"]
        y_cross = scalars[f"{dev}_y_peak_location"]

        x_img = scalars[f"{dev}_image_peak_img_x"]
        y_img = scalars[f"{dev}_image_peak_img_y"]

        xslice_max = scalars[f"{dev}_x_slicemax"]
        yslice_max = scalars[f"{dev}_y_slicemax"]

        x_intensity_term = scalars[f"{dev}_x_invintensity"]
        y_intensity_term = scalars[f"{dev}_y_invintensity"]

        image_total = scalars[f"{self.primary_device}_image_total"]

        objective = np.sqrt(
            ((x_img - x_cross) ** 2)
            + ((y_img - y_cross) ** 2)
            + (15 * (xslice_max - yslice_max) ** 2)
            + 25 * (x_intensity_term**2)
            + 25 * (y_intensity_term**2)
            + (25 * ((1 / x_intensity_term) - (1 / y_intensity_term)) ** 2)
            + ((1.0 / image_total) ** 2)
        )

        return objective

    def compute_observables(self, scalars, bin_number):
        """Expose pixel + calibrated FWHMs alongside the objective."""
        dev = self.primary_device
        return {
            "x_cross": scalars[f"{dev}_x_peak_location"],
            "y_cross": scalars[f"{dev}_y_peak_location"],
            "x_img": scalars[f"{dev}_image_peak_img_x"],
            "y_img": scalars[f"{dev}_image_peak_img_y"],
            "xslice_max": scalars[f"{dev}_x_slicemax"],
            "yslice_max": scalars[f"{dev}_y_slicemax"],
            "x_intensity_term": scalars[f"{dev}_x_invintensity"],
            "y_intensity_term": scalars[f"{dev}_y_invintensity"],
            "image_total": scalars[f"{self.primary_device}_image_total"],
        }
