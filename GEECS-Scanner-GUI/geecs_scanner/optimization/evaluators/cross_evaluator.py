from __future__ import annotations
import numpy as np
from geecs_scanner.optimization.base_evaluator import BaseEvaluator


class SymmetricCrossEvaluator(BaseEvaluator):
    """
    Optimizes for a high-intensity, symmetric cross shape on a diagnostic camera.

    Minimizes function:
    Objective = -TotalCounts + (Symmetry_Weight * |Width_X - Width_Y|)
    """

    def compute_objective(self, scalars, bin_number):
        dev = self.primary_device
        counts_key = f"{dev}_image_total"

        total_counts = scalars[counts_key]

        width_x = scalars.get(f"{dev}_x_fwhm")
        width_y = scalars.get(f"{dev}_y_fwhm")

        asymmetry = abs(width_x - width_y)

        # Since Xopt minimizes, make total_counts negative
        # Leave asymmetry positive (drives the difference down)
        symmetry_weight = 1000.0

        objective_value = -total_counts + (symmetry_weight * asymmetry)
        return float(objective_value)

    def compute_observables(self, scalars, bin_number):
        dev = self.primary_device

        width_x = scalars.get(f"{dev}_x_fwhm", float("nan"))
        width_y = scalars.get(f"{dev}_y_fwhm", float("nan"))

        return {
            "image_total": scalars.get(f"{dev}_image_total", 0.0),
            "x_spread": width_x,
            "y_spread": width_y,
            "asymmetry_delta": abs(width_x - width_y)
            if not np.isnan(width_x)
            else float("nan"),
            "x_CoM": scalars.get(f"{dev}_x_CoM", float("nan")),
            "y_CoM": scalars.get(f"{dev}_y_CoM", float("nan")),
        }
