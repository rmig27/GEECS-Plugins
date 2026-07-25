"""Basic beam statistics utilities.

Provides data structures and functions for computing fundamental beam profile
statistics from images: projection-based stats (CoM, rms, fwhm, peak_location)
along x, y, and ±45° axes, plus image-level totals.

For advanced/optional algorithms (e.g., slope metrics), see separate modules
such as :mod:`image_analysis.algorithms.beam_slopes`.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Set, Tuple
import logging

import numpy as np
from scipy import ndimage

from image_analysis.algorithms.basic_line_stats import LineBasicStats

logger = logging.getLogger(__name__)


class ProjectionStats(NamedTuple):
    """Statistics of a 1‑D projection of a beam image.

    Attributes
    ----------
    CoM : float
        Center‑of‑mass of the projection.
    rms : float
        Root‑mean‑square width of the projection.
    fwhm : float
        Full‑width at half‑maximum of the projection.
    peak_location : float
        Index of the maximum value in the projection.
    """

    # CoM: Optional[float,np.nan]
    CoM: float
    rms: float
    fwhm: float
    peak_location: float
    slicemax: float | None = np.nan
    invintensity: float | None = np.nan


class ImageStats(NamedTuple):
    """Overall statistics of a 2‑D beam image.

    Attributes
    ----------
    total : float
        Sum of all pixel values (total intensity).
    peak_value : float
        Maximum pixel value in the image.
    """

    total: float
    peak_value: float

    # ===============================================================
    # ADDED BY RACHEL
    peak_img_x: float
    peak_img_y: float
    # ===============================================================


class BeamStats(NamedTuple):
    """Container for beam statistics of an image.

    Attributes
    ----------
    image : ImageStats
        Global image statistics.
    x : ProjectionStats
        Statistics of the horizontal (x-axis) projection.
    y : ProjectionStats
        Statistics of the vertical (y-axis) projection.
    x_45 : ProjectionStats
        Statistics of the +45° "column-after-rotation" projection
        (implemented via NW–SE diagonal sums with no resampling).
    y_45 : ProjectionStats
        Statistics of the +45° "row-after-rotation" projection
        (implemented via NE–SW anti-diagonal sums with no resampling).
    """

    image: ImageStats
    x: ProjectionStats
    y: ProjectionStats
    x_45: ProjectionStats
    y_45: ProjectionStats


def _diag_projection(img: np.ndarray) -> np.ndarray:
    """NW–SE diagonal sums (equivalent to column projection after +45° rotate)."""
    img = np.asarray(img, dtype=float)
    h, w = img.shape
    return np.array([np.diag(img, k=k).sum() for k in range(-(h - 1), w)])


def _antidiag_projection(img: np.ndarray) -> np.ndarray:
    """NE–SW anti-diagonal sums (equivalent to row projection after +45° rotate)."""
    img = np.asarray(img, dtype=float)
    flipped = np.fliplr(img)
    h, w = flipped.shape
    return np.array([np.diag(flipped, k=k).sum() for k in range(-(h - 1), w)])


def _projection_to_line_data(projection: np.ndarray, offset: int = 0) -> np.ndarray:
    """Convert a 1D projection array to Nx2 format for LineBasicStats.

    Parameters
    ----------
    projection : np.ndarray
        1D projection array
    offset : int
        Coordinate of the first element in the projection within the parent
        (full-image) coordinate system.  Defaults to 0 (no offset).

    Returns
    -------
    np.ndarray
        Nx2 array where column 0 is coordinates and column 1 is projection values
    """
    x_coords = np.arange(len(projection)) + offset
    return np.column_stack([x_coords, projection])


def _line_stats_to_projection_stats(line_stats: LineBasicStats) -> ProjectionStats:
    """Extract ProjectionStats fields from LineBasicStats.

    Parameters
    ----------
    line_stats : LineBasicStats
        Complete line statistics

    Returns
    -------
    ProjectionStats
        Projection statistics with 4 fields (subset of LineBasicStats)
    """
    return ProjectionStats(
        CoM=line_stats.CoM,
        rms=line_stats.rms,
        fwhm=line_stats.fwhm,
        peak_location=line_stats.peak_location,
        # ============================================================================
        # ADDED BY RACHEL
        slicemax=line_stats.slice_max,
        invintensity=line_stats.inv_intensity_term,
        # ============================================================================
    )


def beam_profile_stats(
    img: np.ndarray,
    roi_offset: Tuple[int, int] = (0, 0),
) -> BeamStats:
    """Compute basic beam profile statistics from a 2-D image.

    Computes projection statistics (CoM, rms, fwhm, peak_location) along
    four axes (x, y, x_45, y_45) and image-level totals (total, peak_value).

    Position statistics (CoM, peak_location) along the x and y axes are
    expressed in the coordinate system of the *parent* image when
    ``roi_offset`` is supplied.  Width statistics (rms, fwhm) are unaffected
    by the offset.  The 45° diagonal projections are always in local
    (cropped-image) index space.

    Parameters
    ----------
    img : np.ndarray
        2D image array (may already be ROI-cropped).
    roi_offset : tuple of int, optional
        ``(x_offset, y_offset)`` — pixel coordinate of the top-left corner of
        ``img`` within the full sensor frame.  Use ``(roi.x_min, roi.y_min)``
        when ``img`` is the result of an ROI crop.  Defaults to ``(0, 0)``
        (no offset — stats are in the local frame).

    Returns
    -------
    BeamStats
        Beam statistics including image-level stats and 4 projection stats.
    """
    img = np.asarray(img, dtype=float)
    total_counts = img.sum()

    # RACHEL ADDED last two np.nan below
    nan_proj = ProjectionStats(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    if total_counts <= 0:
        logger.warning(
            "beam_profile_stats: Image has non-positive total intensity. Returning NaNs."
        )
        nan_img = ImageStats(
            total=total_counts, peak_value=np.nan, peak_x=np.nan, peak_y=np.nan
        )
        return BeamStats(
            image=nan_img, x=nan_proj, y=nan_proj, x_45=nan_proj, y_45=nan_proj
        )

    x_offset, y_offset = roi_offset

    # ====================================================================
    # ADDED BY RACHEL
    # threshold = 500
    # img_treated = np.where(img > threshold, 0, img)
    # blurred = ndimage.median_filter(img_treated, size=3)
    # difference = img_treated  - blurred
    # threshold = 1*np.std(difference)
    # hot_pixels = np.nonzero((np.abs(difference[1:-1,1:-1])>threshold))
    # hot_pixels = np.array(hot_pixels) + 1 #Because we ignored the first row and first column
    # fixed_image = np.copy(img_treated) #This is the image with the hot pixels removed
    # for y,x in zip(hot_pixels[0],hot_pixels[1]):
    #     fixed_image[y,x]=blurred[y,x]

    # img_for_peak = img.copy()
    # img_for_peak = img[2:, 2:].copy()

    # # 2. Threshold top 5% of pixels
    # threshold = np.percentile(img_for_peak, 95)
    # mask = img_for_peak > threshold

    # # 3. Morphological operations (cleaning hot pixels / small noise)
    # # kernel = np.ones((10, 10))
    # kernel = np.ones((3, 3))
    # eroded = ndimage.binary_erosion(mask, kernel)
    # dilated = ndimage.binary_dilation(eroded, kernel)

    # # 4. Connected component labeling
    # labels, num_labels = ndimage.label(dilated)

    # if num_labels > 0:
    #     # Find the largest connected region (the main beam spot)
    #     label_sizes = np.bincount(labels.ravel())
    #     largest_label = np.argmax(label_sizes[1:]) + 1

    #     # Zero out everything outside the main beam region
    #     img_treated = img_for_peak * (labels == largest_label)
    # else:
    #     # # Fallback if thresholding finds no blobs
    #     # img_treated = img_for_peak
    #     # If erosion was too aggressive, use median filter as backup cleanup
    #     img_treated = ndimage.median_filter(img_for_peak, size=3)

    # # 5. Peak finding on the treated image
    # # cropped_peak = np.unravel_index(np.argmax(img_treated), img_treated.shape)

    # # GEMINNI:
    # # 4. Smooth img_treated to suppress hot pixels INSIDE the main beam
    # img_treated = ndimage.gaussian_filter(img_treated, sigma=2.0)

    # # 5. Peak finding directly on img_treated
    # cropped_peak = np.unravel_index(np.argmax(img_treated), img_treated.shape)

    # # Shift coordinates back by +2 to match the original uncropped `img` frame
    # peak_coords = (cropped_peak[0] + 2, cropped_peak[1] + 2)

    img_for_peak = img[2:, 2:].copy()

    # 1. Identify single-pixel spikes using a local median comparison
    median_bg = ndimage.median_filter(img_for_peak, size=3)
    diff = img_for_peak - median_bg

    # 2. Flag pixels that spike significantly above their local neighborhood
    threshold = 3.0 * np.std(diff)
    hot_pixel_mask = diff > threshold

    # 3. Repair ONLY the hot pixels by replacing them with the local median
    img_treated = img_for_peak.copy()
    img_treated[hot_pixel_mask] = median_bg[hot_pixel_mask]

    # 4. Peak finding on the clean image (no zeroed-out background!)
    cropped_peak = np.unravel_index(np.argmax(img_treated), img_treated.shape)
    peak_coords = (cropped_peak[0] + 2, cropped_peak[1] + 2)

    # --- Your Print Statements ---
    print("#### basic beam stats function")
    print(img_treated)
    print(np.sum(img_treated))
    print(img.shape)
    print(np.argmax(img_treated))
    print(peak_coords)
    print(peak_coords[1], peak_coords[0])  # switch 0 and 1, bc in (y,x) form

    x_proj = img_treated.sum(axis=0)
    print("X max index:", np.argmax(x_proj))
    print("Value at 421:", x_proj[421])
    print("Values around 421:", x_proj[418:425])
    # ====================================================================

    # ====================================================================
    # Standard x/y projections — offset coordinates so stats are in the
    # full-image (global) coordinate system.
    x_stats = _line_stats_to_projection_stats(
        LineBasicStats(
            # line_data=_projection_to_line_data(img.sum(axis=0), offset=x_offset)
            line_data=_projection_to_line_data(
                img_treated.sum(axis=0),
                #    offset=x_offset)
                offset=x_offset + 2,
            )  # Account for img[2:, 2:] crop
        )
    )
    y_stats = _line_stats_to_projection_stats(
        LineBasicStats(
            # line_data=_projection_to_line_data(img.sum(axis=1), offset=y_offset)
            line_data=_projection_to_line_data(
                img_treated.sum(axis=1),
                #    offset=y_offset)
                offset=y_offset + 2,
            )  # Account for img[2:, 2:] crop
        )
    )

    # 45° projections — diagonal index space; no simple global offset applies.
    x45_stats = _line_stats_to_projection_stats(
        # LineBasicStats(line_data=_projection_to_line_data(_diag_projection(img)))
        LineBasicStats(
            line_data=_projection_to_line_data(_diag_projection(img_treated))
        )
    )
    y45_stats = _line_stats_to_projection_stats(
        # LineBasicStats(line_data=_projection_to_line_data(_antidiag_projection(img)))
        LineBasicStats(
            line_data=_projection_to_line_data(_antidiag_projection(img_treated))
        )
    )

    # ====================================================================

    # ====================================================================

    return BeamStats(
        # image=ImageStats(total=total_counts, peak_value=float(np.max(img)), peak_img_x=float(peak_coords[0]), peak_img_y=float(peak_coords[1])),
        image=ImageStats(
            total=total_counts,
            peak_value=float(np.max(img_treated)),  # peak_value=float(np.max(img))
            # GEMINI: By assigning peak_img_x = peak_coords[0], you accidentally put the Y peak value into x_img
            peak_img_x=float(peak_coords[1]),  # Column index (X)
            peak_img_y=float(peak_coords[0]),  # Row index (Y)
        ),
        x=x_stats,
        y=y_stats,
        x_45=x45_stats,
        y_45=y45_stats,
    )


def flatten_beam_stats(
    stats: BeamStats,
    include: Optional[Set[str]] = None,
) -> dict[str, float]:
    """Flatten a :class:`BeamStats` instance into a dictionary of bare-key scalars.

    Emits keys of the form ``"{section}_{field}"`` (e.g. ``"image_total"``,
    ``"x_CoM"``, ``"y_fwhm"``) with no prefix or suffix. Naming/disambiguation
    across analyzers is ScanAnalysis's responsibility per issue #412 — the
    scan-side wrapper applies ``metric_prefix`` and ``metric_suffix`` when
    storing per-shot results.

    Parameters
    ----------
    stats : BeamStats
        The beam statistics to flatten.
    include : set of str, optional
        If provided, only emit entries whose key is in this set. ``None``
        (the default) emits all entries.

    Returns
    -------
    dict[str, float]
        Dictionary mapping bare field names to values.
    """
    flat: dict[str, float] = {}
    for field in stats._fields:
        nested = getattr(stats, field)
        for k, v in nested._asdict().items():
            fragment = f"{field}_{k}"
            if include is not None and fragment not in include:
                continue
            flat[fragment] = v
    return flat


# ---------------------------------------------------------------------------
# Backward compatibility: re-export functions moved to basic_line_stats
# ---------------------------------------------------------------------------
_MOVED_TO_BASIC_LINE_STATS = {
    "compute_center_of_mass",
    "compute_rms",
    "compute_fwhm",
    "compute_peak_location",
}


def __getattr__(name):
    """Lazy re-exports for backward compatibility."""
    if name in _MOVED_TO_BASIC_LINE_STATS:
        import warnings

        from image_analysis.algorithms import basic_line_stats

        warnings.warn(
            f"{name} has been moved to image_analysis.algorithms.basic_line_stats. "
            f"Please update your imports. This backward-compatible re-export will be "
            f"removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(basic_line_stats, name)
