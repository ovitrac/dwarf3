"""
RGB color processing for DWARF 3 astrophotography.

This module implements the proper LRGB workflow for producing publication-quality
color images from DWARF 3 acquisitions:

1. Stack RGB correctly (linear, no per-channel normalization)
2. Calibrate colors (background neutralization + white balance)
3. Combine with processed luminance (LRGB technique)

Key principles:
- All RGB operations on LINEAR data (no stretch before calibration)
- White balance using star colors (neutral reference)
- LRGB combination: use processed luminance for structure, RGB for chrominance
- SCNR green reduction for residual color artifacts

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


# =============================================================================
# Color Correction Matrix (CCM) Presets
# =============================================================================
# Row sums should be 1.0 to preserve luminance on neutral tones.
# These matrices transform Camera RGB to display-ready sRGB.

CCM_PRESETS: dict[str, np.ndarray] = {
    # Identity matrix (no color correction)
    "neutral": np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32),

    # Generic "Rich" for Sony IMX sensors (high saturation)
    # Subtracts channel crosstalk to restore color separation
    "rich": np.array([
        [ 1.45, -0.35, -0.10],
        [-0.15,  1.35, -0.20],
        [-0.05, -0.25,  1.30]
    ], dtype=np.float32),

    # More aggressive color separation (vivid colors)
    "vivid": np.array([
        [ 1.60, -0.40, -0.20],
        [-0.20,  1.60, -0.40],
        [-0.10, -0.30,  1.40]
    ], dtype=np.float32),

    # Ha-emission focused: Protects Red channel from excessive subtraction
    # Useful when Red signal is dominant/narrowband (emission nebulae)
    "ha_emission": np.array([
        [ 1.10, -0.10,  0.00],
        [-0.10,  1.10,  0.00],
        [ 0.00,  0.00,  1.00]
    ], dtype=np.float32),
}


@dataclass
class ColorCalibration:
    """Color calibration parameters."""

    # Background offsets (subtracted from each channel)
    bg_offset_r: float = 0.0
    bg_offset_g: float = 0.0
    bg_offset_b: float = 0.0

    # White balance gains (multiplied after background subtraction)
    wb_gain_r: float = 1.0
    wb_gain_g: float = 1.0
    wb_gain_b: float = 1.0

    # Reference star color (for verification)
    ref_star_color_r: float = 1.0
    ref_star_color_g: float = 1.0
    ref_star_color_b: float = 1.0

    # Number of reference stars used
    n_reference_stars: int = 0

    # Quality metrics
    bg_uniformity: float = 0.0  # Lower is better
    color_balance_residual: float = 0.0  # Lower is better


@dataclass
class LRGBResult:
    """Result of LRGB combination."""

    lrgb: np.ndarray  # Combined LRGB image (H, W, 3)
    luminance: np.ndarray  # Luminance used (H, W)
    chrominance: np.ndarray  # Chrominance from RGB (H, W, 3)
    method: str  # Combination method used


def apply_bayer_compensation(
    rgb: np.ndarray,
    pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG"] = "RGGB",
    r_gain: float | None = None,
    b_gain: float | None = None,
    auto: bool = True,
) -> np.ndarray:
    """
    Apply Bayer pattern compensation to balance RGB channels.

    OSC (One-Shot Color) cameras with Bayer patterns have 2x green pixels
    compared to red and blue in each 2x2 superpixel. After proper debayering,
    the channel imbalance is typically small (~3-5%).

    This function can either auto-calculate compensation from channel medians,
    or apply fixed gains.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image (H, W, 3), linear data.
    pattern : str, default "RGGB"
        Bayer pattern. All standard patterns have 2 G pixels per 2x2.
    r_gain : float, optional
        Manual R channel gain. If None and auto=False, uses 1.0.
    b_gain : float, optional
        Manual B channel gain. If None and auto=False, uses 1.0.
    auto : bool, default True
        If True, auto-calculate gains from channel medians to balance RGB.
        If False, use provided r_gain/b_gain or defaults.

    Returns
    -------
    np.ndarray
        Compensated RGB image (H, W, 3).

    Notes
    -----
    With properly implemented bilinear debayering, the channel imbalance
    is typically only ~3-5% (G slightly higher than R and B).

    Auto mode (default) calculates gains to make channel medians equal,
    which works well for typical astronomical images with neutral sky.

    For manual calibration, image a gray card or use sun-type star colors.
    """
    if auto and r_gain is None and b_gain is None:
        # Auto-calculate gains from channel medians
        r_med = np.median(rgb[:, :, 0])
        g_med = np.median(rgb[:, :, 1])
        b_med = np.median(rgb[:, :, 2])

        r_gain = g_med / r_med if r_med > 0 else 1.0
        b_gain = g_med / b_med if b_med > 0 else 1.0

        logger.debug(
            "Auto Bayer compensation: R*%.3f, B*%.3f (medians: R=%.1f, G=%.1f, B=%.1f)",
            r_gain, b_gain, r_med, g_med, b_med
        )
    else:
        # Use provided or default gains
        if r_gain is None:
            r_gain = 1.0
        if b_gain is None:
            b_gain = 1.0

    result = rgb.copy().astype(np.float32)
    result[:, :, 0] *= r_gain
    result[:, :, 2] *= b_gain

    logger.debug(
        "Bayer compensation applied: R*%.2f, B*%.2f (pattern=%s)",
        r_gain, b_gain, pattern
    )

    return result


def apply_ccm(
    rgb: np.ndarray,
    matrix: np.ndarray | str | list | None = "rich",
    saturation_boost: float = 1.0,
    clip_negatives: bool = True,
    preserve_luminance: bool = True,
) -> np.ndarray:
    """
    Apply Color Correction Matrix (CCM) to convert Camera RGB to sRGB.

    This step is essential for accurate color reproduction. Without it,
    colors appear desaturated because sensor spectral response does not
    match sRGB primaries. The CCM "un-mixes" channel crosstalk.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image (H, W, 3), linear data, white-balanced.
    matrix : np.ndarray | str | list, default "rich"
        3x3 Color Correction Matrix.
        Can be a preset name: "neutral", "rich", "vivid", "ha_emission".
        Or a 3x3 array/list for custom matrices.
    saturation_boost : float, default 1.0
        Scalar multiplier for matrix deviation from identity.
        Values > 1.0 increase color separation (more saturated).
        Values < 1.0 reduce color separation (less saturated).
    clip_negatives : bool, default True
        Clip output values < 0 to 0 (essential for high saturation matrices).
    preserve_luminance : bool, default True
        Normalize matrix rows to sum to 1.0 to preserve neutral gray levels.

    Returns
    -------
    np.ndarray
        Color-corrected RGB image (Linear sRGB).

    Examples
    --------
    >>> # Use "rich" preset (default)
    >>> rgb_corrected = apply_ccm(rgb_wb)

    >>> # Use "ha_emission" for emission nebulae
    >>> rgb_corrected = apply_ccm(rgb_wb, matrix="ha_emission")

    >>> # Custom matrix
    >>> custom = [[1.5, -0.3, -0.2], [-0.1, 1.4, -0.3], [0.0, -0.2, 1.2]]
    >>> rgb_corrected = apply_ccm(rgb_wb, matrix=custom)

    Notes
    -----
    The CCM should be applied AFTER white balance and BEFORE non-linear
    stretch (asinh, gamma). The input must be white-balanced so that
    neutral objects have R ≈ G ≈ B.

    Row sums of 1.0 ensure that neutral colors (R=G=B) remain neutral
    after transformation. This is verified/enforced when preserve_luminance=True.
    """
    # 1. Resolve matrix from preset name or array
    if isinstance(matrix, str):
        if matrix not in CCM_PRESETS:
            available = list(CCM_PRESETS.keys())
            raise ValueError(f"Unknown CCM preset: '{matrix}'. Available: {available}")
        ccm = CCM_PRESETS[matrix].copy()
        matrix_name = matrix
    elif matrix is None:
        ccm = CCM_PRESETS["rich"].copy()
        matrix_name = "rich"
    else:
        ccm = np.array(matrix, dtype=np.float32)
        matrix_name = "custom"

    if ccm.shape != (3, 3):
        raise ValueError(f"CCM must be 3x3, got shape {ccm.shape}")

    # 2. Apply saturation boost (deviation from identity)
    if saturation_boost != 1.0:
        # M_new = I + boost * (M - I)
        identity = np.eye(3, dtype=np.float32)
        ccm = identity + saturation_boost * (ccm - identity)
        logger.debug("CCM saturation boost applied: %.2fx", saturation_boost)

    # 3. Validate/normalize row sums for luminance preservation
    if preserve_luminance:
        row_sums = ccm.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1.0, row_sums)

        if not np.allclose(row_sums, 1.0, atol=0.01):
            logger.warning(
                "CCM row sums deviate from 1.0: [%.3f, %.3f, %.3f]. Normalizing.",
                row_sums[0, 0], row_sums[1, 0], row_sums[2, 0]
            )
            ccm = ccm / row_sums
        else:
            logger.debug("CCM row sums verified (preserve_luminance=True)")

    # 4. Apply matrix transformation
    h, w, c = rgb.shape
    rgb_flat = rgb.reshape(-1, 3).astype(np.float32)

    # Output = Input @ Matrix.T (each row of input multiplied by columns of M)
    corrected_flat = rgb_flat @ ccm.T
    corrected = corrected_flat.reshape(h, w, c)

    # 5. Clip negative values (CCM can produce negatives for out-of-gamut colors)
    if clip_negatives:
        n_negative = np.sum(corrected < 0)
        if n_negative > 0:
            logger.debug("Clipping %d negative values (%.2f%%)",
                        n_negative, 100 * n_negative / corrected.size)
        corrected = np.maximum(corrected, 0)

    logger.info("Applied CCM: %s (saturation_boost=%.2f)", matrix_name, saturation_boost)

    return corrected.astype(np.float32)


def compute_common_footprint(
    rgb_stack: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """
    Compute the common valid footprint across an RGB stack.

    This identifies pixels that have valid data in all frames after alignment,
    excluding the black borders from warping.

    Parameters
    ----------
    rgb_stack : np.ndarray
        RGB image (H, W, 3).
    threshold : float, default 0.01
        Minimum value to consider a pixel valid (fraction of max).

    Returns
    -------
    np.ndarray
        Binary mask (True = valid in all channels).
    """
    # Check each channel for valid data
    max_val = np.max(rgb_stack)
    if max_val <= 0:
        return np.ones(rgb_stack.shape[:2], dtype=bool)

    abs_threshold = threshold * max_val

    # Valid if all channels have data above threshold
    valid_r = rgb_stack[:, :, 0] > abs_threshold
    valid_g = rgb_stack[:, :, 1] > abs_threshold
    valid_b = rgb_stack[:, :, 2] > abs_threshold

    # Common footprint
    footprint = valid_r & valid_g & valid_b

    # Erode slightly to remove edge artifacts
    struct = ndimage.generate_binary_structure(2, 1)
    footprint = ndimage.binary_erosion(footprint, struct, iterations=3)

    logger.debug(
        "Common footprint: %.1f%% valid",
        100 * np.mean(footprint),
    )

    return footprint


def estimate_sky_background_rgb(
    rgb: np.ndarray,
    object_mask: np.ndarray | None = None,
    sigma_clip: float = 2.5,
    n_iters: int = 5,
    method: Literal["corners", "masked"] = "corners",
) -> tuple[float, float, float]:
    """
    Estimate sky background level for each RGB channel.

    IMPORTANT: For images with extended objects (galaxies, nebulae), use
    method="corners" to estimate background from image corners only.
    This prevents the faint halo from being mistaken for background.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image (H, W, 3), linear data.
    object_mask : np.ndarray, optional
        Binary mask where True = object (galaxy, star).
        Only used if method="masked".
    sigma_clip : float, default 2.5
        Sigma threshold for outlier rejection.
    n_iters : int, default 5
        Number of sigma-clipping iterations.
    method : {'corners', 'masked'}, default 'corners'
        Background estimation method:
        - 'corners': Use image corners only (safest for extended objects)
        - 'masked': Use all non-masked pixels (can include halo signal)

    Returns
    -------
    tuple[float, float, float]
        Background levels (bg_r, bg_g, bg_b).
    """
    h, w = rgb.shape[:2]

    if method == "corners":
        # Use image corners - safest for extended objects like galaxies
        # Corners are least likely to contain galaxy signal
        corner_size = min(h, w) // 8

        corners_r = []
        corners_g = []
        corners_b = []

        for (y_slice, x_slice) in [
            (slice(0, corner_size), slice(0, corner_size)),  # top-left
            (slice(0, corner_size), slice(-corner_size, None)),  # top-right
            (slice(-corner_size, None), slice(0, corner_size)),  # bottom-left
            (slice(-corner_size, None), slice(-corner_size, None)),  # bottom-right
        ]:
            corners_r.extend(rgb[y_slice, x_slice, 0].flatten())
            corners_g.extend(rgb[y_slice, x_slice, 1].flatten())
            corners_b.extend(rgb[y_slice, x_slice, 2].flatten())

        # Sigma-clipped median for each channel
        backgrounds = []
        for pixels in [corners_r, corners_g, corners_b]:
            pixels = np.array(pixels, dtype=np.float64)
            for _ in range(n_iters):
                med = np.median(pixels)
                mad = np.median(np.abs(pixels - med))
                sigma_est = 1.4826 * mad
                if sigma_est < 1e-6:
                    break
                mask = np.abs(pixels - med) < sigma_clip * sigma_est
                if np.sum(mask) < 100:
                    break
                pixels = pixels[mask]
            backgrounds.append(float(np.median(pixels)))

        logger.debug(
            "Sky background (corners): R=%.1f, G=%.1f, B=%.1f",
            backgrounds[0], backgrounds[1], backgrounds[2],
        )
        return tuple(backgrounds)

    # method == "masked"
    if object_mask is None:
        # Create simple mask from luminance
        luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        median_luma = np.median(luma)
        mad_luma = np.median(np.abs(luma - median_luma))
        threshold = median_luma + 3 * 1.4826 * mad_luma
        object_mask = luma > threshold

        # Dilate to be conservative
        struct = ndimage.generate_binary_structure(2, 1)
        object_mask = ndimage.binary_dilation(object_mask, struct, iterations=10)

    # Sky pixels
    sky_mask = ~object_mask

    backgrounds = []
    for c in range(3):
        channel = rgb[:, :, c]
        sky_pixels = channel[sky_mask].astype(np.float64)

        if len(sky_pixels) < 100:
            # Fallback to simple median
            backgrounds.append(float(np.median(channel)))
            continue

        # Sigma-clipped median
        for _ in range(n_iters):
            med = np.median(sky_pixels)
            mad = np.median(np.abs(sky_pixels - med))
            sigma_est = 1.4826 * mad
            if sigma_est < 1e-6:
                break
            mask = np.abs(sky_pixels - med) < sigma_clip * sigma_est
            if np.sum(mask) < 100:
                break
            sky_pixels = sky_pixels[mask]

        backgrounds.append(float(np.median(sky_pixels)))

    logger.debug(
        "Sky background (masked): R=%.1f, G=%.1f, B=%.1f",
        backgrounds[0], backgrounds[1], backgrounds[2],
    )

    return tuple(backgrounds)


def detect_reference_stars(
    rgb: np.ndarray,
    object_mask: np.ndarray | None = None,
    threshold_sigma: float = 10.0,
    min_area: int = 16,
    max_area: int = 500,
    saturation_limit: float = 0.95,
    n_stars: int = 50,
) -> list[tuple[int, int, float, float, float]]:
    """
    Detect reference stars for white balance.

    Finds mid-brightness, unsaturated stars that can serve as neutral
    color references.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image (H, W, 3), linear data.
    object_mask : np.ndarray, optional
        Mask of extended objects (galaxies) to exclude.
    threshold_sigma : float, default 10.0
        Detection threshold in sigma above background.
    min_area : int, default 16
        Minimum star area in pixels.
    max_area : int, default 500
        Maximum star area (avoid galaxies/nebulae).
    saturation_limit : float, default 0.95
        Exclude stars with any channel above this fraction of max.
    n_stars : int, default 50
        Target number of reference stars.

    Returns
    -------
    list[tuple]
        List of (y, x, r_flux, g_flux, b_flux) for each star.
    """
    # Compute luminance
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Background statistics
    bg_median = np.median(luma)
    bg_mad = np.median(np.abs(luma - bg_median))
    bg_sigma = 1.4826 * bg_mad

    # Detection threshold
    threshold = bg_median + threshold_sigma * bg_sigma

    # Detect sources
    binary = luma > threshold

    # Exclude extended objects if mask provided
    if object_mask is not None:
        binary = binary & ~object_mask

    # Label connected components
    labeled, n_features = ndimage.label(binary)

    if n_features == 0:
        logger.warning("No reference stars detected")
        return []

    # Get component properties
    component_sizes = ndimage.sum(binary, labeled, range(1, n_features + 1))

    # Maximum value in image for saturation check
    max_val = max(rgb[:, :, 0].max(), rgb[:, :, 1].max(), rgb[:, :, 2].max())
    sat_threshold = saturation_limit * max_val

    stars = []
    for label_idx in range(1, n_features + 1):
        size = component_sizes[label_idx - 1]

        # Filter by size
        if size < min_area or size > max_area:
            continue

        # Get star pixels
        star_mask = labeled == label_idx
        y_coords, x_coords = np.where(star_mask)

        # Centroid
        y_center = int(np.mean(y_coords))
        x_center = int(np.mean(x_coords))

        # Check saturation
        r_vals = rgb[:, :, 0][star_mask]
        g_vals = rgb[:, :, 1][star_mask]
        b_vals = rgb[:, :, 2][star_mask]

        if np.max(r_vals) > sat_threshold or np.max(g_vals) > sat_threshold or np.max(b_vals) > sat_threshold:
            continue  # Saturated star

        # Compute flux (sum of values)
        r_flux = float(np.sum(r_vals))
        g_flux = float(np.sum(g_vals))
        b_flux = float(np.sum(b_vals))

        if r_flux < 1 or g_flux < 1 or b_flux < 1:
            continue  # Invalid flux

        stars.append((y_center, x_center, r_flux, g_flux, b_flux))

    # Sort by brightness (G channel) and take middle-brightness stars
    # Tuple is (y, x, r_flux, g_flux, b_flux), so s[3] is G flux
    stars.sort(key=lambda s: s[3])  # Sort by G flux

    # Skip brightest and dimmest, take middle
    if len(stars) > n_stars * 2:
        start = len(stars) // 4
        end = start + n_stars
        stars = stars[start:end]
    elif len(stars) > n_stars:
        stars = stars[:n_stars]

    logger.debug("Detected %d reference stars", len(stars))
    return stars


def compute_white_balance(
    rgb: np.ndarray,
    object_mask: np.ndarray | None = None,
    method: Literal["stars", "gray_world"] = "stars",
    target_neutral: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, float, float]:
    """
    Compute white balance gains.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image (H, W, 3), linear, background-subtracted.
    object_mask : np.ndarray, optional
        Mask of extended objects to exclude from star detection.
    method : {'stars', 'gray_world'}, default 'stars'
        White balance method:
        - 'stars': Use median star color as neutral reference
        - 'gray_world': Assume average scene color is neutral
    target_neutral : tuple, default (1.0, 1.0, 1.0)
        Target color ratio for neutral (e.g., slightly warm bias).

    Returns
    -------
    tuple[float, float, float]
        White balance gains (wb_r, wb_g, wb_b).
    """
    if method == "stars":
        stars = detect_reference_stars(rgb, object_mask=object_mask)

        if len(stars) < 5:
            logger.warning("Too few reference stars (%d), using gray world", len(stars))
            method = "gray_world"
        else:
            # Compute median color ratio
            ratios_rg = [s[2] / s[3] for s in stars]  # R/G
            ratios_bg = [s[4] / s[3] for s in stars]  # B/G

            median_rg = np.median(ratios_rg)
            median_bg = np.median(ratios_bg)

            # Gains to make stars neutral
            # If stars appear red (high R/G), reduce R gain
            wb_g = 1.0
            wb_r = target_neutral[0] / (median_rg * target_neutral[1]) if median_rg > 0 else 1.0
            wb_b = target_neutral[2] / (median_bg * target_neutral[1]) if median_bg > 0 else 1.0

            # Normalize so G gain is 1
            # (or normalize so max gain is 1 to avoid boosting noise)
            max_gain = max(wb_r, wb_g, wb_b)
            wb_r /= max_gain
            wb_g /= max_gain
            wb_b /= max_gain

            logger.info(
                "White balance (stars, n=%d): R=%.3f, G=%.3f, B=%.3f",
                len(stars),
                wb_r,
                wb_g,
                wb_b,
            )
            return (wb_r, wb_g, wb_b)

    if method == "gray_world":
        # Assume average is neutral
        # Use median for robustness
        med_r = np.median(rgb[:, :, 0])
        med_g = np.median(rgb[:, :, 1])
        med_b = np.median(rgb[:, :, 2])

        if med_g < 1e-6:
            return (1.0, 1.0, 1.0)

        # Target: make median neutral
        avg_med = (med_r + med_g + med_b) / 3

        wb_r = avg_med / med_r if med_r > 0 else 1.0
        wb_g = avg_med / med_g if med_g > 0 else 1.0
        wb_b = avg_med / med_b if med_b > 0 else 1.0

        # Normalize
        max_gain = max(wb_r, wb_g, wb_b)
        wb_r /= max_gain
        wb_g /= max_gain
        wb_b /= max_gain

        logger.info(
            "White balance (gray world): R=%.3f, G=%.3f, B=%.3f",
            wb_r,
            wb_g,
            wb_b,
        )
        return (wb_r, wb_g, wb_b)

    raise ValueError(f"Unknown white balance method: {method}")


def calibrate_rgb(
    rgb: np.ndarray,
    object_mask: np.ndarray | None = None,
    wb_method: Literal["stars", "gray_world", "none"] = "stars",
    bg_mode: Literal["none", "common", "per_channel"] = "per_channel",
    rgb_gains: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, ColorCalibration]:
    """
    Calibrate RGB image (background neutralization + white balance).

    This is the main color calibration function. It:
    1. Computes white balance on ORIGINAL data (before background subtraction)
    2. Optionally estimates and subtracts sky background
    3. Applies white balance gains or manual RGB gains
    4. Returns calibrated linear RGB

    For extended objects like galaxies, use bg_mode="none" to preserve
    the faint halo signal that would otherwise be removed as "background".

    Parameters
    ----------
    rgb : np.ndarray
        RGB image (H, W, 3), linear data from stacking.
    object_mask : np.ndarray, optional
        Binary mask where True = object (galaxy, star, nebula).
        Used to exclude objects from background estimation.
    wb_method : {'stars', 'gray_world', 'none'}, default 'stars'
        White balance method.
    bg_mode : {'none', 'common', 'per_channel'}, default 'per_channel'
        Background subtraction mode:
        - 'none': No background subtraction (safest for extended objects)
        - 'common': Single scalar offset for all channels (preserves color ratios)
        - 'per_channel': Different offset per channel (can distort colors)
    rgb_gains : tuple[float, float, float], optional
        Manual RGB gains (r, g, b) to apply instead of computed WB.
        Useful for DWARF Astro filter color correction.

    Returns
    -------
    tuple[np.ndarray, ColorCalibration]
        (calibrated_rgb, calibration_params)
    """
    calibration = ColorCalibration()

    # Step 1: Estimate sky background per channel (always measure, may not subtract)
    bg_r, bg_g, bg_b = estimate_sky_background_rgb(rgb, object_mask=object_mask)

    # Determine actual background offsets based on mode
    if bg_mode == "none":
        actual_bg_r = actual_bg_g = actual_bg_b = 0.0
        logger.info("Background subtraction DISABLED (galaxy mode)")
    elif bg_mode == "common":
        # Use single common offset (minimum of the three) to preserve color ratios
        # This only removes the true "black level", not per-channel differences
        common_bg = min(bg_r, bg_g, bg_b)
        actual_bg_r = actual_bg_g = actual_bg_b = common_bg
        logger.info("Common background offset: %.1f (preserves color ratios)", common_bg)
    else:  # per_channel
        actual_bg_r, actual_bg_g, actual_bg_b = bg_r, bg_g, bg_b
        logger.info("Per-channel background: R=%.1f, G=%.1f, B=%.1f", bg_r, bg_g, bg_b)

    calibration.bg_offset_r = actual_bg_r
    calibration.bg_offset_g = actual_bg_g
    calibration.bg_offset_b = actual_bg_b

    # Step 2: Determine color gains
    # Priority: manual rgb_gains > computed WB > no correction
    if rgb_gains is not None:
        # Use manual RGB gains (e.g., for DWARF Astro filter correction)
        wb_r, wb_g, wb_b = rgb_gains
        logger.info("Using manual RGB gains: R=%.3f, G=%.3f, B=%.3f", wb_r, wb_g, wb_b)
    elif wb_method != "none":
        # Compute white balance on ORIGINAL data
        # IMPORTANT: WB must be computed BEFORE background subtraction!
        wb_r, wb_g, wb_b = _compute_white_balance_raw(
            rgb,
            bg_r, bg_g, bg_b,
            object_mask=object_mask,
            method=wb_method,
        )
    else:
        wb_r, wb_g, wb_b = 1.0, 1.0, 1.0

    calibration.wb_gain_r = wb_r
    calibration.wb_gain_g = wb_g
    calibration.wb_gain_b = wb_b

    # Step 3: Create calibrated output
    calibrated = rgb.copy().astype(np.float32)

    # Step 3a: Apply background subtraction (based on bg_mode)
    if bg_mode != "none":
        calibrated[:, :, 0] -= actual_bg_r
        calibrated[:, :, 1] -= actual_bg_g
        calibrated[:, :, 2] -= actual_bg_b
        # Clip negative values (from noise)
        calibrated = np.maximum(calibrated, 0)

    # Step 4: Apply color gains
    calibrated[:, :, 0] *= wb_r
    calibrated[:, :, 1] *= wb_g
    calibrated[:, :, 2] *= wb_b

    logger.info(
        "RGB calibration complete: bg_mode=%s, gains=(%.3f, %.3f, %.3f)",
        bg_mode,
        calibration.wb_gain_r,
        calibration.wb_gain_g,
        calibration.wb_gain_b,
    )

    return calibrated, calibration


def _aperture_photometry(
    rgb: np.ndarray,
    y: int,
    x: int,
    aperture_radius: int = 4,
    annulus_inner: int = 6,
    annulus_outer: int = 10,
) -> tuple[float, float, float] | None:
    """
    Perform aperture photometry with local annulus background estimation.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image (H, W, 3).
    y, x : int
        Star center coordinates.
    aperture_radius : int
        Radius of circular aperture for star flux.
    annulus_inner, annulus_outer : int
        Inner and outer radii of background annulus.

    Returns
    -------
    tuple[float, float, float] or None
        Background-corrected (R, G, B) fluxes, or None if invalid.
    """
    h, w = rgb.shape[:2]

    # Check bounds - need full annulus to be within image
    margin = annulus_outer + 1
    if y < margin or y >= h - margin or x < margin or x >= w - margin:
        return None

    # Create coordinate grids (centered at star position)
    yy, xx = np.ogrid[y - annulus_outer:y + annulus_outer + 1,
                      x - annulus_outer:x + annulus_outer + 1]
    # Distance from star center
    dist = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)

    # Aperture mask (circular)
    aperture_mask = dist <= aperture_radius

    # Annulus mask (ring between inner and outer radii)
    annulus_mask = (dist >= annulus_inner) & (dist <= annulus_outer)

    # Extract patches for all channels
    patch = rgb[y - annulus_outer:y + annulus_outer + 1,
                x - annulus_outer:x + annulus_outer + 1, :]

    # Compute aperture flux and area
    aperture_area = np.sum(aperture_mask)
    if aperture_area < 5:  # Too small
        return None

    # Compute local background from annulus (per channel)
    annulus_area = np.sum(annulus_mask)
    if annulus_area < 15:  # Need enough pixels for reliable bg
        return None

    net_fluxes = []
    min_net_flux = float('inf')

    for c in range(3):
        # Aperture sum
        aperture_flux = np.sum(patch[:, :, c][aperture_mask])

        # Annulus background (sigma-clipped median for robustness)
        annulus_pixels = patch[:, :, c][annulus_mask]
        median_bg = np.median(annulus_pixels)
        mad = np.median(np.abs(annulus_pixels - median_bg))

        # Sigma-clip outliers in annulus (e.g., other stars)
        if mad > 0:
            sigma_est = 1.4826 * mad
            clipped = annulus_pixels[np.abs(annulus_pixels - median_bg) < 2.5 * sigma_est]
            if len(clipped) > 10:
                annulus_bg = np.median(clipped)
            else:
                annulus_bg = median_bg
        else:
            annulus_bg = median_bg

        # Net flux = aperture - (area * background)
        net_flux = aperture_flux - aperture_area * annulus_bg
        net_fluxes.append(net_flux)
        min_net_flux = min(min_net_flux, net_flux)

    # All channels must have positive signal (at least a minimum threshold)
    # Use relative threshold based on aperture area and typical noise
    min_threshold = aperture_area * 0.5  # Very lenient threshold
    if min_net_flux < min_threshold:
        return None

    return tuple(net_fluxes)


def _compute_white_balance_raw(
    rgb: np.ndarray,
    bg_r: float,
    bg_g: float,
    bg_b: float,
    object_mask: np.ndarray | None = None,
    method: Literal["stars", "gray_world"] = "stars",
) -> tuple[float, float, float]:
    """
    Compute white balance from raw (un-subtracted) RGB data.

    Uses aperture photometry with local annulus background for robust
    star color measurement that works even in crowded or extended fields.

    Parameters
    ----------
    rgb : np.ndarray
        Original RGB image (H, W, 3), linear data.
    bg_r, bg_g, bg_b : float
        Per-channel background levels (used for fallback only).
    object_mask : np.ndarray, optional
        Mask of extended objects to exclude.
    method : str
        White balance method.

    Returns
    -------
    tuple[float, float, float]
        White balance gains (wb_r, wb_g, wb_b).
    """
    if method == "stars":
        # Detect stars in the original image
        # NOTE: Don't pass object_mask here - it includes stars!
        stars = detect_reference_stars(rgb, object_mask=None)

        if len(stars) < 5:
            logger.warning("Too few reference stars (%d), using gray world", len(stars))
            method = "gray_world"
        else:
            # Use aperture photometry with local annulus background
            # This is much more robust than global background subtraction
            corrected_ratios_rg = []
            corrected_ratios_bg = []

            n_failed = 0
            for y, x, _, _, _ in stars:  # We'll remeasure with proper aperture
                # Perform aperture photometry with local background
                # Use smaller radii that work better for typical DWARF 3 stellar PSF
                result = _aperture_photometry(
                    rgb, int(y), int(x),
                    aperture_radius=4,
                    annulus_inner=6,
                    annulus_outer=10,
                )

                if result is None:
                    n_failed += 1
                    continue

                r_net, g_net, b_net = result

                # Compute color ratios (relative to G)
                if g_net > 0:
                    corrected_ratios_rg.append(r_net / g_net)
                    corrected_ratios_bg.append(b_net / g_net)

            logger.debug("Aperture photometry: %d valid, %d failed",
                        len(corrected_ratios_rg), n_failed)

            if len(corrected_ratios_rg) < 5:
                logger.warning("Too few valid star photometry (%d), using gray world",
                               len(corrected_ratios_rg))
                method = "gray_world"
            else:
                # Use robust statistics (median)
                median_rg = np.median(corrected_ratios_rg)
                median_bg = np.median(corrected_ratios_bg)

                # Also compute MAD for quality check
                mad_rg = np.median(np.abs(np.array(corrected_ratios_rg) - median_rg))
                mad_bg = np.median(np.abs(np.array(corrected_ratios_bg) - median_bg))

                logger.info(
                    "Star color ratios (aperture photometry): R/G=%.3f±%.3f, B/G=%.3f±%.3f (n=%d)",
                    median_rg, mad_rg, median_bg, mad_bg, len(corrected_ratios_rg)
                )

                # Gains to make median star neutral (R=G=B)
                wb_g = 1.0
                wb_r = 1.0 / median_rg if median_rg > 0.1 else 1.0
                wb_b = 1.0 / median_bg if median_bg > 0.1 else 1.0

                # Normalize so max gain is 1 (avoid boosting noise)
                max_gain = max(wb_r, wb_g, wb_b)
                wb_r /= max_gain
                wb_g /= max_gain
                wb_b /= max_gain

                logger.info(
                    "White balance (stars, n=%d): R=%.3f, G=%.3f, B=%.3f",
                    len(corrected_ratios_rg), wb_r, wb_g, wb_b
                )
                return (wb_r, wb_g, wb_b)

    if method == "gray_world":
        # For gray world, use background-subtracted medians
        med_r = np.median(rgb[:, :, 0]) - bg_r
        med_g = np.median(rgb[:, :, 1]) - bg_g
        med_b = np.median(rgb[:, :, 2]) - bg_b

        # Clip to positive
        med_r = max(med_r, 1.0)
        med_g = max(med_g, 1.0)
        med_b = max(med_b, 1.0)

        # Target: make median neutral
        avg_med = (med_r + med_g + med_b) / 3

        wb_r = avg_med / med_r
        wb_g = avg_med / med_g
        wb_b = avg_med / med_b

        # Normalize
        max_gain = max(wb_r, wb_g, wb_b)
        wb_r /= max_gain
        wb_g /= max_gain
        wb_b /= max_gain

        logger.info(
            "White balance (gray world): R=%.3f, G=%.3f, B=%.3f",
            wb_r, wb_g, wb_b
        )
        return (wb_r, wb_g, wb_b)

    return (1.0, 1.0, 1.0)


def scnr_green(
    rgb: np.ndarray,
    strength: float = 0.5,
    method: Literal["average", "maximum", "additive"] = "average",
) -> np.ndarray:
    """
    Selective Color Noise Reduction for green channel.

    SCNR reduces green color cast common in OSC astrophotography,
    especially around stars and in background.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image (H, W, 3).
    strength : float, default 0.5
        Reduction strength (0 = none, 1 = full).
    method : {'average', 'maximum', 'additive'}, default 'average'
        SCNR algorithm:
        - 'average': G = min(G, (R+B)/2)
        - 'maximum': G = min(G, max(R,B))
        - 'additive': G = G - strength * (G - (R+B)/2) where G > (R+B)/2

    Returns
    -------
    np.ndarray
        RGB with reduced green.
    """
    result = rgb.copy().astype(np.float32)
    r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]

    if method == "average":
        neutral = (r + b) / 2
        excess = g - neutral
        excess = np.maximum(excess, 0)  # Only reduce, never add
        g_new = g - strength * excess

    elif method == "maximum":
        neutral = np.maximum(r, b)
        excess = g - neutral
        excess = np.maximum(excess, 0)
        g_new = g - strength * excess

    elif method == "additive":
        neutral = (r + b) / 2
        mask = g > neutral
        excess = np.where(mask, g - neutral, 0)
        g_new = g - strength * excess

    else:
        raise ValueError(f"Unknown SCNR method: {method}")

    result[:, :, 1] = np.maximum(g_new, 0)

    reduction = np.mean(g - result[:, :, 1])
    logger.debug("SCNR green: mean reduction=%.2f (strength=%.2f)", reduction, strength)

    return result


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to CIE Lab color space.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image (H, W, 3), values in [0, 1].

    Returns
    -------
    np.ndarray
        Lab image (H, W, 3) with L in [0, 100], a/b in [-128, 127].
    """
    # Linearize (assume already linear for astro data)
    # Convert to XYZ (sRGB D65 matrix)
    xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    # Reshape for matrix multiplication
    h, w = rgb.shape[:2]
    rgb_flat = rgb.reshape(-1, 3)
    xyz_flat = rgb_flat @ xyz_matrix.T

    # Normalize by D65 white point
    xyz_flat[:, 0] /= 0.95047
    xyz_flat[:, 1] /= 1.00000
    xyz_flat[:, 2] /= 1.08883

    # Lab conversion
    delta = 6 / 29
    delta_sq = delta ** 2
    delta_cb = delta ** 3

    def f(t):
        return np.where(t > delta_cb, np.cbrt(t), t / (3 * delta_sq) + 4 / 29)

    fx = f(xyz_flat[:, 0])
    fy = f(xyz_flat[:, 1])
    fz = f(xyz_flat[:, 2])

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    lab = np.stack([L, a, b], axis=-1).reshape(h, w, 3)
    return lab.astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert CIE Lab to RGB color space.

    Parameters
    ----------
    lab : np.ndarray
        Lab image (H, W, 3).

    Returns
    -------
    np.ndarray
        RGB image (H, W, 3), values clipped to [0, 1].
    """
    h, w = lab.shape[:2]
    lab_flat = lab.reshape(-1, 3)

    L, a, b = lab_flat[:, 0], lab_flat[:, 1], lab_flat[:, 2]

    # Lab to XYZ
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    delta = 6 / 29
    delta_sq = delta ** 2

    def f_inv(t):
        return np.where(t > delta, t ** 3, 3 * delta_sq * (t - 4 / 29))

    x = f_inv(fx) * 0.95047
    y = f_inv(fy) * 1.00000
    z = f_inv(fz) * 1.08883

    xyz_flat = np.stack([x, y, z], axis=-1)

    # XYZ to RGB (inverse sRGB D65 matrix)
    rgb_matrix = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ])

    rgb_flat = xyz_flat @ rgb_matrix.T

    # Clip and reshape
    rgb = np.clip(rgb_flat, 0, 1).reshape(h, w, 3)
    return rgb.astype(np.float32)


def combine_lrgb(
    luminance: np.ndarray,
    rgb: np.ndarray,
    method: Literal["lab", "hsv", "direct"] = "lab",
    lum_weight: float = 1.0,
    chroma_boost: float = 1.0,
) -> LRGBResult:
    """
    Combine processed luminance with RGB chrominance (LRGB technique).

    This is the key function for producing publication-quality color images.
    The processed luminance provides structure (dust lanes, spiral arms),
    while the RGB provides color information.

    Parameters
    ----------
    luminance : np.ndarray
        Processed luminance image (H, W), normalized to [0, 1].
    rgb : np.ndarray
        Calibrated RGB image (H, W, 3), can be linear or stretched.
    method : {'lab', 'hsv', 'direct'}, default 'lab'
        Combination method:
        - 'lab': Replace L channel in Lab space (best color fidelity)
        - 'hsv': Replace V channel in HSV (simpler but less accurate)
        - 'direct': Weight RGB by luminance ratio
    lum_weight : float, default 1.0
        Weight of luminance contribution (1.0 = full replacement).
    chroma_boost : float, default 1.0
        Boost factor for chrominance (saturation).

    Returns
    -------
    LRGBResult
        Combined result with LRGB image and metadata.
    """
    h, w = luminance.shape

    # Ensure RGB is same size as luminance
    if rgb.shape[:2] != (h, w):
        raise ValueError(
            f"Size mismatch: luminance {luminance.shape}, RGB {rgb.shape[:2]}"
        )

    # Normalize RGB to [0, 1] if needed
    rgb_norm = rgb.astype(np.float32)
    rgb_max = rgb_norm.max()
    if rgb_max > 1.0:
        rgb_norm = rgb_norm / rgb_max

    # Normalize luminance to [0, 1]
    lum_norm = luminance.astype(np.float32)
    if lum_norm.max() > 1.0:
        lum_norm = lum_norm / lum_norm.max()

    if method == "lab":
        # Convert RGB to Lab
        lab = rgb_to_lab(rgb_norm)

        # Scale luminance to Lab L range [0, 100]
        new_L = lum_norm * 100

        # Optional: blend with original L
        if lum_weight < 1.0:
            new_L = lum_weight * new_L + (1 - lum_weight) * lab[:, :, 0]

        # Optional: boost chrominance
        if chroma_boost != 1.0:
            lab[:, :, 1] *= chroma_boost
            lab[:, :, 2] *= chroma_boost

        # Replace L channel
        lab[:, :, 0] = new_L

        # Convert back to RGB
        lrgb = lab_to_rgb(lab)
        chrominance = np.stack([lab[:, :, 1], lab[:, :, 2]], axis=-1)
        # Pad chrominance to 3 channels for consistency
        chrominance_3ch = np.zeros_like(rgb_norm)
        chrominance_3ch[:, :, 1] = lab[:, :, 1] / 128  # Normalize a
        chrominance_3ch[:, :, 2] = lab[:, :, 2] / 128  # Normalize b

    elif method == "hsv":
        # Convert RGB to HSV
        from colorsys import rgb_to_hsv, hsv_to_rgb

        hsv = np.zeros_like(rgb_norm)
        for i in range(h):
            for j in range(w):
                hsv[i, j] = rgb_to_hsv(
                    rgb_norm[i, j, 0],
                    rgb_norm[i, j, 1],
                    rgb_norm[i, j, 2],
                )

        # Replace V channel
        hsv[:, :, 2] = lum_norm

        # Optional: boost saturation
        if chroma_boost != 1.0:
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * chroma_boost, 0, 1)

        # Convert back to RGB
        lrgb = np.zeros_like(rgb_norm)
        for i in range(h):
            for j in range(w):
                lrgb[i, j] = hsv_to_rgb(
                    hsv[i, j, 0],
                    hsv[i, j, 1],
                    hsv[i, j, 2],
                )

        chrominance_3ch = np.zeros_like(rgb_norm)
        chrominance_3ch[:, :, 0] = hsv[:, :, 0]  # H
        chrominance_3ch[:, :, 1] = hsv[:, :, 1]  # S

    elif method == "direct":
        # Compute original luminance from RGB
        orig_lum = 0.299 * rgb_norm[:, :, 0] + 0.587 * rgb_norm[:, :, 1] + 0.114 * rgb_norm[:, :, 2]
        orig_lum = np.maximum(orig_lum, 1e-6)  # Avoid division by zero

        # Ratio of new to old luminance
        ratio = lum_norm / orig_lum

        # Apply ratio to RGB (preserves color ratios)
        lrgb = rgb_norm * ratio[:, :, np.newaxis]

        # Optional: boost saturation
        if chroma_boost != 1.0:
            gray = lum_norm[:, :, np.newaxis]
            lrgb = gray + chroma_boost * (lrgb - gray)

        lrgb = np.clip(lrgb, 0, 1)
        chrominance_3ch = rgb_norm / (orig_lum[:, :, np.newaxis] + 1e-6)

    else:
        raise ValueError(f"Unknown LRGB method: {method}")

    logger.info("LRGB combination complete (method=%s)", method)

    return LRGBResult(
        lrgb=lrgb.astype(np.float32),
        luminance=lum_norm,
        chrominance=chrominance_3ch.astype(np.float32),
        method=method,
    )


def stretch_rgb_linked(
    rgb: np.ndarray,
    black_point_percentile: float = 0.5,
    white_point_percentile: float = 99.9,
    asinh_stretch: float = 0.1,
) -> np.ndarray:
    """
    Apply linked stretch to RGB (same parameters for all channels).

    This preserves color balance unlike per-channel stretching.

    Parameters
    ----------
    rgb : np.ndarray
        Linear RGB image (H, W, 3).
    black_point_percentile : float, default 0.5
        Percentile for black point.
    white_point_percentile : float, default 99.9
        Percentile for white point.
    asinh_stretch : float, default 0.1
        Asinh stretch parameter.

    Returns
    -------
    np.ndarray
        Stretched RGB in [0, 1].
    """
    # Compute luminance for percentile calculation
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Get black and white points from luminance
    black_point = np.percentile(luma, black_point_percentile)
    white_point = np.percentile(luma, white_point_percentile)

    if white_point <= black_point:
        white_point = black_point + 1

    # Normalize all channels with same parameters
    normalized = (rgb - black_point) / (white_point - black_point)
    normalized = np.clip(normalized, 0, None)

    # Apply asinh stretch to all channels
    if asinh_stretch > 0:
        stretched = np.arcsinh(normalized / asinh_stretch) / np.arcsinh(1.0 / asinh_stretch)
    else:
        stretched = normalized

    # Normalize to [0, 1]
    stretched = stretched / stretched.max()
    stretched = np.clip(stretched, 0, 1)

    return stretched.astype(np.float32)


def create_extended_object_mask(
    rgb: np.ndarray,
    star_sigma: float = 5.0,
    extended_sigma: float = 1.5,
    smooth_scale: int = 50,
) -> np.ndarray:
    """
    Create mask of extended objects (galaxies, nebulae) for background estimation.

    This mask identifies regions that should NOT be used for background
    estimation because they contain astronomical signal.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image (H, W, 3).
    star_sigma : float, default 5.0
        Detection threshold for stars.
    extended_sigma : float, default 1.5
        Detection threshold for extended structures.
    smooth_scale : int, default 50
        Smoothing scale for extended detection.

    Returns
    -------
    np.ndarray
        Binary mask (True = object, False = sky).
    """
    # Compute luminance
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Background statistics
    bg_median = np.median(luma)
    bg_mad = np.median(np.abs(luma - bg_median))
    bg_sigma = 1.4826 * bg_mad

    if bg_sigma < 1e-6:
        return np.zeros(luma.shape, dtype=bool)

    # 1. Star mask (point sources at high threshold)
    star_threshold = bg_median + star_sigma * bg_sigma
    star_mask = luma > star_threshold

    # Dilate stars generously
    struct = ndimage.generate_binary_structure(2, 1)
    star_mask = ndimage.binary_dilation(star_mask, struct, iterations=5)

    # 2. Extended structure mask (smoothed image at lower threshold)
    smoothed = ndimage.gaussian_filter(luma.astype(np.float64), sigma=smooth_scale)
    smooth_median = np.median(smoothed)
    smooth_mad = np.median(np.abs(smoothed - smooth_median))
    smooth_sigma = 1.4826 * smooth_mad

    if smooth_sigma > 1e-6:
        extended_threshold = smooth_median + extended_sigma * smooth_sigma
        extended_mask = smoothed > extended_threshold

        # Dilate extended structures
        extended_mask = ndimage.binary_dilation(extended_mask, struct, iterations=20)
    else:
        extended_mask = np.zeros_like(star_mask)

    # Combine
    combined = star_mask | extended_mask

    logger.debug(
        "Extended object mask: %.1f%% covered (stars %.1f%%, extended %.1f%%)",
        100 * np.mean(combined),
        100 * np.mean(star_mask),
        100 * np.mean(extended_mask),
    )

    return combined
