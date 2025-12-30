"""
Post-stack luminance processing for DWARF 3 images.

This module provides astrophotographic processing routines for producing
publication-quality grayscale images from stacked FITS masters.

Key principles:
- Luminance-first: treat DWARF 3 output as broadband luminance
- Structure over color: focus on contrast and dynamic range
- Physically honest: no fake RGB from broadband data

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

logger = logging.getLogger(__name__)


def estimate_background(
    data: np.ndarray,
    cell_size: int = 256,
    sigma_clip: float = 2.5,
    n_iters: int = 5,
    object_mask: np.ndarray | None = None,
    model: Literal["polynomial", "spline"] = "polynomial",
    poly_order: int = 2,
) -> np.ndarray:
    """
    Estimate large-scale background using sigma-clipped sampling.

    IMPORTANT: For targets with extended structures (galaxies, nebulae),
    provide an object_mask to exclude them from background estimation.
    Without this, the galaxy halo will be treated as background and removed.

    Parameters
    ----------
    data : np.ndarray
        Input image (2D array).
    cell_size : int, default 256
        Size of the sampling grid cells in pixels. Larger values (256-512)
        are safer for images with extended objects.
    sigma_clip : float, default 2.5
        Sigma threshold for outlier rejection in each cell.
    n_iters : int, default 5
        Number of sigma-clipping iterations.
    object_mask : np.ndarray, optional
        Binary mask where True = object (galaxy, nebula, stars).
        These pixels are excluded from background estimation.
        CRITICAL for preserving extended structures like M31's halo.
    model : {"polynomial", "spline"}, default "polynomial"
        Background model type:
        - "polynomial": Low-order surface (safer, less flexible)
        - "spline": Bicubic spline (more flexible, can overfit)
    poly_order : int, default 2
        Polynomial order (1=plane, 2=quadratic). Only used if model="polynomial".

    Returns
    -------
    np.ndarray
        Background model at full resolution.

    Notes
    -----
    The polynomial model is STRONGLY recommended for deep-sky imaging.
    Splines can easily interpret galaxy halos as "background" and subtract them.
    """
    h, w = data.shape

    # Compute grid dimensions - use larger cells for safety
    n_cells_y = max(4, h // cell_size)
    n_cells_x = max(4, w // cell_size)

    # Cell centers
    y_centers = np.linspace(cell_size // 2, h - cell_size // 2, n_cells_y)
    x_centers = np.linspace(cell_size // 2, w - cell_size // 2, n_cells_x)

    # Compute sigma-clipped median for each cell
    bg_grid = np.zeros((n_cells_y, n_cells_x), dtype=np.float32)
    bg_weights = np.zeros((n_cells_y, n_cells_x), dtype=np.float32)

    for i, y in enumerate(y_centers):
        for j, x in enumerate(x_centers):
            # Extract cell
            y0 = max(0, int(y) - cell_size // 2)
            y1 = min(h, int(y) + cell_size // 2)
            x0 = max(0, int(x) - cell_size // 2)
            x1 = min(w, int(x) + cell_size // 2)

            cell_data = data[y0:y1, x0:x1]

            # Apply object mask if provided
            if object_mask is not None:
                cell_mask = object_mask[y0:y1, x0:x1]
                # Only use pixels NOT in the object mask
                sky_pixels = cell_data[~cell_mask].flatten().astype(np.float64)
            else:
                sky_pixels = cell_data.flatten().astype(np.float64)

            if len(sky_pixels) < 10:
                # Not enough sky pixels - mark as invalid
                bg_grid[i, j] = np.nan
                bg_weights[i, j] = 0
                continue

            # Sigma-clipped median
            for _ in range(n_iters):
                med = np.median(sky_pixels)
                mad = np.median(np.abs(sky_pixels - med))
                sigma_est = 1.4826 * mad  # MAD to sigma
                if sigma_est < 1e-6:
                    break
                mask = np.abs(sky_pixels - med) < sigma_clip * sigma_est
                if np.sum(mask) < 10:
                    break
                sky_pixels = sky_pixels[mask]

            bg_grid[i, j] = np.median(sky_pixels)
            bg_weights[i, j] = len(sky_pixels)  # Weight by number of valid pixels

    # Handle cells with no valid data
    valid_mask = ~np.isnan(bg_grid) & (bg_weights > 0)
    if np.sum(valid_mask) < 3:
        logger.warning("Too few valid background samples, returning constant")
        return np.full((h, w), np.nanmedian(bg_grid), dtype=np.float32)

    if model == "polynomial":
        background = _fit_polynomial_background(
            bg_grid, bg_weights, y_centers, x_centers, h, w, poly_order
        )
    else:  # spline
        background = _fit_spline_background(
            bg_grid, y_centers, x_centers, h, w
        )

    logger.debug(
        "Background model (%s): min=%.1f, max=%.1f, mean=%.1f, range=%.1f",
        model,
        background.min(),
        background.max(),
        background.mean(),
        background.max() - background.min(),
    )

    return background


def _fit_polynomial_background(
    bg_grid: np.ndarray,
    bg_weights: np.ndarray,
    y_centers: np.ndarray,
    x_centers: np.ndarray,
    h: int,
    w: int,
    order: int = 2,
) -> np.ndarray:
    """
    Fit a low-order polynomial surface to background samples.

    This is much safer than splines for preserving extended structures.
    Order 1 = plane (tilt correction only)
    Order 2 = quadratic (handles vignetting)
    """
    # Build coordinate arrays for valid samples
    valid_mask = ~np.isnan(bg_grid) & (bg_weights > 0)

    yy, xx = np.meshgrid(y_centers, x_centers, indexing='ij')
    y_valid = yy[valid_mask].flatten()
    x_valid = xx[valid_mask].flatten()
    z_valid = bg_grid[valid_mask].flatten()
    w_valid = bg_weights[valid_mask].flatten()

    # Normalize coordinates for numerical stability
    y_norm = (y_valid - h / 2) / h
    x_norm = (x_valid - w / 2) / w

    # Build design matrix for polynomial
    if order == 1:
        # Plane: z = a + b*x + c*y
        A = np.column_stack([np.ones_like(x_norm), x_norm, y_norm])
    elif order == 2:
        # Quadratic: z = a + b*x + c*y + d*x^2 + e*y^2 + f*x*y
        A = np.column_stack([
            np.ones_like(x_norm),
            x_norm, y_norm,
            x_norm**2, y_norm**2, x_norm * y_norm
        ])
    else:
        raise ValueError(f"Unsupported polynomial order: {order}")

    # Weighted least squares fit
    W = np.diag(np.sqrt(w_valid))
    try:
        coeffs, _, _, _ = np.linalg.lstsq(W @ A, W @ z_valid, rcond=None)
    except np.linalg.LinAlgError:
        logger.warning("Polynomial fit failed, returning median")
        return np.full((h, w), np.median(z_valid), dtype=np.float32)

    # Evaluate polynomial at all pixel positions
    y_full, x_full = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    y_full_norm = (y_full - h / 2) / h
    x_full_norm = (x_full - w / 2) / w

    if order == 1:
        background = (coeffs[0] +
                      coeffs[1] * x_full_norm +
                      coeffs[2] * y_full_norm)
    else:  # order == 2
        background = (coeffs[0] +
                      coeffs[1] * x_full_norm +
                      coeffs[2] * y_full_norm +
                      coeffs[3] * x_full_norm**2 +
                      coeffs[4] * y_full_norm**2 +
                      coeffs[5] * x_full_norm * y_full_norm)

    return background.astype(np.float32)


def _fit_spline_background(
    bg_grid: np.ndarray,
    y_centers: np.ndarray,
    x_centers: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray:
    """
    Fit bicubic spline to background samples.

    WARNING: Splines are very flexible and can interpret galaxy halos
    as background. Use polynomial model for deep-sky imaging.
    """
    # Replace NaN with median for spline fitting
    valid_values = bg_grid[~np.isnan(bg_grid)]
    if len(valid_values) == 0:
        return np.zeros((h, w), dtype=np.float32)

    bg_grid_filled = np.where(np.isnan(bg_grid), np.median(valid_values), bg_grid)

    # Normalize coordinates
    y_norm = y_centers / h
    x_norm = x_centers / w

    spline = RectBivariateSpline(y_norm, x_norm, bg_grid_filled, kx=3, ky=3)

    y_full = np.arange(h) / h
    x_full = np.arange(w) / w

    return spline(y_full, x_full).astype(np.float32)


def subtract_background(
    data: np.ndarray,
    background: np.ndarray | None = None,
    cell_size: int = 256,
    mode: Literal["subtract", "divide", "none"] = "subtract",
    object_mask: np.ndarray | None = None,
    bg_model: Literal["polynomial", "spline"] = "polynomial",
    poly_order: int = 2,
) -> np.ndarray:
    """
    Remove background from image.

    Parameters
    ----------
    data : np.ndarray
        Input image (2D array).
    background : np.ndarray, optional
        Pre-computed background model. If None, will be estimated.
    cell_size : int, default 256
        Cell size for background estimation. Larger values (256-512)
        are safer for extended objects.
    mode : {'subtract', 'divide', 'none'}, default 'subtract'
        Correction mode:
        - 'none': Return data unchanged (safest for broadband galaxies)
        - 'subtract': data - background (additive correction)
        - 'divide': data / background * median(background) (multiplicative)
    object_mask : np.ndarray, optional
        Binary mask of objects to exclude from background estimation.
        CRITICAL for preserving galaxy halos.
    bg_model : {'polynomial', 'spline'}, default 'polynomial'
        Background model type. Polynomial is safer.
    poly_order : int, default 2
        Polynomial order (1=plane, 2=quadratic).

    Returns
    -------
    np.ndarray
        Background-corrected image.
    """
    if mode == "none":
        return data.astype(np.float32)

    if background is None:
        background = estimate_background(
            data,
            cell_size=cell_size,
            object_mask=object_mask,
            model=bg_model,
            poly_order=poly_order,
        )

    if mode == "subtract":
        corrected = data - background
        # Shift to positive values
        corrected = corrected - corrected.min()
    elif mode == "divide":
        # Avoid division by zero
        bg_safe = np.maximum(background, 1.0)
        bg_median = np.median(background)
        corrected = (data / bg_safe) * bg_median
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return corrected.astype(np.float32)


def asinh_stretch(
    data: np.ndarray,
    black_point: float = 0.0,
    stretch: float = 0.1,
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply asinh stretch for high dynamic range compression.

    Parameters
    ----------
    data : np.ndarray
        Input image (already normalized to [0, 1] or similar).
    black_point : float, default 0.0
        Black point (values below become 0).
    stretch : float, default 0.1
        Stretch parameter. Lower = more aggressive stretch.
    normalize : bool, default True
        Normalize output to [0, 1].

    Returns
    -------
    np.ndarray
        Stretched image.

    Notes
    -----
    The asinh function compresses high values while preserving
    linearity near the black point. This is ideal for revealing
    faint structures while preserving the bright galaxy core.
    """
    # Shift to black point
    shifted = data - black_point
    shifted = np.maximum(shifted, 0)

    # Asinh stretch
    if stretch > 0:
        stretched = np.arcsinh(shifted / stretch) / np.arcsinh(1.0 / stretch)
    else:
        stretched = shifted

    if normalize:
        vmax = stretched.max()
        if vmax > 0:
            stretched = stretched / vmax

    return stretched.astype(np.float32)


def two_stage_stretch(
    data: np.ndarray,
    core_percentile: float = 99.5,
    faint_stretch: float = 0.05,
    core_stretch: float = 0.5,
    blend_width: float = 0.1,
) -> np.ndarray:
    """
    Apply two-stage stretch with core protection.

    This preserves the bright galaxy core while revealing faint structures.

    Parameters
    ----------
    data : np.ndarray
        Input image (2D array, linear values).
    core_percentile : float, default 99.5
        Percentile threshold separating core from faint regions.
    faint_stretch : float, default 0.05
        Aggressive stretch for faint structures.
    core_stretch : float, default 0.5
        Gentle stretch for bright core.
    blend_width : float, default 0.1
        Width of transition zone (as fraction of range).

    Returns
    -------
    np.ndarray
        Stretched image with protected core.

    Notes
    -----
    Uses a smooth blend between aggressive and gentle stretches
    to avoid visible transitions.
    """
    # Normalize input
    vmin, vmax = np.percentile(data, [0.5, 99.9])
    normalized = (data - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    # Threshold for core
    core_threshold = np.percentile(normalized, core_percentile)

    # Stretch faint regions aggressively
    faint_stretched = asinh_stretch(normalized, stretch=faint_stretch, normalize=True)

    # Stretch core gently
    core_stretched = asinh_stretch(normalized, stretch=core_stretch, normalize=True)

    # Create smooth blend mask
    # Values below threshold use faint_stretched
    # Values above use core_stretched
    blend_mask = (normalized - core_threshold) / (blend_width * core_threshold + 1e-6)
    blend_mask = np.clip(blend_mask, 0, 1)

    # Smooth the mask to avoid artifacts
    blend_mask = ndimage.gaussian_filter(blend_mask, sigma=10)

    # Blend
    result = (1 - blend_mask) * faint_stretched + blend_mask * core_stretched

    return result.astype(np.float32)


def create_star_mask(
    data: np.ndarray,
    threshold_sigma: float = 5.0,
    min_area: int = 9,
    dilate_radius: int = 3,
) -> np.ndarray:
    """
    Create a binary mask of star locations.

    Parameters
    ----------
    data : np.ndarray
        Input image (2D array).
    threshold_sigma : float, default 5.0
        Detection threshold in sigma above background.
    min_area : int, default 9
        Minimum area for star detection.
    dilate_radius : int, default 3
        Dilation radius to expand masks around stars.

    Returns
    -------
    np.ndarray
        Binary mask (True where stars are).
    """
    # Estimate background
    bg_median = np.median(data)
    bg_mad = np.median(np.abs(data - bg_median))
    bg_sigma = 1.4826 * bg_mad

    # Threshold
    threshold = bg_median + threshold_sigma * bg_sigma
    binary = data > threshold

    # Label connected components
    labeled, n_features = ndimage.label(binary)

    if n_features == 0:
        return np.zeros_like(data, dtype=bool)

    # Filter by area
    component_sizes = ndimage.sum(binary, labeled, range(1, n_features + 1))
    small_mask = component_sizes < min_area
    small_labels = np.arange(1, n_features + 1)[small_mask]

    for label in small_labels:
        binary[labeled == label] = False

    # Dilate to expand masks
    if dilate_radius > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        binary = ndimage.binary_dilation(binary, struct, iterations=dilate_radius)

    return binary


def create_object_mask(
    data: np.ndarray,
    star_sigma: float = 5.0,
    extended_sigma: float = 2.0,
    min_star_area: int = 9,
    star_dilate: int = 5,
    extended_smooth: int = 30,
) -> np.ndarray:
    """
    Create a combined mask of stars and extended structures (galaxies, nebulae).

    This provides a more complete object mask than create_star_mask alone,
    suitable for background-only noise reduction.

    Parameters
    ----------
    data : np.ndarray
        Input image (2D array).
    star_sigma : float, default 5.0
        Detection threshold for stars in sigma above background.
    extended_sigma : float, default 2.0
        Detection threshold for extended structures.
    min_star_area : int, default 9
        Minimum area for star detection.
    star_dilate : int, default 5
        Dilation radius for star masks.
    extended_smooth : int, default 30
        Smoothing scale for detecting extended structures.

    Returns
    -------
    np.ndarray
        Binary mask (True where objects are detected).

    Notes
    -----
    Uses a two-tier approach:
    1. Detect point sources (stars) at high threshold
    2. Detect extended structures (galaxies) at lower threshold after smoothing
    The masks are combined to protect both types during noise reduction.
    """
    # Estimate robust background
    bg_median = np.median(data)
    bg_mad = np.median(np.abs(data - bg_median))
    bg_sigma = 1.4826 * bg_mad

    if bg_sigma < 1e-6:
        return np.zeros_like(data, dtype=bool)

    # 1. Star mask (high threshold, point sources)
    star_threshold = bg_median + star_sigma * bg_sigma
    star_binary = data > star_threshold

    # Label and filter by area
    labeled, n_features = ndimage.label(star_binary)
    if n_features > 0:
        component_sizes = ndimage.sum(star_binary, labeled, range(1, n_features + 1))
        small_mask = component_sizes < min_star_area
        small_labels = np.arange(1, n_features + 1)[small_mask]
        for label in small_labels:
            star_binary[labeled == label] = False

    # Dilate stars
    if star_dilate > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        star_binary = ndimage.binary_dilation(star_binary, struct, iterations=star_dilate)

    # 2. Extended structure mask (lower threshold, smoothed)
    # Smooth to detect large-scale structures
    smoothed = ndimage.gaussian_filter(data.astype(np.float64), sigma=extended_smooth)
    smooth_median = np.median(smoothed)
    smooth_mad = np.median(np.abs(smoothed - smooth_median))
    smooth_sigma = 1.4826 * smooth_mad

    if smooth_sigma > 1e-6:
        extended_threshold = smooth_median + extended_sigma * smooth_sigma
        extended_binary = smoothed > extended_threshold

        # Dilate extended structures slightly
        struct = ndimage.generate_binary_structure(2, 1)
        extended_binary = ndimage.binary_dilation(extended_binary, struct, iterations=10)
    else:
        extended_binary = np.zeros_like(data, dtype=bool)

    # Combine masks
    combined = star_binary | extended_binary

    logger.debug(
        "Object mask: stars=%d%%, extended=%d%%, combined=%d%%",
        100 * np.mean(star_binary),
        100 * np.mean(extended_binary),
        100 * np.mean(combined),
    )

    return combined


def create_galaxy_mask(
    data: np.ndarray,
    threshold_sigma: float = 1.5,
    smoothing_scales: tuple[int, ...] = (50, 100, 200),
    dilation_radius: int = 50,
    min_fraction: float = 0.01,
) -> np.ndarray:
    """
    Create an aggressive mask for extended galaxy/nebula halos.

    This function uses multi-scale detection to find the full extent of
    extended objects like M31's halo, which can span a significant portion
    of the image. This mask is intended for background estimation protection.

    Parameters
    ----------
    data : np.ndarray
        Input image (2D array).
    threshold_sigma : float, default 1.5
        Detection threshold in sigma above background. Lower values
        catch more of the faint halo.
    smoothing_scales : tuple of int, default (50, 100, 200)
        Multiple smoothing scales to detect structure at different sizes.
        Larger values catch larger, fainter features.
    dilation_radius : int, default 50
        Large dilation to create a buffer zone around detected structures.
    min_fraction : float, default 0.01
        Minimum fraction of image that must be detected to be considered
        a valid detection (prevents empty masks from noise).

    Returns
    -------
    np.ndarray
        Binary mask (True where galaxy/nebula is detected).

    Notes
    -----
    This is more aggressive than create_object_mask() and is specifically
    designed for protecting galaxy halos during background subtraction.
    The multi-scale approach ensures both the bright core and faint outer
    halo are captured.
    """
    h, w = data.shape
    combined_mask = np.zeros((h, w), dtype=bool)

    # Estimate robust background from image corners (less likely to have galaxy)
    corner_size = min(h, w) // 8
    corners = [
        data[:corner_size, :corner_size],
        data[:corner_size, -corner_size:],
        data[-corner_size:, :corner_size],
        data[-corner_size:, -corner_size:],
    ]
    corner_data = np.concatenate([c.flatten() for c in corners])
    bg_median = np.median(corner_data)
    bg_mad = np.median(np.abs(corner_data - bg_median))
    bg_sigma = 1.4826 * bg_mad

    if bg_sigma < 1e-6:
        logger.warning("Galaxy mask: zero background sigma, returning empty mask")
        return combined_mask

    logger.debug(
        "Galaxy mask: bg_median=%.1f, bg_sigma=%.1f (from corners)",
        bg_median, bg_sigma
    )

    # Multi-scale detection
    for scale in smoothing_scales:
        # Smooth at this scale
        smoothed = ndimage.gaussian_filter(data.astype(np.float64), sigma=scale)

        # Threshold relative to background
        threshold = bg_median + threshold_sigma * bg_sigma
        scale_mask = smoothed > threshold

        # Only include if significant detection
        if np.mean(scale_mask) >= min_fraction:
            combined_mask |= scale_mask
            logger.debug(
                "Galaxy mask scale %d: %.1f%% detected",
                scale, 100 * np.mean(scale_mask)
            )

    # Large dilation to create buffer zone
    if dilation_radius > 0 and np.any(combined_mask):
        struct = ndimage.generate_binary_structure(2, 1)
        combined_mask = ndimage.binary_dilation(
            combined_mask, struct, iterations=dilation_radius
        )

    # Fill holes in the mask
    combined_mask = ndimage.binary_fill_holes(combined_mask)

    logger.info(
        "Galaxy mask: %.1f%% of image protected",
        100 * np.mean(combined_mask)
    )

    return combined_mask


def create_soft_mask(
    binary_mask: np.ndarray,
    feather_radius: int = 10,
) -> np.ndarray:
    """
    Create a soft-edged mask from a binary mask.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask (True/False).
    feather_radius : int, default 10
        Gaussian sigma for feathering the edges.

    Returns
    -------
    np.ndarray
        Soft mask with values in [0, 1], where 1 = fully masked.
    """
    soft = binary_mask.astype(np.float32)
    if feather_radius > 0:
        soft = ndimage.gaussian_filter(soft, sigma=feather_radius)
    return soft


def reduce_background_noise(
    data: np.ndarray,
    method: Literal["median", "gaussian", "bilateral"] = "median",
    filter_size: int = 3,
    object_mask: np.ndarray | None = None,
    feather_radius: int = 10,
    create_mask: bool = True,
    mask_star_sigma: float = 5.0,
    mask_extended_sigma: float = 2.0,
) -> np.ndarray:
    """
    Apply noise reduction only to background regions, preserving objects.

    This is the key function for masked noise reduction. It detects or uses
    a provided mask of astronomical objects and applies noise filtering only
    to the background (unmasked) regions.

    Parameters
    ----------
    data : np.ndarray
        Input image (2D array).
    method : {'median', 'gaussian', 'bilateral'}, default 'median'
        Noise reduction method:
        - 'median': Median filter (good for salt-and-pepper noise)
        - 'gaussian': Gaussian blur (simple smoothing)
        - 'bilateral': Bilateral filter (edge-preserving, requires skimage)
    filter_size : int, default 3
        Filter kernel size (for median/gaussian) or spatial sigma (bilateral).
    object_mask : np.ndarray, optional
        Pre-computed binary mask of objects (True = object, False = background).
        If None and create_mask=True, will be computed automatically.
    feather_radius : int, default 10
        Radius for feathering mask edges to avoid sharp transitions.
    create_mask : bool, default True
        Whether to create object mask automatically if not provided.
    mask_star_sigma : float, default 5.0
        Star detection threshold (if creating mask).
    mask_extended_sigma : float, default 2.0
        Extended structure detection threshold (if creating mask).

    Returns
    -------
    np.ndarray
        Noise-reduced image with objects preserved.

    Notes
    -----
    The masked approach ensures that:
    - Stars remain sharp (no softening of point sources)
    - Galaxy/nebula structure is preserved (no loss of detail)
    - Background noise is reduced (cleaner appearance)

    The soft mask blending ensures no visible edge artifacts at the
    boundary between filtered and unfiltered regions.
    """
    result = data.astype(np.float32)

    # Get or create object mask
    if object_mask is None and create_mask:
        object_mask = create_object_mask(
            data,
            star_sigma=mask_star_sigma,
            extended_sigma=mask_extended_sigma,
        )
    elif object_mask is None:
        # No mask - apply to entire image
        object_mask = np.zeros_like(data, dtype=bool)

    # Create soft mask (1 = object/protected, 0 = background/filter)
    soft_mask = create_soft_mask(object_mask, feather_radius=feather_radius)

    # Apply noise reduction to get filtered version
    if method == "median":
        filtered = ndimage.median_filter(result, size=filter_size)

    elif method == "gaussian":
        sigma = filter_size / 2.0
        filtered = ndimage.gaussian_filter(result, sigma=sigma)

    elif method == "bilateral":
        # Bilateral filter requires skimage
        try:
            from skimage.restoration import denoise_bilateral

            # Estimate sigma for bilateral filter
            bg_mad = np.median(np.abs(data - np.median(data)))
            sigma_color = 1.4826 * bg_mad * 2  # Slightly larger than noise

            filtered = denoise_bilateral(
                result,
                sigma_color=sigma_color,
                sigma_spatial=filter_size,
            ).astype(np.float32)
        except ImportError:
            logger.warning("skimage not available, falling back to median filter")
            filtered = ndimage.median_filter(result, size=filter_size)
    else:
        raise ValueError(f"Unknown noise reduction method: {method}")

    # Blend: use original where objects are, filtered where background is
    # soft_mask is 1 for objects, 0 for background
    result = soft_mask * data + (1 - soft_mask) * filtered

    n_background = np.sum(soft_mask < 0.5)
    n_total = data.size
    logger.info(
        "Noise reduction (%s, size=%d): %.1f%% background filtered",
        method,
        filter_size,
        100 * n_background / n_total,
    )

    return result.astype(np.float32)


def local_contrast_enhancement(
    data: np.ndarray,
    scale: int = 50,
    strength: float = 0.5,
    star_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Enhance local contrast using unsharp masking.

    Parameters
    ----------
    data : np.ndarray
        Input image (2D array, already stretched).
    scale : int, default 50
        Scale of local contrast enhancement in pixels.
    strength : float, default 0.5
        Enhancement strength (0 = no change, 1 = full enhancement).
    star_mask : np.ndarray, optional
        Binary mask of stars. Enhancement is reduced where True.

    Returns
    -------
    np.ndarray
        Contrast-enhanced image.
    """
    # Create smoothed version
    smoothed = ndimage.gaussian_filter(data, sigma=scale)

    # High-pass component
    high_pass = data - smoothed

    # Apply enhancement
    enhanced = data + strength * high_pass

    # Protect stars if mask provided
    if star_mask is not None:
        # Reduce enhancement in star regions
        star_strength = 0.2  # Much weaker enhancement for stars
        enhanced = np.where(
            star_mask,
            data + star_strength * high_pass,
            enhanced,
        )

    # Clip to valid range
    enhanced = np.clip(enhanced, 0, 1)

    return enhanced.astype(np.float32)


def process_luminance(
    data: np.ndarray,
    subtract_bg: bool = True,
    bg_cell_size: int = 64,
    stretch_mode: Literal["single", "two_stage"] = "two_stage",
    asinh_stretch_param: float = 0.1,
    enhance_contrast: bool = True,
    contrast_scale: int = 50,
    contrast_strength: float = 0.3,
    protect_stars: bool = True,
    reduce_noise: bool = False,
    noise_method: Literal["median", "gaussian", "bilateral"] = "median",
    noise_filter_size: int = 3,
) -> np.ndarray:
    """
    Full luminance processing pipeline.

    This applies the standard post-stack processing sequence:
    1. Background subtraction
    2. Masked noise reduction (optional, background only)
    3. Dynamic range stretch
    4. Local contrast enhancement

    Parameters
    ----------
    data : np.ndarray
        Input image (2D array, linear values from stacking).
    subtract_bg : bool, default True
        Apply background subtraction.
    bg_cell_size : int, default 64
        Cell size for background estimation.
    stretch_mode : {'single', 'two_stage'}, default 'two_stage'
        Stretch method:
        - 'single': standard asinh stretch
        - 'two_stage': core-protected two-stage stretch
    asinh_stretch_param : float, default 0.1
        Stretch parameter for single mode.
    enhance_contrast : bool, default True
        Apply local contrast enhancement.
    contrast_scale : int, default 50
        Scale for contrast enhancement.
    contrast_strength : float, default 0.3
        Strength of contrast enhancement.
    protect_stars : bool, default True
        Reduce enhancement around stars.
    reduce_noise : bool, default False
        Apply masked noise reduction to background regions.
    noise_method : {'median', 'gaussian', 'bilateral'}, default 'median'
        Noise reduction filter type.
    noise_filter_size : int, default 3
        Noise filter kernel size.

    Returns
    -------
    np.ndarray
        Processed image, normalized to [0, 1].

    Notes
    -----
    The noise reduction step is applied early (before stretch) to work
    on linear data where noise characteristics are well-defined. It uses
    automatic object masking to preserve stars and extended structures
    while reducing noise only in background regions.
    """
    logger.info("Starting luminance processing pipeline")
    result = data.astype(np.float32)

    # Step 1: Background subtraction
    if subtract_bg:
        logger.info("Subtracting background (cell_size=%d)", bg_cell_size)
        result = subtract_background(result, cell_size=bg_cell_size)

    # Step 2: Masked noise reduction (before stretch, on linear data)
    if reduce_noise:
        logger.info(
            "Reducing background noise (%s, size=%d)",
            noise_method,
            noise_filter_size,
        )
        result = reduce_background_noise(
            result,
            method=noise_method,
            filter_size=noise_filter_size,
            create_mask=True,
        )

    # Step 3: Dynamic range stretch
    if stretch_mode == "single":
        logger.info("Applying asinh stretch (a=%.3f)", asinh_stretch_param)
        # Normalize first
        vmin, vmax = np.percentile(result, [0.5, 99.9])
        result = (result - vmin) / (vmax - vmin)
        result = np.clip(result, 0, 1)
        result = asinh_stretch(result, stretch=asinh_stretch_param)
    elif stretch_mode == "two_stage":
        logger.info("Applying two-stage stretch with core protection")
        result = two_stage_stretch(result)
    else:
        raise ValueError(f"Unknown stretch_mode: {stretch_mode}")

    # Step 4: Local contrast enhancement
    if enhance_contrast:
        logger.info(
            "Enhancing contrast (scale=%d, strength=%.2f)",
            contrast_scale,
            contrast_strength,
        )
        star_mask = None
        if protect_stars:
            # Create mask on stretched image
            star_mask = create_star_mask(result, threshold_sigma=3.0)

        result = local_contrast_enhancement(
            result,
            scale=contrast_scale,
            strength=contrast_strength,
            star_mask=star_mask,
        )

    logger.info("Luminance processing complete")
    return result
