"""
Image stacking algorithms for DWARF 3 pipeline.

Implements sigma-clipped mean stacking with configurable parameters.
Designed for memory efficiency with 500+ frames.

Supports optional GPU acceleration via CuPy when available.

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clip
from scipy.ndimage import distance_transform_edt

from .backend import is_gpu_available, sigma_clip_mean_gpu, get_backend_summary

logger = logging.getLogger(__name__)


@dataclass
class StackStatistics:
    """Statistics from a stacking operation."""

    n_frames: int
    total_exposure_s: float
    mean_clipped_fraction: float  # Average fraction of pixels clipped per position
    snr_proxy: float  # Simple SNR estimate


def sigma_clip_mean(
    frames: list[np.ndarray] | np.ndarray,
    sigma: float = 3.0,
    maxiters: int = 5,
    axis: int = 0,
    chunk_rows: int = 64,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sigma-clipped mean of image stack using chunked processing.

    Parameters
    ----------
    frames : list[np.ndarray] or np.ndarray
        List of 2D images or 3D array with shape (n_frames, height, width).
    sigma : float, default 3.0
        Number of standard deviations for clipping threshold.
    maxiters : int, default 5
        Maximum number of clipping iterations.
    axis : int, default 0
        Axis along which to stack (typically 0 for first dimension).
    chunk_rows : int, default 64
        Number of rows to process at a time. Reduces memory usage from
        O(n_frames × height × width) to O(n_frames × chunk_rows × width).
    use_gpu : bool, default False
        Use GPU acceleration via CuPy if available.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (stacked_image, mask_count)
        stacked_image: The sigma-clipped mean.
        mask_count: Number of frames contributing to each pixel (or clipped count for GPU).

    Notes
    -----
    Sigma clipping iteratively rejects outliers (cosmic rays, satellites,
    hot pixels) that deviate more than `sigma` standard deviations from
    the mean at each pixel position.

    Memory-efficient: Processes image in horizontal chunks to avoid
    loading the full 3D cube into memory. For 500+ frames at 3840×2160,
    this reduces memory from ~14 GB to ~500 MB.

    GPU mode: When use_gpu=True and CuPy is available, uses GPU-accelerated
    stacking which can be significantly faster for large stacks.
    """
    # Try GPU path if requested
    if use_gpu:
        if is_gpu_available():
            logger.info("Using GPU backend: %s", get_backend_summary())
            if isinstance(frames, np.ndarray):
                # Convert 3D array to list of frames
                frames = [frames[i] for i in range(frames.shape[0])]
            return sigma_clip_mean_gpu(frames, sigma=sigma, maxiters=maxiters, use_gpu=True)
        else:
            logger.warning("GPU requested but not available, falling back to CPU")

    if isinstance(frames, list):
        if len(frames) == 0:
            raise ValueError("Empty frame list")
        frame_list = frames
        n_frames = len(frames)
        height, width = frames[0].shape
    else:
        # If already a 3D array, use direct stacking (caller handles memory)
        return _sigma_clip_mean_direct(frames, sigma, maxiters, axis)

    logger.info(
        "Stacking %d frames (%dx%d) with sigma=%.1f, maxiters=%d, chunk_rows=%d",
        n_frames, height, width, sigma, maxiters, chunk_rows
    )

    # Allocate output arrays
    stacked = np.zeros((height, width), dtype=np.float32)
    mask_count = np.zeros((height, width), dtype=np.int16)

    # Process in chunks of rows
    n_chunks = (height + chunk_rows - 1) // chunk_rows
    for chunk_idx in range(n_chunks):
        row_start = chunk_idx * chunk_rows
        row_end = min(row_start + chunk_rows, height)
        actual_chunk_rows = row_end - row_start

        if chunk_idx % 10 == 0:
            logger.debug(
                "Processing chunk %d/%d (rows %d-%d)",
                chunk_idx + 1, n_chunks, row_start, row_end - 1
            )

        # Build chunk cube: (n_frames, chunk_rows, width)
        chunk_cube = np.zeros((n_frames, actual_chunk_rows, width), dtype=np.float32)
        for i, frame in enumerate(frame_list):
            chunk_cube[i, :, :] = frame[row_start:row_end, :]

        # Apply sigma clipping to this chunk
        clipped = sigma_clip(
            chunk_cube,
            sigma=sigma,
            maxiters=maxiters,
            axis=0,
            masked=True,
            copy=False,
        )

        # Store results
        stacked[row_start:row_end, :] = np.ma.mean(clipped, axis=0).data
        mask_count[row_start:row_end, :] = n_frames - np.sum(clipped.mask, axis=0)

    logger.info(
        "Stack complete. Mean contributing frames: %.1f, min: %d, max: %d",
        np.mean(mask_count),
        np.min(mask_count),
        np.max(mask_count),
    )

    return stacked, mask_count


def _sigma_clip_mean_direct(
    cube: np.ndarray,
    sigma: float,
    maxiters: int,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Direct sigma-clipped mean for pre-allocated 3D arrays.

    Used when caller has already constructed the cube and accepts
    the memory cost. This is the original implementation path.
    """
    n_frames = cube.shape[0]
    logger.info("Stacking %d frames (direct) with sigma=%.1f, maxiters=%d", n_frames, sigma, maxiters)

    clipped = sigma_clip(
        cube,
        sigma=sigma,
        maxiters=maxiters,
        axis=axis,
        masked=True,
        copy=True,
    )

    stacked = np.ma.mean(clipped, axis=axis).data
    mask_count = n_frames - np.sum(clipped.mask, axis=axis)

    logger.info(
        "Stack complete. Mean contributing frames: %.1f, min: %d, max: %d",
        np.mean(mask_count),
        np.min(mask_count),
        np.max(mask_count),
    )

    return stacked.astype(np.float32), mask_count.astype(np.int16)


def compute_stack_statistics(
    stacked: np.ndarray,
    mask_count: np.ndarray,
    n_frames: int,
    exptime: float,
) -> StackStatistics:
    """
    Compute statistics for the stacked result.

    Parameters
    ----------
    stacked : np.ndarray
        Stacked image.
    mask_count : np.ndarray
        Number of frames contributing per pixel.
    n_frames : int
        Total number of input frames.
    exptime : float
        Exposure time per frame in seconds.

    Returns
    -------
    StackStatistics
        Statistics dataclass.
    """
    total_exposure = n_frames * exptime

    # Average clipped fraction
    avg_contributing = np.mean(mask_count)
    clipped_fraction = 1.0 - (avg_contributing / n_frames)

    # Simple SNR proxy: signal / noise estimate
    # Using median as signal proxy, MAD-based noise estimate
    signal = np.median(stacked)
    noise = 1.4826 * np.median(np.abs(stacked - signal))
    snr_proxy = signal / noise if noise > 0 else 0.0

    return StackStatistics(
        n_frames=n_frames,
        total_exposure_s=total_exposure,
        mean_clipped_fraction=clipped_fraction,
        snr_proxy=snr_proxy,
    )


def weighted_mean(
    frames: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """
    Compute weighted mean of image stack.

    Parameters
    ----------
    frames : list[np.ndarray]
        List of 2D images.
    weights : list[float]
        Weight for each frame (e.g., from quality scores).

    Returns
    -------
    np.ndarray
        Weighted mean image.

    Notes
    -----
    This is a simpler alternative to sigma-clipping when
    quality-based weighting is preferred over outlier rejection.
    Not used in MVP but provided for future use.
    """
    if len(frames) != len(weights):
        raise ValueError("Number of frames must match number of weights")

    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize

    cube = np.stack(frames, axis=0)
    # Reshape weights for broadcasting: (n_frames, 1, 1)
    weights_3d = weights.reshape(-1, 1, 1)

    weighted = np.sum(cube * weights_3d, axis=0)
    return weighted.astype(np.float32)


def median_stack(frames: list[np.ndarray]) -> np.ndarray:
    """
    Simple median stack (no sigma clipping).

    Parameters
    ----------
    frames : list[np.ndarray]
        List of 2D images.

    Returns
    -------
    np.ndarray
        Median-combined image.

    Notes
    -----
    Median is more robust to outliers than mean but less
    optimal for noise reduction. Useful for quick previews
    or when sigma-clipping fails.
    """
    cube = np.stack(frames, axis=0)
    return np.median(cube, axis=0).astype(np.float32)


def sigma_clip_mean_rgb(
    frames: list[np.ndarray],
    sigma: float = 3.0,
    maxiters: int = 5,
    chunk_rows: int = 64,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sigma-clipped mean of RGB image stack, processing each channel.

    Parameters
    ----------
    frames : list[np.ndarray]
        List of RGB images with shape (H, W, 3).
    sigma : float, default 3.0
        Number of standard deviations for clipping threshold.
    maxiters : int, default 5
        Maximum number of clipping iterations.
    chunk_rows : int, default 64
        Number of rows to process at a time for memory efficiency.
    use_gpu : bool, default False
        Use GPU acceleration via CuPy if available.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (stacked_rgb, mask_count)
        stacked_rgb: RGB image with shape (H, W, 3)
        mask_count: Minimum contributing frames across channels (H, W)
    """
    if len(frames) == 0:
        raise ValueError("Empty frame list")

    n_frames = len(frames)
    height, width, channels = frames[0].shape

    if channels != 3:
        raise ValueError(f"Expected 3-channel images, got {channels}")

    logger.info(
        "Stacking %d RGB frames (%dx%d) with sigma=%.1f, maxiters=%d",
        n_frames, height, width, sigma, maxiters
    )

    # Stack each channel separately
    stacked_rgb = np.zeros((height, width, 3), dtype=np.float32)
    mask_counts = np.zeros((height, width, 3), dtype=np.int16)

    for c, channel_name in enumerate(['R', 'G', 'B']):
        logger.info("Stacking %s channel...", channel_name)

        # Extract channel from all frames
        channel_frames = [f[:, :, c] for f in frames]

        # Stack this channel using chunked processing
        stacked_channel, channel_mask = sigma_clip_mean(
            channel_frames,
            sigma=sigma,
            maxiters=maxiters,
            chunk_rows=chunk_rows,
            use_gpu=use_gpu,
        )

        stacked_rgb[:, :, c] = stacked_channel
        mask_counts[:, :, c] = channel_mask

    # Return minimum mask count across channels
    mask_count_min = np.min(mask_counts, axis=2)

    logger.info(
        "RGB stack complete. Mean contributing frames: %.1f",
        np.mean(mask_count_min),
    )

    return stacked_rgb, mask_count_min


def compute_stack_statistics_rgb(
    stacked: np.ndarray,
    mask_count: np.ndarray,
    n_frames: int,
    exptime: float,
) -> StackStatistics:
    """
    Compute statistics for an RGB stacked result.

    Parameters
    ----------
    stacked : np.ndarray
        Stacked RGB image (H, W, 3).
    mask_count : np.ndarray
        Number of frames contributing per pixel (H, W).
    n_frames : int
        Total number of input frames.
    exptime : float
        Exposure time per frame in seconds.

    Returns
    -------
    StackStatistics
        Statistics dataclass.
    """
    total_exposure = n_frames * exptime

    # Average clipped fraction
    avg_contributing = np.mean(mask_count)
    clipped_fraction = 1.0 - (avg_contributing / n_frames)

    # SNR proxy using green channel (most sensitive)
    green = stacked[:, :, 1]
    signal = np.median(green)
    noise = 1.4826 * np.median(np.abs(green - signal))
    snr_proxy = signal / noise if noise > 0 else 0.0

    return StackStatistics(
        n_frames=n_frames,
        total_exposure_s=total_exposure,
        mean_clipped_fraction=clipped_fraction,
        snr_proxy=snr_proxy,
    )


def feather_mask(
    mask: np.ndarray,
    feather_width: int = 10,
) -> np.ndarray:
    """
    Apply soft feathering to mask edges using distance transform.

    Creates a smooth transition from 0 to 1 at mask boundaries,
    preventing hard edges in the final stacked image.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W), float or bool. 1/True = valid, 0/False = invalid.
    feather_width : int, default 10
        Width of the feathering zone in pixels. Larger values create
        smoother transitions.

    Returns
    -------
    np.ndarray
        Feathered mask (H, W), float32 with values in [0, 1].
        Pixels far from edges are 1.0, pixels near edges taper to 0.0.

    Notes
    -----
    Uses distance transform to compute pixel distance from mask boundary.
    The weight function is:

        w(d) = min(1, d / feather_width)

    where d is distance from the nearest invalid (zero) pixel.

    This ensures:
    - Core of valid region: weight = 1.0
    - Edge pixels: weight tapers linearly to 0.0
    - Invalid pixels: weight = 0.0
    """
    if feather_width <= 0:
        return mask.astype(np.float32)

    # Ensure binary mask
    binary_mask = (mask > 0.5).astype(np.uint8)

    # Distance from valid pixels to nearest invalid pixel
    # (distance inside the valid region to its boundary)
    distance = distance_transform_edt(binary_mask)

    # Linear taper: full weight at distance >= feather_width
    feathered = np.clip(distance / feather_width, 0, 1).astype(np.float32)

    return feathered


def crop_to_coverage(
    image: np.ndarray,
    coverage: np.ndarray,
    min_coverage_fraction: float = 0.8,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Crop image to the region with sufficient frame coverage.

    When stacking rotated frames, the corners have fewer contributing
    frames. This function finds the largest rectangular region where
    coverage exceeds a threshold and crops the result.

    Parameters
    ----------
    image : np.ndarray
        Image to crop, shape (H, W) or (H, W, C).
    coverage : np.ndarray
        Coverage map (H, W), integer counts of frames per pixel.
    min_coverage_fraction : float, default 0.8
        Minimum coverage as fraction of maximum. Pixels with
        coverage < min_coverage_fraction * max_coverage are excluded.

    Returns
    -------
    tuple[np.ndarray, tuple[int, int, int, int]]
        (cropped_image, (top, bottom, left, right))
        The bounding box indices for the cropped region.

    Notes
    -----
    Algorithm:
    1. Compute threshold = min_coverage_fraction * max(coverage)
    2. Find all pixels with coverage >= threshold
    3. Compute bounding box of valid region
    4. Return cropped image and bounding box

    A more sophisticated approach could use the largest inscribed
    rectangle, but the simple bounding box is usually adequate for
    field rotation artifacts.
    """
    max_coverage = coverage.max()
    if max_coverage == 0:
        logger.warning("Coverage map is all zeros, returning original image")
        return image, (0, image.shape[0], 0, image.shape[1])

    threshold = min_coverage_fraction * max_coverage
    valid_mask = coverage >= threshold

    # Find bounding box of valid region
    rows_valid = np.any(valid_mask, axis=1)
    cols_valid = np.any(valid_mask, axis=0)

    if not rows_valid.any() or not cols_valid.any():
        logger.warning("No pixels meet coverage threshold, returning original")
        return image, (0, image.shape[0], 0, image.shape[1])

    top = np.argmax(rows_valid)
    bottom = len(rows_valid) - np.argmax(rows_valid[::-1])
    left = np.argmax(cols_valid)
    right = len(cols_valid) - np.argmax(cols_valid[::-1])

    # Crop
    if image.ndim == 2:
        cropped = image[top:bottom, left:right]
    else:
        cropped = image[top:bottom, left:right, :]

    logger.info(
        "Cropped to coverage >= %.0f%% of max: (%d:%d, %d:%d) -> %dx%d",
        min_coverage_fraction * 100,
        top, bottom, left, right,
        bottom - top, right - left,
    )

    return cropped, (top, bottom, left, right)


def get_coverage_bounds(
    coverage: np.ndarray,
    min_coverage_fraction: float = 0.8,
) -> tuple[int, int, int, int]:
    """
    Get bounding box for region with sufficient coverage (without cropping).

    Parameters
    ----------
    coverage : np.ndarray
        Coverage map (H, W).
    min_coverage_fraction : float, default 0.8
        Minimum coverage as fraction of maximum.

    Returns
    -------
    tuple[int, int, int, int]
        (top, bottom, left, right) bounding box indices.
    """
    max_coverage = coverage.max()
    if max_coverage == 0:
        return (0, coverage.shape[0], 0, coverage.shape[1])

    threshold = min_coverage_fraction * max_coverage
    valid_mask = coverage >= threshold

    rows_valid = np.any(valid_mask, axis=1)
    cols_valid = np.any(valid_mask, axis=0)

    if not rows_valid.any() or not cols_valid.any():
        return (0, coverage.shape[0], 0, coverage.shape[1])

    top = int(np.argmax(rows_valid))
    bottom = int(len(rows_valid) - np.argmax(rows_valid[::-1]))
    left = int(np.argmax(cols_valid))
    right = int(len(cols_valid) - np.argmax(cols_valid[::-1]))

    return (top, bottom, left, right)


def mask_aware_mean_rgb(
    frames: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
    threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mask-aware weighted mean of RGB frames.

    This is the correct stacking method when frames have been rotated/warped
    and have varying coverage at each pixel position. At each pixel, the
    result is:

        final(x,y) = sum(frame(x,y) * mask(x,y)) / sum(mask(x,y))

    This prevents black borders from warping from corrupting the average.

    Parameters
    ----------
    frames : list[np.ndarray]
        List of RGB images with shape (H, W, 3). Warped frames should have
        zero values where there's no data.
    masks : list[np.ndarray], optional
        Explicit validity masks (H, W) for each frame. True = valid pixel.
        If None, masks are inferred from frames (non-zero = valid).
    threshold : float, default 0.01
        Fraction of max value below which pixels are considered invalid
        (used when masks is None).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (stacked_rgb, coverage_map)
        stacked_rgb: RGB image (H, W, 3)
        coverage_map: Number of frames contributing at each pixel (H, W)

    Notes
    -----
    Unlike sigma_clip_mean_rgb, this function:
    - Does NOT do sigma clipping (outlier rejection)
    - DOES handle varying pixel coverage correctly
    - Is designed for rotated/warped frames

    For best results, use sigma clipping first on frames that share coverage,
    then use this for final combination across different coverage regions.
    """
    if len(frames) == 0:
        raise ValueError("Empty frame list")

    n_frames = len(frames)
    height, width, channels = frames[0].shape

    if channels != 3:
        raise ValueError(f"Expected 3-channel images, got {channels}")

    logger.info(
        "Mask-aware stacking %d RGB frames (%dx%d)",
        n_frames, height, width
    )

    # Generate masks if not provided
    if masks is None:
        masks = []
        for frame in frames:
            # Compute luminance for threshold
            luma = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
            max_val = luma.max()
            if max_val > 0:
                mask = luma > (threshold * max_val)
            else:
                mask = np.ones((height, width), dtype=bool)
            masks.append(mask)
        logger.debug("Generated %d masks from frame luminance", len(masks))

    # Accumulate weighted sums
    sum_rgb = np.zeros((height, width, 3), dtype=np.float64)
    sum_weight = np.zeros((height, width), dtype=np.float64)

    for i, (frame, mask) in enumerate(zip(frames, masks)):
        mask_float = mask.astype(np.float64)

        # Add weighted contribution
        for c in range(3):
            sum_rgb[:, :, c] += frame[:, :, c].astype(np.float64) * mask_float
        sum_weight += mask_float

    # Compute weighted mean (avoid division by zero)
    stacked_rgb = np.zeros((height, width, 3), dtype=np.float32)
    valid_pixels = sum_weight > 0

    for c in range(3):
        stacked_rgb[:, :, c] = np.divide(
            sum_rgb[:, :, c],
            sum_weight,
            out=np.zeros((height, width), dtype=np.float32),
            where=valid_pixels
        )

    # Coverage map (integer count of contributing frames)
    coverage_map = sum_weight.astype(np.int16)

    # Statistics
    min_coverage = coverage_map[valid_pixels].min() if valid_pixels.any() else 0
    max_coverage = coverage_map.max()
    mean_coverage = coverage_map[valid_pixels].mean() if valid_pixels.any() else 0

    logger.info(
        "Mask-aware stack complete. Coverage: min=%d, max=%d, mean=%.1f",
        min_coverage, max_coverage, mean_coverage
    )

    return stacked_rgb, coverage_map


def sigma_clip_mask_aware_rgb(
    frames: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
    sigma: float = 3.0,
    maxiters: int = 5,
    threshold: float = 0.01,
    feather_width: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sigma-clipped mean with mask-aware weighting for RGB frames.

    Combines sigma clipping (outlier rejection) with proper mask-aware
    averaging for warped frames with varying coverage.

    Parameters
    ----------
    frames : list[np.ndarray]
        List of RGB images (H, W, 3).
    masks : list[np.ndarray], optional
        Validity masks for each frame. If None, inferred from frame values.
    sigma : float, default 3.0
        Sigma threshold for outlier rejection.
    maxiters : int, default 5
        Maximum clipping iterations.
    threshold : float, default 0.01
        Threshold for automatic mask generation.
    feather_width : int, default 0
        Edge feathering width in pixels. If > 0, applies soft tapering
        at frame boundaries to prevent hard edges. Typical values: 5-20.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (stacked_rgb, coverage_map)

    Notes
    -----
    Algorithm:
    1. At each pixel, consider only frames where mask is valid
    2. Apply sigma clipping to those frames
    3. Compute weighted mean of non-clipped, valid frames
    4. Normalize by total weight at each pixel

    When feather_width > 0, masks are converted to soft weights using
    distance transform, creating smooth transitions at frame boundaries.
    This prevents visible edges from field rotation in the final stack.
    """
    if len(frames) == 0:
        raise ValueError("Empty frame list")

    n_frames = len(frames)
    height, width, channels = frames[0].shape

    if channels != 3:
        raise ValueError(f"Expected 3-channel images, got {channels}")

    logger.info(
        "Sigma-clip mask-aware stacking %d RGB frames (%dx%d) sigma=%.1f",
        n_frames, height, width, sigma
    )

    # Generate masks if not provided
    if masks is None:
        masks = []
        for frame in frames:
            luma = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
            max_val = luma.max()
            mask = luma > (threshold * max_val) if max_val > 0 else np.ones((height, width), dtype=bool)
            masks.append(mask.astype(np.float32))
    else:
        # Ensure masks are float
        masks = [m.astype(np.float32) for m in masks]

    # Apply feathering if requested
    if feather_width > 0:
        logger.debug("Applying edge feathering with width=%d pixels", feather_width)
        masks = [feather_mask(m, feather_width) for m in masks]

    # Convert to arrays for vectorized operations
    # Stack: (n_frames, H, W, 3) and (n_frames, H, W)
    frame_stack = np.stack(frames, axis=0).astype(np.float32)
    weight_stack = np.stack(masks, axis=0).astype(np.float32)

    # Binary mask for sigma clipping (any weight > 0.5 is valid for clipping)
    binary_mask_stack = weight_stack > 0.5

    # Initialize output
    stacked_rgb = np.zeros((height, width, 3), dtype=np.float32)
    coverage_map = np.zeros((height, width), dtype=np.int16)

    # Process each channel
    for c in range(3):
        channel_data = frame_stack[:, :, :, c]  # (n_frames, H, W)

        # Create masked array for sigma clipping: invalid pixels are masked out
        masked_data = np.ma.array(channel_data, mask=~binary_mask_stack)

        # Sigma clipping on masked data (uses binary masks for outlier detection)
        for _ in range(maxiters):
            mean = np.ma.mean(masked_data, axis=0)
            std = np.ma.std(masked_data, axis=0)

            # Mark outliers
            deviation = np.abs(masked_data - mean)
            outlier_mask = deviation > (sigma * std)

            # Update mask (combine original mask with outlier mask)
            masked_data.mask = masked_data.mask | outlier_mask

        # Final mask after sigma clipping
        final_valid = ~masked_data.mask  # (n_frames, H, W)

        # Compute weighted mean using feathered weights
        # Only use weights where sigma clipping kept the pixel
        effective_weights = weight_stack * final_valid.astype(np.float32)

        # Weighted sum and normalization
        weighted_sum = np.sum(channel_data * effective_weights, axis=0)
        total_weight = np.sum(effective_weights, axis=0)

        # Avoid division by zero
        valid_weight = total_weight > 1e-6
        stacked_rgb[:, :, c] = np.where(
            valid_weight,
            weighted_sum / total_weight,
            0.0
        )

        # Track coverage (use first channel for coverage map)
        if c == 0:
            # Count frames with non-zero effective weight
            coverage_map = (effective_weights > 0.5).sum(axis=0).astype(np.int16)

    # Statistics
    valid_pixels = coverage_map > 0
    min_cov = coverage_map[valid_pixels].min() if valid_pixels.any() else 0
    max_cov = coverage_map.max()
    mean_cov = coverage_map[valid_pixels].mean() if valid_pixels.any() else 0

    logger.info(
        "Sigma-clip mask-aware stack complete. Coverage: min=%d, max=%d, mean=%.1f",
        min_cov, max_cov, mean_cov
    )

    return stacked_rgb, coverage_map


def sigma_clip_mask_aware_rgb_streaming(
    frames: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
    sigma: float = 3.0,
    threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient sigma-clipped mean with mask-aware weighting.

    Uses two-pass streaming algorithm to avoid loading all frames into memory:
    - Pass 1: Compute per-pixel mean and variance using Welford's algorithm
    - Pass 2: Re-accumulate only values within sigma threshold

    Memory usage: O(H × W × C) instead of O(N × H × W × C)

    Parameters
    ----------
    frames : list[np.ndarray]
        List of RGB images (H, W, 3). Frames are accessed twice but not stacked.
    masks : list[np.ndarray], optional
        Validity masks for each frame. If None, inferred from frame values.
    sigma : float, default 3.0
        Sigma threshold for outlier rejection.
    threshold : float, default 0.01
        Threshold for automatic mask generation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (stacked_rgb, coverage_map)
    """
    if len(frames) == 0:
        raise ValueError("Empty frame list")

    n_frames = len(frames)
    height, width, channels = frames[0].shape

    if channels != 3:
        raise ValueError(f"Expected 3-channel images, got {channels}")

    logger.info(
        "Streaming sigma-clip mask-aware stacking %d RGB frames (%dx%d) sigma=%.1f",
        n_frames, height, width, sigma
    )

    # Generate masks if not provided (lazily evaluated)
    def get_mask(idx: int) -> np.ndarray:
        if masks is not None:
            return masks[idx].astype(np.float32)
        frame = frames[idx]
        luma = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        max_val = luma.max()
        if max_val > 0:
            return (luma > (threshold * max_val)).astype(np.float32)
        return np.ones((height, width), dtype=np.float32)

    # ===== PASS 1: Welford's online algorithm for mean and variance =====
    # Per-channel accumulators
    count = np.zeros((height, width, 3), dtype=np.float32)
    mean = np.zeros((height, width, 3), dtype=np.float32)
    m2 = np.zeros((height, width, 3), dtype=np.float32)  # Sum of squared differences

    logger.debug("Pass 1: Computing running statistics...")
    for i in range(n_frames):
        frame = frames[i].astype(np.float32)
        mask = get_mask(i)

        # Expand mask to 3 channels
        mask_3c = mask[:, :, np.newaxis]
        valid = mask_3c > 0.5

        # Welford update (only where valid)
        count = np.where(valid, count + 1, count)
        delta = frame - mean
        # Avoid division by zero: use count where valid, else 1
        safe_count = np.where(count > 0, count, 1)
        mean = np.where(valid, mean + delta / safe_count, mean)
        delta2 = frame - mean
        m2 = np.where(valid, m2 + delta * delta2, m2)

    # Compute variance (avoid div by zero)
    variance = np.where(count > 1, m2 / (count - 1), 0)
    std = np.sqrt(variance)

    logger.debug("Pass 1 complete. Mean range: %.2f - %.2f", mean.min(), mean.max())

    # ===== PASS 2: Re-accumulate with sigma clipping =====
    # Weighted sum and weight accumulators
    weighted_sum = np.zeros((height, width, 3), dtype=np.float64)
    total_weight = np.zeros((height, width, 3), dtype=np.float64)
    clip_count = np.zeros((height, width), dtype=np.int32)

    logger.debug("Pass 2: Sigma-clipped accumulation...")
    for i in range(n_frames):
        frame = frames[i].astype(np.float32)
        mask = get_mask(i)

        # Expand mask
        mask_3c = mask[:, :, np.newaxis]

        # Check if within sigma bounds (per channel)
        deviation = np.abs(frame - mean)
        within_sigma = deviation <= (sigma * std + 1e-6)  # Small epsilon for numerical stability

        # Pixel is valid if mask is valid AND within sigma for ALL channels
        valid_all_channels = within_sigma.all(axis=2) & (mask > 0.5)

        # Accumulate (use mask as weight)
        weight = valid_all_channels.astype(np.float32)[:, :, np.newaxis]
        weighted_sum += frame * weight
        total_weight += weight

        # Track clipped pixels
        clip_count += valid_all_channels.astype(np.int32)

    # Final normalization
    stacked_rgb = np.where(
        total_weight > 1e-6,
        weighted_sum / total_weight,
        0.0
    ).astype(np.float32)

    coverage_map = clip_count.astype(np.int16)

    # Statistics
    valid_pixels = coverage_map > 0
    if valid_pixels.any():
        min_cov = coverage_map[valid_pixels].min()
        max_cov = coverage_map.max()
        mean_cov = coverage_map[valid_pixels].mean()
    else:
        min_cov, max_cov, mean_cov = 0, 0, 0

    clipped_frac = 1.0 - (coverage_map.sum() / (n_frames * valid_pixels.sum())) if valid_pixels.sum() > 0 else 0

    logger.info(
        "Streaming sigma-clip complete. Coverage: min=%d, max=%d, mean=%.1f, clipped=%.1f%%",
        min_cov, max_cov, mean_cov, clipped_frac * 100
    )

    return stacked_rgb, coverage_map


def scale_transform_2k_to_4k(matrix: np.ndarray) -> np.ndarray:
    """
    Scale affine transform matrix from 2K to 4K coordinates.

    The transform was computed on superpixel (half-res) images.
    To apply to full-res bilinear images, translation must be scaled 2x.

    Parameters
    ----------
    matrix : np.ndarray
        3x3 affine matrix computed on 2K images.

    Returns
    -------
    np.ndarray
        3x3 affine matrix for 4K images.
    """
    scaled = matrix.copy()
    # Scale translation components (tx, ty) by 2x
    scaled[0, 2] *= 2  # tx
    scaled[1, 2] *= 2  # ty
    return scaled


def stream_plane_stack(
    frame_paths: list,
    transforms: list,
    reference_path: str,
    sigma: float = 3.0,
    show_progress: bool = True,
    coverage_threshold: float = 0.5,
    chroma_cleanup: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plane-based stacking with two-pass sigma clipping and chroma cleanup.

    This is the recommended approach (Option B1 from TODO2.md) because:
    - No double interpolation (debayer-then-warp causes artifacts)
    - Each plane is half-res (1080×1920), same as alignment - no transform scaling
    - Two-pass sigma clipping for robust outlier rejection (hot pixels, cosmic rays)
    - Chroma-only cleanup to remove color speckles while preserving luminance texture

    Algorithm:
    1. Pass 1: Accumulate mean and variance per pixel using Welford's algorithm
    2. Pass 2: Re-accumulate rejecting outliers beyond sigma threshold
    3. Rebuild Bayer mosaic from stacked planes
    4. Demosaic ONCE at the end
    5. Apply chroma-only median filter (preserves luminance sharpness)
    6. Crop/mask low-coverage regions

    Parameters
    ----------
    frame_paths : list
        Paths to raw Bayer FITS frames.
    transforms : list
        List of AlignmentTransform objects (computed on superpixel images).
    reference_path : str
        Path to reference frame.
    sigma : float, default 3.0
        Sigma threshold for outlier rejection.
    show_progress : bool, default True
        Show progress bar.
    coverage_threshold : float, default 0.5
        Minimum coverage fraction (relative to max) to keep pixels.
    chroma_cleanup : bool, default True
        Apply 3×3 median filter on chroma channels only.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (stacked_rgb_4k, coverage_map)
        stacked_rgb_4k has shape (H, W, 3) at full 4K resolution.
        coverage_map has shape (H/2, W/2) showing per-pixel frame count.
    """
    from skimage.transform import AffineTransform, warp
    from scipy.ndimage import median_filter
    from .io import read_fits
    from .debayer import extract_bayer_planes, rebuild_bayer_from_planes, debayer_rggb
    from .cli_output import create_progress_bar

    n_frames = len(frame_paths)
    if n_frames == 0:
        raise ValueError("Empty frame list")

    # Read reference to get output shape
    ref_bayer = read_fits(reference_path)
    ref_r, ref_g1, ref_g2, ref_b = extract_bayer_planes(ref_bayer)
    del ref_bayer

    h2, w2 = ref_r.shape  # Half-res (1080, 1920)
    total_frames = n_frames + 1  # Including reference
    logger.info(
        "Plane-based stack: %d frames, plane size %dx%d, sigma=%.1f",
        total_frames, h2, w2, sigma
    )

    # Helper: warp a plane with high-quality interpolation (order=4 for Lanczos-like)
    def warp_plane(plane, inverse_affine, is_mask=False):
        order = 0 if is_mask else 4  # order=4 is higher quality than bicubic
        return warp(
            plane, inverse_affine, output_shape=(h2, w2),
            preserve_range=True, order=order, cval=0
        ).astype(np.float32)

    # ========== PASS 1: Compute mean and variance (Welford's online algorithm) ==========
    logger.info("Pass 1/2: Computing per-pixel mean and variance")

    # Accumulators for Welford's algorithm (per channel)
    count = np.zeros((h2, w2), dtype=np.float64)
    mean_r = np.zeros((h2, w2), dtype=np.float64)
    mean_g1 = np.zeros((h2, w2), dtype=np.float64)
    mean_g2 = np.zeros((h2, w2), dtype=np.float64)
    mean_b = np.zeros((h2, w2), dtype=np.float64)
    m2_r = np.zeros((h2, w2), dtype=np.float64)  # Sum of squared deviations
    m2_g1 = np.zeros((h2, w2), dtype=np.float64)
    m2_g2 = np.zeros((h2, w2), dtype=np.float64)
    m2_b = np.zeros((h2, w2), dtype=np.float64)

    # Process reference first (no transform)
    count += 1.0
    mean_r = ref_r.astype(np.float64)
    mean_g1 = ref_g1.astype(np.float64)
    mean_g2 = ref_g2.astype(np.float64)
    mean_b = ref_b.astype(np.float64)
    del ref_r, ref_g1, ref_g2, ref_b

    pbar = create_progress_bar(
        total=n_frames,
        desc="Pass 1",
        unit="frame",
        disable=not show_progress,
    )

    with pbar:
        for frame_path, transform in zip(frame_paths, transforms):
            if not transform.success or transform.matrix is None:
                pbar.update(1)
                continue

            bayer = read_fits(frame_path)
            r, g1, g2, b = extract_bayer_planes(bayer)
            del bayer

            affine = AffineTransform(matrix=transform.matrix)
            inverse_affine = affine.inverse

            r_w = warp_plane(r, inverse_affine)
            g1_w = warp_plane(g1, inverse_affine)
            g2_w = warp_plane(g2, inverse_affine)
            b_w = warp_plane(b, inverse_affine)
            mask_w = warp_plane(np.ones((h2, w2), dtype=np.float32), inverse_affine, is_mask=True)
            del r, g1, g2, b

            valid = mask_w > 0.5
            count[valid] += 1.0

            # Welford update: delta = x - mean; mean += delta/n; m2 += delta*(x - new_mean)
            n_valid = count[valid]
            delta_r = r_w[valid].astype(np.float64) - mean_r[valid]
            mean_r[valid] += delta_r / n_valid
            m2_r[valid] += delta_r * (r_w[valid].astype(np.float64) - mean_r[valid])

            delta_g1 = g1_w[valid].astype(np.float64) - mean_g1[valid]
            mean_g1[valid] += delta_g1 / n_valid
            m2_g1[valid] += delta_g1 * (g1_w[valid].astype(np.float64) - mean_g1[valid])

            delta_g2 = g2_w[valid].astype(np.float64) - mean_g2[valid]
            mean_g2[valid] += delta_g2 / n_valid
            m2_g2[valid] += delta_g2 * (g2_w[valid].astype(np.float64) - mean_g2[valid])

            delta_b = b_w[valid].astype(np.float64) - mean_b[valid]
            mean_b[valid] += delta_b / n_valid
            m2_b[valid] += delta_b * (b_w[valid].astype(np.float64) - mean_b[valid])

            del r_w, g1_w, g2_w, b_w, mask_w
            pbar.update(1)

    # Compute standard deviation (avoid division by zero)
    eps = 1e-10
    std_r = np.sqrt(np.where(count > 1, m2_r / (count - 1 + eps), 0))
    std_g1 = np.sqrt(np.where(count > 1, m2_g1 / (count - 1 + eps), 0))
    std_g2 = np.sqrt(np.where(count > 1, m2_g2 / (count - 1 + eps), 0))
    std_b = np.sqrt(np.where(count > 1, m2_b / (count - 1 + eps), 0))
    del m2_r, m2_g1, m2_g2, m2_b

    # ========== PASS 2: Re-accumulate with sigma clipping ==========
    logger.info("Pass 2/2: Sigma-clipped accumulation (sigma=%.1f)", sigma)

    sum_r = np.zeros((h2, w2), dtype=np.float64)
    sum_g1 = np.zeros((h2, w2), dtype=np.float64)
    sum_g2 = np.zeros((h2, w2), dtype=np.float64)
    sum_b = np.zeros((h2, w2), dtype=np.float64)
    weight_map = np.zeros((h2, w2), dtype=np.float64)
    reject_count = np.zeros((h2, w2), dtype=np.int32)

    # Re-read reference
    ref_bayer = read_fits(reference_path)
    ref_r, ref_g1, ref_g2, ref_b = extract_bayer_planes(ref_bayer)
    del ref_bayer

    # Check reference against thresholds (always include reference)
    sum_r += ref_r.astype(np.float64)
    sum_g1 += ref_g1.astype(np.float64)
    sum_g2 += ref_g2.astype(np.float64)
    sum_b += ref_b.astype(np.float64)
    weight_map += 1.0
    del ref_r, ref_g1, ref_g2, ref_b

    pbar = create_progress_bar(
        total=n_frames,
        desc="Pass 2",
        unit="frame",
        disable=not show_progress,
    )

    with pbar:
        for frame_path, transform in zip(frame_paths, transforms):
            if not transform.success or transform.matrix is None:
                pbar.update(1)
                continue

            bayer = read_fits(frame_path)
            r, g1, g2, b = extract_bayer_planes(bayer)
            del bayer

            affine = AffineTransform(matrix=transform.matrix)
            inverse_affine = affine.inverse

            r_w = warp_plane(r, inverse_affine)
            g1_w = warp_plane(g1, inverse_affine)
            g2_w = warp_plane(g2, inverse_affine)
            b_w = warp_plane(b, inverse_affine)
            mask_w = warp_plane(np.ones((h2, w2), dtype=np.float32), inverse_affine, is_mask=True)
            del r, g1, g2, b

            valid = mask_w > 0.5

            # Sigma clipping: reject if ANY channel exceeds threshold
            r_dev = np.abs(r_w.astype(np.float64) - mean_r) / (std_r + eps)
            g1_dev = np.abs(g1_w.astype(np.float64) - mean_g1) / (std_g1 + eps)
            g2_dev = np.abs(g2_w.astype(np.float64) - mean_g2) / (std_g2 + eps)
            b_dev = np.abs(b_w.astype(np.float64) - mean_b) / (std_b + eps)

            # Pixel is good if all channels are within sigma threshold
            good = valid & (r_dev < sigma) & (g1_dev < sigma) & (g2_dev < sigma) & (b_dev < sigma)
            rejected = valid & ~good

            sum_r[good] += r_w[good].astype(np.float64)
            sum_g1[good] += g1_w[good].astype(np.float64)
            sum_g2[good] += g2_w[good].astype(np.float64)
            sum_b[good] += b_w[good].astype(np.float64)
            weight_map[good] += 1.0
            reject_count[rejected] += 1

            del r_w, g1_w, g2_w, b_w, mask_w, r_dev, g1_dev, g2_dev, b_dev
            pbar.update(1)

    del mean_r, mean_g1, mean_g2, mean_b, std_r, std_g1, std_g2, std_b

    # Report rejection stats
    total_rejects = reject_count.sum()
    total_possible = (count * (h2 * w2)).sum() if count.max() > 0 else 1
    reject_pct = 100.0 * total_rejects / max(total_possible, 1)
    logger.info("Sigma clipping rejected %.2f%% of pixel samples", reject_pct)

    # Normalize by weight (coverage-aware)
    stacked_r = np.where(weight_map > eps, sum_r / weight_map, 0).astype(np.float32)
    stacked_g1 = np.where(weight_map > eps, sum_g1 / weight_map, 0).astype(np.float32)
    stacked_g2 = np.where(weight_map > eps, sum_g2 / weight_map, 0).astype(np.float32)
    stacked_b = np.where(weight_map > eps, sum_b / weight_map, 0).astype(np.float32)

    # Coverage threshold: mask low-coverage regions
    coverage_map = weight_map.astype(np.int16)
    max_coverage = coverage_map.max()
    min_required = int(coverage_threshold * max_coverage)
    low_coverage = coverage_map < min_required

    if low_coverage.any():
        logger.info(
            "Masking %d pixels with coverage < %d (%.0f%% of max %d)",
            low_coverage.sum(), min_required, coverage_threshold * 100, max_coverage
        )
        stacked_r[low_coverage] = 0
        stacked_g1[low_coverage] = 0
        stacked_g2[low_coverage] = 0
        stacked_b[low_coverage] = 0

    # Rebuild Bayer mosaic from stacked planes
    stacked_bayer = rebuild_bayer_from_planes(stacked_r, stacked_g1, stacked_g2, stacked_b)
    del stacked_r, stacked_g1, stacked_g2, stacked_b

    # Demosaic ONCE at the end (bilinear for full 4K)
    stacked_rgb = debayer_rggb(stacked_bayer, mode="bilinear")
    del stacked_bayer

    # ========== Chroma-only cleanup (preserve luminance texture) ==========
    if chroma_cleanup:
        logger.info("Applying chroma-only median cleanup (3x3)")
        # Convert RGB to YCbCr-like space
        # Y = 0.299*R + 0.587*G + 0.114*B
        # Cb = B - Y (blue difference)
        # Cr = R - Y (red difference)
        r_ch = stacked_rgb[:, :, 0]
        g_ch = stacked_rgb[:, :, 1]
        b_ch = stacked_rgb[:, :, 2]

        y = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
        cb = b_ch - y
        cr = r_ch - y

        # Apply 3×3 median filter ONLY to chroma channels
        cb_clean = median_filter(cb, size=3)
        cr_clean = median_filter(cr, size=3)

        # Reconstruct RGB from cleaned chroma + original luminance
        r_clean = (y + cr_clean).clip(0, None)
        b_clean = (y + cb_clean).clip(0, None)
        g_clean = ((y - 0.299 * r_clean - 0.114 * b_clean) / 0.587).clip(0, None)

        stacked_rgb = np.stack([r_clean, g_clean, b_clean], axis=-1).astype(np.float32)
        del y, cb, cr, cb_clean, cr_clean, r_clean, g_clean, b_clean

    # Statistics
    valid_pixels = coverage_map > min_required
    if valid_pixels.any():
        min_cov = coverage_map[valid_pixels].min()
        max_cov = coverage_map.max()
        mean_cov = coverage_map[valid_pixels].mean()
    else:
        min_cov, max_cov, mean_cov = 0, 0, 0

    logger.info(
        "Plane stack complete: %dx%d RGB, coverage min=%d max=%d mean=%.1f",
        stacked_rgb.shape[0], stacked_rgb.shape[1], min_cov, max_cov, mean_cov
    )

    return stacked_rgb, coverage_map


def stream_fullres_stack(
    frame_paths: list,
    transforms: list,
    reference_path: str,
    sigma: float = 3.0,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Streaming full-resolution stacking with pre-computed transforms.

    Memory-efficient: processes one frame at a time at full 4K resolution.
    Uses two-pass Welford algorithm for sigma-clipped mean.

    Parameters
    ----------
    frame_paths : list
        Paths to raw Bayer FITS frames.
    transforms : list
        List of AlignmentTransform objects (computed on 2K superpixel images).
    reference_path : str
        Path to reference frame.
    sigma : float, default 3.0
        Sigma threshold for outlier rejection.
    show_progress : bool, default True
        Show progress bar.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (stacked_rgb_4k, coverage_map)

    Notes
    -----
    Algorithm:
    1. Pass 1: Stream through frames computing Welford running mean/variance
    2. Pass 2: Re-accumulate only values within sigma threshold

    Transforms computed on 2K are scaled to 4K coordinates.
    """
    from pathlib import Path
    from .io import read_fits
    from .debayer import debayer_rggb
    from .align import apply_transform_to_image
    from .cli_output import create_progress_bar

    n_frames = len(frame_paths)

    if n_frames == 0:
        raise ValueError("Empty frame list")

    # Read reference to get output shape
    ref_bayer = read_fits(reference_path)
    ref_rgb = debayer_rggb(ref_bayer, mode="bilinear")  # Full 4K
    del ref_bayer

    height, width, channels = ref_rgb.shape
    logger.info(
        "Streaming full-res stack: %d frames at %dx%d (4K)",
        n_frames + 1, height, width  # +1 for reference
    )

    # ===== PASS 1: Welford's online algorithm for mean and variance =====
    count = np.zeros((height, width, 3), dtype=np.float32)
    mean = np.zeros((height, width, 3), dtype=np.float32)
    m2 = np.zeros((height, width, 3), dtype=np.float32)

    def welford_update(frame: np.ndarray, mask: np.ndarray):
        """Update Welford accumulators with one frame."""
        nonlocal count, mean, m2
        mask_3c = mask[:, :, np.newaxis]
        valid = mask_3c > 0.5

        count = np.where(valid, count + 1, count)
        delta = frame - mean
        safe_count = np.where(count > 0, count, 1)
        mean = np.where(valid, mean + delta / safe_count, mean)
        delta2 = frame - mean
        m2 = np.where(valid, m2 + delta * delta2, m2)

    # Process reference frame (no transform needed)
    ref_mask = np.ones((height, width), dtype=np.float32)
    welford_update(ref_rgb.astype(np.float32), ref_mask)

    pbar = create_progress_bar(
        total=n_frames * 2,  # Two passes
        desc="FullRes Stack",
        unit="frame",
        disable=not show_progress,
    )

    logger.debug("Pass 1: Computing running statistics at 4K...")
    with pbar:
        pbar.set_postfix(stage="pass1")

        for i, (frame_path, transform) in enumerate(zip(frame_paths, transforms)):
            if not transform.success or transform.matrix is None:
                pbar.update(1)
                continue

            # Read and debayer to full 4K
            bayer = read_fits(frame_path)
            rgb = debayer_rggb(bayer, mode="bilinear")
            del bayer

            # Scale transform from 2K to 4K
            matrix_4k = scale_transform_2k_to_4k(transform.matrix)

            # Apply transform with mask
            aligned_rgb, mask = apply_transform_to_image(
                rgb, matrix_4k, return_mask=True
            )
            del rgb

            # Update Welford accumulators
            welford_update(aligned_rgb.astype(np.float32), mask.astype(np.float32))
            del aligned_rgb, mask

            pbar.update(1)

        # Compute variance and std
        variance = np.where(count > 1, m2 / (count - 1), 0)
        std = np.sqrt(variance)
        del m2, variance

        logger.debug("Pass 1 complete. Mean range: %.2f - %.2f", mean.min(), mean.max())

        # ===== PASS 2: Re-accumulate with sigma clipping =====
        weighted_sum = np.zeros((height, width, 3), dtype=np.float64)
        total_weight = np.zeros((height, width, 3), dtype=np.float64)
        clip_count = np.zeros((height, width), dtype=np.int32)

        def accumulate_clipped(frame: np.ndarray, mask: np.ndarray):
            """Accumulate frame if within sigma bounds."""
            nonlocal weighted_sum, total_weight, clip_count

            mask_3c = mask[:, :, np.newaxis]

            deviation = np.abs(frame - mean)
            within_sigma = deviation <= (sigma * std + 1e-6)

            valid_all_channels = within_sigma.all(axis=2) & (mask > 0.5)

            weight = valid_all_channels.astype(np.float32)[:, :, np.newaxis]
            weighted_sum += frame * weight
            total_weight += weight
            clip_count += valid_all_channels.astype(np.int32)

        pbar.set_postfix(stage="pass2")
        logger.debug("Pass 2: Sigma-clipped accumulation at 4K...")

        # Process reference again
        accumulate_clipped(ref_rgb.astype(np.float32), ref_mask)
        del ref_rgb, ref_mask

        for i, (frame_path, transform) in enumerate(zip(frame_paths, transforms)):
            if not transform.success or transform.matrix is None:
                pbar.update(1)
                continue

            # Read and debayer to full 4K again
            bayer = read_fits(frame_path)
            rgb = debayer_rggb(bayer, mode="bilinear")
            del bayer

            # Scale transform from 2K to 4K
            matrix_4k = scale_transform_2k_to_4k(transform.matrix)

            # Apply transform with mask
            aligned_rgb, mask = apply_transform_to_image(
                rgb, matrix_4k, return_mask=True
            )
            del rgb

            # Accumulate if within sigma
            accumulate_clipped(aligned_rgb.astype(np.float32), mask.astype(np.float32))
            del aligned_rgb, mask

            pbar.update(1)

    # Final normalization
    stacked_rgb = np.where(
        total_weight > 1e-6,
        weighted_sum / total_weight,
        0.0
    ).astype(np.float32)

    coverage_map = clip_count.astype(np.int16)

    # Statistics
    valid_pixels = coverage_map > 0
    if valid_pixels.any():
        min_cov = coverage_map[valid_pixels].min()
        max_cov = coverage_map.max()
        mean_cov = coverage_map[valid_pixels].mean()
    else:
        min_cov, max_cov, mean_cov = 0, 0, 0

    logger.info(
        "Full-res 4K stack complete: %dx%d, coverage min=%d max=%d mean=%.1f",
        height, width, min_cov, max_cov, mean_cov
    )

    return stacked_rgb, coverage_map
