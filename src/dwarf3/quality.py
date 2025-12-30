"""
Frame quality assessment for DWARF 3 images.

Provides fast, explainable, deterministic quality metrics for frame selection.
No black-box ML - all metrics are transparent and auditable.

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import ndimage

from .config import FrameScore
from .io import read_fits

logger = logging.getLogger(__name__)

# Default number of workers for parallel processing
# Use all available CPUs but leave one free for system
DEFAULT_WORKERS = max(1, os.cpu_count() - 1) if os.cpu_count() else 4

# Typical uint16 max value after BZERO correction
UINT16_MAX = 65535.0


def median_absolute_deviation(data: np.ndarray) -> float:
    """
    Compute the Median Absolute Deviation (MAD).

    MAD = median(|x - median(x)|)

    This is a robust measure of statistical dispersion,
    less sensitive to outliers than standard deviation.

    Parameters
    ----------
    data : np.ndarray
        Input data array.

    Returns
    -------
    float
        MAD value.
    """
    median = np.median(data)
    return float(np.median(np.abs(data - median)))


def estimate_noise_mad(data: np.ndarray) -> float:
    """
    Estimate noise level using MAD-based robust estimator.

    For Gaussian noise: sigma ≈ 1.4826 * MAD

    Parameters
    ----------
    data : np.ndarray
        Image data.

    Returns
    -------
    float
        Estimated noise level (in ADU).
    """
    mad = median_absolute_deviation(data)
    # Scale factor for Gaussian: 1.4826
    return 1.4826 * mad


def saturation_fraction(
    data: np.ndarray,
    threshold: float = 0.95,
    max_value: float = UINT16_MAX,
) -> float:
    """
    Compute fraction of saturated pixels.

    Parameters
    ----------
    data : np.ndarray
        Image data.
    threshold : float, default 0.95
        Fraction of max_value above which pixels are considered saturated.
    max_value : float, default 65535
        Maximum possible pixel value.

    Returns
    -------
    float
        Fraction of pixels above saturation threshold (0.0 to 1.0).
    """
    sat_level = threshold * max_value
    n_saturated = np.sum(data >= sat_level)
    return float(n_saturated / data.size)


def star_density_proxy(
    data: np.ndarray,
    sigma_threshold: float = 5.0,
    min_area: int = 4,
) -> int:
    """
    Fast proxy for star count using threshold + connected components.

    This is NOT a rigorous star detection but provides a quick
    relative measure for frame comparison.

    Parameters
    ----------
    data : np.ndarray
        Image data.
    sigma_threshold : float, default 5.0
        Detection threshold in units of estimated background noise.
    min_area : int, default 4
        Minimum connected component area to count as a detection.

    Returns
    -------
    int
        Number of detected bright regions (star proxy count).
    """
    # Robust background estimation
    bg_median = np.median(data)
    bg_noise = estimate_noise_mad(data)

    if bg_noise < 1e-6:
        # Avoid division issues with very clean data
        return 0

    # Threshold above background
    threshold = bg_median + sigma_threshold * bg_noise
    binary = data > threshold

    # Label connected components
    labeled, n_features = ndimage.label(binary)

    if n_features == 0:
        return 0

    # Filter by minimum area
    component_sizes = ndimage.sum(binary, labeled, range(1, n_features + 1))
    n_valid = int(np.sum(component_sizes >= min_area))

    return n_valid


def score_frame(
    path: str | Path,
    saturation_threshold: float = 0.95,
) -> FrameScore:
    """
    Compute quality metrics for a single frame.

    Parameters
    ----------
    path : str or Path
        Path to the FITS file.
    saturation_threshold : float, default 0.95
        Fraction of max value for saturation detection.

    Returns
    -------
    FrameScore
        Dataclass containing all computed metrics.

    Notes
    -----
    The composite score is designed so that higher = better:
    - Low background is good (sky brightness penalty)
    - Low noise is good (cleaner data)
    - Low saturation is good (no clipped stars)
    - More stars is good (better tracking/focus proxy)

    The formula balances these factors empirically:
        score = star_proxy / (noise * (1 + 10*saturation) * (1 + bg_normalized))

    This is a heuristic ranking, not an absolute quality measure.
    """
    path = Path(path)
    data = read_fits(path)

    # Background statistics
    bg_median = float(np.median(data))
    bg_mad = median_absolute_deviation(data)
    noise = estimate_noise_mad(data)

    # Saturation
    sat_frac = saturation_fraction(data, threshold=saturation_threshold)

    # Star proxy (optional - can be slow for many frames)
    # For MVP, we use a simpler approach based on high-sigma pixel count
    star_proxy = star_density_proxy(data)

    # Composite score (higher = better)
    # Normalize background to typical range
    bg_normalized = bg_median / UINT16_MAX

    # Avoid division by zero
    noise_term = max(noise, 1.0)
    sat_penalty = 1.0 + 10.0 * sat_frac  # Heavy penalty for saturation
    bg_penalty = 1.0 + bg_normalized  # Mild penalty for bright background

    # More stars + less noise + less saturation + darker background = better
    composite = (1.0 + star_proxy) / (noise_term * sat_penalty * bg_penalty)

    return FrameScore(
        path=str(path),
        background_median=bg_median,
        background_mad=bg_mad,
        noise_proxy=noise,
        saturation_fraction=sat_frac,
        composite_score=composite,
    )


def _score_frame_wrapper(args: tuple) -> FrameScore | None:
    """
    Wrapper for score_frame to use with ProcessPoolExecutor.

    Parameters
    ----------
    args : tuple
        (path, saturation_threshold) tuple.

    Returns
    -------
    FrameScore or None
        Score if successful, None if failed.
    """
    path, saturation_threshold = args
    try:
        return score_frame(path, saturation_threshold=saturation_threshold)
    except Exception as e:
        # Can't use logger in subprocess - return None and log in main process
        return None


def rank_frames(
    paths: list[Path],
    saturation_threshold: float = 0.95,
    show_progress: bool = True,
    workers: int | None = None,
) -> list[FrameScore]:
    """
    Score and rank multiple frames by quality.

    Parameters
    ----------
    paths : list[Path]
        List of FITS file paths to evaluate.
    saturation_threshold : float, default 0.95
        Saturation detection threshold.
    show_progress : bool, default True
        Show progress bar.
    workers : int or None, default None
        Number of parallel workers. None uses auto-detection (CPU count - 1).
        Set to 1 for sequential processing.

    Returns
    -------
    list[FrameScore]
        Scores sorted by composite_score (descending, best first).
    """
    from .cli_output import create_progress_bar

    if workers is None:
        workers = DEFAULT_WORKERS

    n_frames = len(paths)

    # Use sequential for small frame counts or single worker
    if workers <= 1 or n_frames < 10:
        return _rank_frames_sequential(paths, saturation_threshold, show_progress)

    # Parallel processing
    scores = []
    failed_count = 0

    # Prepare arguments
    args_list = [(path, saturation_threshold) for path in paths]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = {executor.submit(_score_frame_wrapper, args): args[0] for args in args_list}

        # Process results as they complete with progress bar
        pbar = create_progress_bar(
            total=n_frames,
            desc=f"Scoring ({workers} workers)",
            unit="frame",
            disable=not show_progress,
        )
        with pbar:
            for future in as_completed(futures):
                path = futures[future]
                result = future.result()
                if result is not None:
                    scores.append(result)
                else:
                    failed_count += 1
                    logger.warning("Failed to score %s", Path(path).name)
                pbar.update(1)

    if failed_count > 0:
        logger.warning("Failed to score %d frames", failed_count)

    # Sort by composite score, descending (best first)
    scores.sort(key=lambda s: s.composite_score, reverse=True)

    logger.info(
        "Ranked %d frames. Best: %.4f, Worst: %.4f",
        len(scores),
        scores[0].composite_score if scores else 0,
        scores[-1].composite_score if scores else 0,
    )

    return scores


def _rank_frames_sequential(
    paths: list[Path],
    saturation_threshold: float = 0.95,
    show_progress: bool = True,
) -> list[FrameScore]:
    """
    Score and rank frames sequentially (single-threaded).

    Parameters
    ----------
    paths : list[Path]
        List of FITS file paths to evaluate.
    saturation_threshold : float, default 0.95
        Saturation detection threshold.
    show_progress : bool, default True
        Show progress bar.

    Returns
    -------
    list[FrameScore]
        Scores sorted by composite_score (descending, best first).
    """
    from .cli_output import create_progress_bar

    scores = []
    pbar = create_progress_bar(
        total=len(paths),
        desc="Scoring frames",
        unit="frame",
        disable=not show_progress,
    )

    with pbar:
        for path in paths:
            try:
                score = score_frame(path, saturation_threshold=saturation_threshold)
                scores.append(score)
            except Exception as e:
                logger.warning("Failed to score %s: %s", path.name, e)
            pbar.update(1)

    # Sort by composite score, descending (best first)
    scores.sort(key=lambda s: s.composite_score, reverse=True)

    logger.info(
        "Ranked %d frames. Best: %.4f, Worst: %.4f",
        len(scores),
        scores[0].composite_score if scores else 0,
        scores[-1].composite_score if scores else 0,
    )

    return scores


def select_frames(
    scores: list[FrameScore],
    keep_fraction: float = 0.92,
) -> tuple[list[FrameScore], list[FrameScore]]:
    """
    Select top frames based on quality scores.

    Parameters
    ----------
    scores : list[FrameScore]
        Pre-sorted scores (best first).
    keep_fraction : float, default 0.92
        Fraction of frames to keep.

    Returns
    -------
    tuple[list[FrameScore], list[FrameScore]]
        (kept_scores, rejected_scores)
    """
    n_total = len(scores)
    n_keep = max(1, int(n_total * keep_fraction))

    kept = scores[:n_keep]
    rejected = scores[n_keep:]

    logger.info(
        "Selected %d/%d frames (%.1f%%), rejected %d",
        n_keep,
        n_total,
        100 * n_keep / n_total if n_total > 0 else 0,
        len(rejected),
    )

    return kept, rejected
