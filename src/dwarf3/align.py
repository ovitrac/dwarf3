"""
Frame registration/alignment for DWARF 3 images.

Uses astroalign for star-based alignment (asterism matching).
Provides explicit failure handling and transform logging.
Supports RGB alignment using transform-based approach.

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from multiprocessing import shared_memory
from pathlib import Path
from typing import Callable

import numpy as np
from skimage.transform import warp, AffineTransform

try:
    import astroalign as aa
except ImportError:
    aa = None

from .io import read_fits
from .debayer import bayer_luma_rggb

logger = logging.getLogger(__name__)

# Default number of workers for parallel processing
DEFAULT_WORKERS = max(1, os.cpu_count() - 1) if os.cpu_count() else 4


@dataclass
class AlignmentTransform:
    """Record of an alignment transformation."""

    source_path: str
    reference_path: str
    success: bool
    error_message: str = ""
    # Affine transform matrix (3x3) if successful
    matrix: np.ndarray | None = None
    # Number of matched stars
    n_matches: int = 0


@dataclass
class AlignmentResult:
    """Result of aligning a single frame."""

    path: str
    success: bool
    aligned_data: np.ndarray | None
    transform: AlignmentTransform


def transform_to_dict(transform: AlignmentTransform) -> dict:
    """
    Convert AlignmentTransform to a JSON-serializable dict.

    Parameters
    ----------
    transform : AlignmentTransform
        Transform to convert.

    Returns
    -------
    dict
        JSON-serializable dictionary.
    """
    d = {
        "source_path": transform.source_path,
        "reference_path": transform.reference_path,
        "success": transform.success,
        "error_message": transform.error_message,
        "n_matches": transform.n_matches,
    }
    if transform.matrix is not None:
        d["matrix"] = transform.matrix.tolist()
    else:
        d["matrix"] = None
    return d


def dict_to_transform(d: dict) -> AlignmentTransform:
    """
    Convert a dict back to AlignmentTransform.

    Parameters
    ----------
    d : dict
        Dictionary from JSON.

    Returns
    -------
    AlignmentTransform
        Reconstructed transform.
    """
    matrix = None
    if d.get("matrix") is not None:
        matrix = np.array(d["matrix"], dtype=np.float64)

    return AlignmentTransform(
        source_path=d["source_path"],
        reference_path=d["reference_path"],
        success=d["success"],
        error_message=d.get("error_message", ""),
        matrix=matrix,
        n_matches=d.get("n_matches", 0),
    )


def save_transforms(
    transforms: list[AlignmentTransform],
    output_path: str | Path,
    reference_path: str = "",
    metadata: dict | None = None,
) -> None:
    """
    Save alignment transforms to a JSON file.

    Parameters
    ----------
    transforms : list[AlignmentTransform]
        List of transforms to save.
    output_path : str or Path
        Output JSON file path.
    reference_path : str, optional
        Path to reference frame (for metadata).
    metadata : dict, optional
        Additional metadata to include.

    Notes
    -----
    The JSON file contains:
    - version: format version
    - reference_path: path to reference frame
    - n_transforms: number of transforms
    - n_successful: number of successful transforms
    - metadata: optional extra info
    - transforms: list of transform dicts
    """
    output_path = Path(output_path)

    n_successful = sum(1 for t in transforms if t.success)

    data = {
        "version": "1.0",
        "reference_path": reference_path,
        "n_transforms": len(transforms),
        "n_successful": n_successful,
        "metadata": metadata or {},
        "transforms": [transform_to_dict(t) for t in transforms],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(
        "Saved %d transforms (%d successful) to %s",
        len(transforms),
        n_successful,
        output_path,
    )


def load_transforms(
    input_path: str | Path,
) -> tuple[list[AlignmentTransform], str, dict]:
    """
    Load alignment transforms from a JSON file.

    Parameters
    ----------
    input_path : str or Path
        Path to JSON file.

    Returns
    -------
    tuple
        (transforms, reference_path, metadata)
    """
    input_path = Path(input_path)

    with open(input_path) as f:
        data = json.load(f)

    transforms = [dict_to_transform(d) for d in data["transforms"]]
    reference_path = data.get("reference_path", "")
    metadata = data.get("metadata", {})

    logger.info(
        "Loaded %d transforms from %s",
        len(transforms),
        input_path,
    )

    return transforms, reference_path, metadata


def apply_cached_transforms(
    frame_paths: list[Path],
    transforms: list[AlignmentTransform],
    reference_data: np.ndarray,
    reference_path: str,
    show_progress: bool = True,
) -> tuple[list[np.ndarray], list[AlignmentResult]]:
    """
    Apply cached transforms to frames without re-computing alignment.

    Parameters
    ----------
    frame_paths : list[Path]
        Paths to frames to transform.
    transforms : list[AlignmentTransform]
        Cached transforms (must match frame_paths).
    reference_data : np.ndarray
        Reference image data.
    reference_path : str
        Path to reference frame.
    show_progress : bool, default True
        Show progress bar.

    Returns
    -------
    tuple[list[np.ndarray], list[AlignmentResult]]
        (aligned_frames, results)

    Notes
    -----
    Frames are matched to transforms by filename. If a transform is not
    found for a frame, alignment fails for that frame.
    """
    from .cli_output import create_progress_bar

    # Build lookup by source filename
    transform_lookup = {
        Path(t.source_path).name: t for t in transforms
    }

    aligned_frames = []
    all_results = []

    pbar = create_progress_bar(
        total=len(frame_paths),
        desc="Applying cached transforms",
        unit="frame",
        disable=not show_progress,
    )

    with pbar:
        for fpath in frame_paths:
            fname = fpath.name
            transform = transform_lookup.get(fname)

            if transform is None:
                # No cached transform for this frame
                result = AlignmentResult(
                    path=str(fpath),
                    success=False,
                    aligned_data=None,
                    transform=AlignmentTransform(
                        source_path=str(fpath),
                        reference_path=reference_path,
                        success=False,
                        error_message="No cached transform found",
                    ),
                )
                all_results.append(result)
                pbar.update(1)
                continue

            if not transform.success or transform.matrix is None:
                # Cached transform indicates failure
                result = AlignmentResult(
                    path=str(fpath),
                    success=False,
                    aligned_data=None,
                    transform=transform,
                )
                all_results.append(result)
                pbar.update(1)
                continue

            # Read frame and apply transform
            try:
                source_data = read_fits(fpath)
                aligned_data = apply_transform_to_image(
                    source_data,
                    transform.matrix,
                    output_shape=reference_data.shape,
                )

                result = AlignmentResult(
                    path=str(fpath),
                    success=True,
                    aligned_data=aligned_data,
                    transform=transform,
                )
                aligned_frames.append(aligned_data)

            except Exception as e:
                logger.error("Failed to apply transform to %s: %s", fpath, e)
                result = AlignmentResult(
                    path=str(fpath),
                    success=False,
                    aligned_data=None,
                    transform=AlignmentTransform(
                        source_path=str(fpath),
                        reference_path=reference_path,
                        success=False,
                        error_message=f"Apply error: {e}",
                    ),
                )

            all_results.append(result)

            # Update progress
            n_success = len(aligned_frames)
            n_total = len(all_results)
            pbar.set_postfix(
                ok=f"{n_success}/{n_total}",
                rate=f"{100*n_success/n_total:.0f}%",
            )
            pbar.update(1)

    n_success = len(aligned_frames)
    n_failed = sum(1 for r in all_results if not r.success)
    logger.info(
        "Applied cached transforms: %d successful, %d failed",
        n_success,
        n_failed,
    )

    return aligned_frames, all_results


def analyze_field_rotation(
    frame_paths: list[str | Path],
    ref_path: str | Path,
    sample_size: int = 5
) -> float:
    """
    Estimate max field rotation in degrees using a sample of frames.

    This helps determine if robust affine alignment is needed (rotation > 0.1°)
    or if faster integer alignment is sufficient (rotation ≈ 0°).

    Parameters
    ----------
    frame_paths : list[str | Path]
        List of all frame paths.
    ref_path : str | Path
        Path to the reference frame.
    sample_size : int, default 5
        Number of frames to sample (distributed evenly).

    Returns
    -------
    float
        Maximum detected rotation in degrees (absolute value).
    """
    if len(frame_paths) < 2:
        return 0.0

    # Ensure paths are Path objects
    frames = [Path(p) for p in frame_paths]
    ref = Path(ref_path)

    # Read reference luminance (superpixel for speed)
    try:
        ref_luma = bayer_luma_rggb(read_fits(ref))
    except Exception as e:
        logger.warning("Could not read reference for rotation check: %s", e)
        return 0.0

    max_rotation = 0.0
    
    # Select sample indices (start, end, and middle points)
    indices = np.linspace(0, len(frames) - 1, sample_size, dtype=int)
    
    for idx in indices:
        fpath = frames[idx]
        if fpath == ref:
            continue
            
        try:
            luma = bayer_luma_rggb(read_fits(fpath))
            transform = find_transform(luma, ref_luma)
            
            if transform.success and transform.matrix is not None:
                # Extract rotation from affine matrix
                # Matrix is [[a, b, tx], [c, d, ty]]
                # rotation = arctan2(b, a)
                # Note: skimage/astroalign matrices are typically [a b tx; c d ty]
                # row 0: x' = ax + by + tx -> a=cos, b=-sin? No, depends on convention.
                # astroalign returns scikit-image compatible transform.
                # Rotation is arctan2(M[1,0], M[0,0]) usually
                
                # Using standard decomposition:
                # sx = sqrt(a^2 + c^2)
                # theta = atan2(c, a) = atan2(M[1,0], M[0,0])
                rotation_rad = np.arctan2(transform.matrix[1, 0], transform.matrix[0, 0])
                rotation_deg = np.abs(np.degrees(rotation_rad))
                
                if rotation_deg > max_rotation:
                    max_rotation = rotation_deg
                    
        except Exception:
            continue
            
    return max_rotation


def register_to_reference(
    source: np.ndarray,
    reference: np.ndarray,
    source_path: str = "",
    reference_path: str = "",
) -> AlignmentResult:
    """
    Align source image to reference using astroalign.

    Parameters
    ----------
    source : np.ndarray
        Source image to align.
    reference : np.ndarray
        Reference image (target).
    source_path : str, optional
        Path for logging/reporting.
    reference_path : str, optional
        Path for logging/reporting.

    Returns
    -------
    AlignmentResult
        Contains aligned data (if successful) and transform info.

    Notes
    -----
    astroalign finds matching star asterisms and computes an affine
    transformation. It handles rotation, translation, and scaling.
    """
    if aa is None:
        raise ImportError("astroalign is required for registration")

    transform = AlignmentTransform(
        source_path=source_path,
        reference_path=reference_path,
        success=False,
    )

    try:
        # astroalign.register returns (aligned_image, footprint)
        # We only need the aligned image
        aligned, footprint = aa.register(source, reference)

        # Get transform details
        try:
            # Find transform separately for logging
            transf, (s_list, t_list) = aa.find_transform(source, reference)
            transform.matrix = transf.params
            transform.n_matches = len(s_list)
        except Exception:
            # Transform found but couldn't extract details
            transform.matrix = None
            transform.n_matches = 0

        transform.success = True

        return AlignmentResult(
            path=source_path,
            success=True,
            aligned_data=aligned,
            transform=transform,
        )

    except aa.MaxIterError as e:
        transform.error_message = f"MaxIterError: {e}"
        logger.warning("Alignment failed (max iter) for %s", source_path)

    except Exception as e:
        transform.error_message = str(e)
        logger.warning("Alignment failed for %s: %s", source_path, e)

    return AlignmentResult(
        path=source_path,
        success=False,
        aligned_data=None,
        transform=transform,
    )


def align_frames(
    frame_paths: list[Path],
    reference_data: np.ndarray,
    reference_path: str,
    max_failures: int = 50,
    show_progress: bool = True,
    on_aligned: Callable[[np.ndarray, str], None] | None = None,
) -> tuple[list[np.ndarray], list[AlignmentResult]]:
    """
    Align multiple frames to a reference.

    Parameters
    ----------
    frame_paths : list[Path]
        Paths to frames to align.
    reference_data : np.ndarray
        Reference image data.
    reference_path : str
        Path to reference frame (for logging).
    max_failures : int, default 50
        Maximum consecutive failures before aborting.
    show_progress : bool, default True
        Show progress bar.
    on_aligned : callable, optional
        Callback function(aligned_data, path) called after each successful alignment.
        Useful for streaming to disk without holding all in memory.

    Returns
    -------
    tuple[list[np.ndarray], list[AlignmentResult]]
        (aligned_frames, all_results)
        aligned_frames contains only successful alignments.
        all_results contains results for all attempted frames.
    """
    from .cli_output import create_progress_bar

    aligned_frames = []
    all_results = []
    consecutive_failures = 0

    pbar = create_progress_bar(
        total=len(frame_paths),
        desc="Aligning frames",
        unit="frame",
        disable=not show_progress,
    )

    with pbar:
        for fpath in frame_paths:
            # Read source frame
            try:
                source_data = read_fits(fpath)
            except Exception as e:
                logger.error("Failed to read %s: %s", fpath, e)
                result = AlignmentResult(
                    path=str(fpath),
                    success=False,
                    aligned_data=None,
                    transform=AlignmentTransform(
                        source_path=str(fpath),
                        reference_path=reference_path,
                        success=False,
                        error_message=f"Read error: {e}",
                    ),
                )
                all_results.append(result)
                consecutive_failures += 1
                pbar.update(1)
                continue

            # Align to reference
            result = register_to_reference(
                source_data,
                reference_data,
                source_path=str(fpath),
                reference_path=reference_path,
            )
            all_results.append(result)

            if result.success:
                consecutive_failures = 0
                aligned_frames.append(result.aligned_data)

                if on_aligned is not None:
                    on_aligned(result.aligned_data, str(fpath))
            else:
                consecutive_failures += 1

            # Update progress bar with success rate
            n_success = len(aligned_frames)
            n_total = len(all_results)
            pbar.set_postfix(
                ok=f"{n_success}/{n_total}",
                rate=f"{100*n_success/n_total:.0f}%",
            )
            pbar.update(1)

            # Check abort condition
            if consecutive_failures >= max_failures:
                logger.error(
                    "Aborting alignment: %d consecutive failures",
                    consecutive_failures,
                )
                break

    n_success = len(aligned_frames)
    n_failed = sum(1 for r in all_results if not r.success)
    logger.info(
        "Alignment complete: %d successful, %d failed",
        n_success,
        n_failed,
    )

    return aligned_frames, all_results


def _align_single_frame(args: tuple) -> tuple[AlignmentResult, np.ndarray | None]:
    """
    Worker function for parallel alignment.

    Parameters
    ----------
    args : tuple
        (frame_path, reference_data, reference_path) tuple.

    Returns
    -------
    tuple
        (AlignmentResult, aligned_data or None)
    """
    frame_path, reference_data, reference_path = args

    try:
        source_data = read_fits(frame_path)
    except Exception as e:
        result = AlignmentResult(
            path=str(frame_path),
            success=False,
            aligned_data=None,
            transform=AlignmentTransform(
                source_path=str(frame_path),
                reference_path=reference_path,
                success=False,
                error_message=f"Read error: {e}",
            ),
        )
        return result, None

    result = register_to_reference(
        source_data,
        reference_data,
        source_path=str(frame_path),
        reference_path=reference_path,
    )

    # Return aligned data separately (not in result to avoid duplication)
    aligned_data = result.aligned_data
    result.aligned_data = None  # Clear from result to reduce memory in transit

    return result, aligned_data


def align_frames_parallel(
    frame_paths: list[Path],
    reference_data: np.ndarray,
    reference_path: str,
    max_failures: int = 50,
    show_progress: bool = True,
    workers: int | None = None,
) -> tuple[list[np.ndarray], list[AlignmentResult]]:
    """
    Align multiple frames to a reference using parallel processing.

    Parameters
    ----------
    frame_paths : list[Path]
        Paths to frames to align.
    reference_data : np.ndarray
        Reference image data.
    reference_path : str
        Path to reference frame (for logging).
    max_failures : int, default 50
        Maximum total failures before logging warning.
    show_progress : bool, default True
        Show progress bar.
    workers : int or None, default None
        Number of parallel workers. None uses auto-detection (CPU count - 1).

    Returns
    -------
    tuple[list[np.ndarray], list[AlignmentResult]]
        (aligned_frames, all_results)
        aligned_frames contains only successful alignments.
        all_results contains results for all attempted frames.

    Notes
    -----
    Unlike the sequential version, this cannot abort on consecutive failures
    since tasks are processed out of order. It tracks total failures instead.
    """
    from .cli_output import create_progress_bar

    if workers is None:
        workers = DEFAULT_WORKERS

    n_frames = len(frame_paths)

    # Use sequential for small frame counts or single worker
    if workers <= 1 or n_frames < 10:
        return align_frames(
            frame_paths,
            reference_data,
            reference_path,
            max_failures=max_failures,
            show_progress=show_progress,
        )

    aligned_frames = []
    all_results = []
    failed_count = 0

    # Prepare arguments - reference data is shared
    args_list = [
        (path, reference_data, reference_path)
        for path in frame_paths
    ]

    pbar = create_progress_bar(
        total=n_frames,
        desc=f"Aligning ({workers} workers)",
        unit="frame",
        disable=not show_progress,
    )

    # Use multiprocessing.Pool.imap_unordered instead of ProcessPoolExecutor
    # to avoid deadlock issues with as_completed and large arrays
    from multiprocessing import Pool, get_context

    # Use 'spawn' context to avoid fork issues
    ctx = get_context('spawn')

    with ctx.Pool(processes=workers) as pool:
        with pbar:
            # imap_unordered processes in any order (like as_completed) but is more robust
            for result_tuple in pool.imap_unordered(_align_single_frame, args_list, chunksize=1):
                try:
                    result, aligned_data = result_tuple

                    # Restore aligned_data to result for consistency
                    if aligned_data is not None:
                        result.aligned_data = aligned_data

                    all_results.append(result)

                    if result.success and aligned_data is not None:
                        aligned_frames.append(aligned_data)
                    else:
                        failed_count += 1

                except Exception as e:
                    logger.error("Alignment task failed: %s", e)
                    failed_count += 1
                    result = AlignmentResult(
                        path="unknown",
                        success=False,
                        aligned_data=None,
                        transform=AlignmentTransform(
                            source_path="unknown",
                            reference_path=reference_path,
                            success=False,
                            error_message=f"Worker error: {e}",
                        ),
                    )
                    all_results.append(result)

                # Update progress bar
                n_success = len(aligned_frames)
                n_total = len(all_results)
                pbar.set_postfix(
                    ok=f"{n_success}/{n_total}",
                    rate=f"{100*n_success/n_total:.0f}%",
                )
                pbar.update(1)

    if failed_count >= max_failures:
        logger.warning(
            "High failure rate in alignment: %d failures",
            failed_count,
        )

    n_success = len(aligned_frames)
    n_failed = sum(1 for r in all_results if not r.success)
    logger.info(
        "Parallel alignment complete: %d successful, %d failed",
        n_success,
        n_failed,
    )

    return aligned_frames, all_results


def select_reference_frame(
    frame_paths: list[Path],
    scores: list | None = None,
    method: str = "best",
) -> tuple[Path, np.ndarray]:
    """
    Select the reference frame for alignment.

    Parameters
    ----------
    frame_paths : list[Path]
        Available frame paths.
    scores : list, optional
        Pre-computed quality scores (FrameScore objects).
        Required if method="best".
    method : str, default "best"
        Selection method: "best" (highest quality) or "first".

    Returns
    -------
    tuple[Path, np.ndarray]
        (reference_path, reference_data)
    """
    if not frame_paths:
        raise ValueError("No frames provided for reference selection")

    if method == "first":
        ref_path = frame_paths[0]
        logger.info("Selected first frame as reference: %s", ref_path.name)

    elif method == "best":
        if scores is None or len(scores) == 0:
            raise ValueError("Scores required for method='best'")

        # scores should already be sorted best-first
        # Find the path in frame_paths that matches the best score
        best_score = scores[0]
        ref_path = Path(best_score.path)
        logger.info(
            "Selected best-scoring frame as reference: %s (score=%.4f)",
            ref_path.name,
            best_score.composite_score,
        )

    else:
        raise ValueError(f"Unknown reference selection method: {method}")

    ref_data = read_fits(ref_path)
    return ref_path, ref_data


def find_transform(
    source_luma: np.ndarray,
    reference_luma: np.ndarray,
    source_path: str = "",
    reference_path: str = "",
) -> AlignmentTransform:
    """
    Find alignment transform between two luminance images.

    Parameters
    ----------
    source_luma : np.ndarray
        Source luminance image (2D).
    reference_luma : np.ndarray
        Reference luminance image (2D).
    source_path : str, optional
        Path for logging/reporting.
    reference_path : str, optional
        Path for logging/reporting.

    Returns
    -------
    AlignmentTransform
        Transform information including the affine matrix if successful.
    """
    if aa is None:
        raise ImportError("astroalign is required for registration")

    transform = AlignmentTransform(
        source_path=source_path,
        reference_path=reference_path,
        success=False,
    )

    try:
        transf, (s_list, t_list) = aa.find_transform(source_luma, reference_luma)
        transform.matrix = transf.params
        transform.n_matches = len(s_list)
        transform.success = True
        return transform

    except aa.MaxIterError as e:
        transform.error_message = f"MaxIterError: {e}"
        logger.warning("Transform failed (max iter) for %s", source_path)

    except Exception as e:
        transform.error_message = str(e)
        logger.warning("Transform failed for %s: %s", source_path, e)

    return transform


def apply_transform_to_image(
    image: np.ndarray,
    transform_matrix: np.ndarray,
    output_shape: tuple[int, ...] | None = None,
    allow_bayer_warp: bool = False,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Apply affine transform to an image (2D or 3D RGB).

    Parameters
    ----------
    image : np.ndarray
        Image to transform. Can be 2D grayscale or 3D RGB (H, W, 3).
    transform_matrix : np.ndarray
        3x3 affine transformation matrix.
    output_shape : tuple, optional
        Output shape. If None, uses input shape.
    allow_bayer_warp : bool, default False
        If False (default) and image appears to be a Bayer mosaic
        (2D array with DWARF 3 typical dimensions), a warning is logged.
        Set to True to suppress this warning (e.g., for luminance-only
        applications where color doesn't matter).
    return_mask : bool, default False
        If True, also return a validity mask indicating which pixels
        in the output have valid data from the source. The mask is
        warped with nearest-neighbor interpolation to preserve sharp
        boundaries. This is essential for proper mask-aware stacking.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        If return_mask=False: Transformed image with same dtype as input.
        If return_mask=True: (warped_image, validity_mask) where
            validity_mask is float32 with 1.0=valid, 0.0=invalid.

    Warnings
    --------
    Warping a 2D Bayer mosaic with sub-pixel interpolation DESTROYS color
    information! Adjacent pixels in a Bayer mosaic are DIFFERENT colors
    (R, G1, G2, B), so interpolating between them mixes color channels.

    For Bayer data, use one of these alternatives:
    - `align_bayer_integer()`: Integer-only shifts, preserves RGGB phase
    - `align_rgb_debayer_first()`: Debayer first, then warp RGB safely

    See Also
    --------
    align_bayer_integer : Integer-only alignment preserving Bayer phase
    align_rgb_debayer_first : Debayer-first alignment for affine transforms
    apply_integer_shift : Integer shift without interpolation

    Notes
    -----
    The validity mask is warped with order=0 (nearest-neighbor) to produce
    clean binary boundaries. This follows the standard algorithm for
    geometry-aware stacking where the mask defines the pixel-dependent
    ensemble for averaging.
    """
    # Safeguard: warn if warping what looks like a Bayer mosaic
    if not allow_bayer_warp and image.ndim == 2:
        h, w = image.shape
        # DWARF 3 typical dimensions: 3840x2160 or similar even dimensions
        if h % 2 == 0 and w % 2 == 0 and h > 1000 and w > 1000:
            logger.warning(
                "apply_transform_to_image() called on 2D array (%dx%d) - "
                "if this is a Bayer mosaic, sub-pixel interpolation will "
                "DESTROY color! Use align_bayer_integer() or "
                "align_rgb_debayer_first() instead. "
                "Set allow_bayer_warp=True to suppress this warning.",
                h, w
            )
    if output_shape is None:
        output_shape = image.shape

    # Create inverse transform for warping
    # astroalign gives us source->target, but warp needs target->source
    affine = AffineTransform(matrix=transform_matrix)
    inverse_affine = affine.inverse

    # Prepare validity mask if requested
    warped_mask = None
    if return_mask:
        # Create source mask (all ones = all valid in source)
        source_mask = np.ones(image.shape[:2], dtype=np.float32)
        # Warp mask with nearest-neighbor (order=0) for sharp boundaries
        warped_mask = warp(
            source_mask,
            inverse_affine,
            output_shape=output_shape[:2],
            preserve_range=True,
            order=0,  # Nearest-neighbor for clean binary mask
            cval=0.0,  # Outside source = invalid
        ).astype(np.float32)

    if image.ndim == 2:
        # Grayscale image
        warped = warp(
            image,
            inverse_affine,
            output_shape=output_shape[:2],
            preserve_range=True,
            order=1,  # Bilinear interpolation
        )
        result = warped.astype(image.dtype)
        if return_mask:
            return result, warped_mask
        return result

    elif image.ndim == 3 and image.shape[2] == 3:
        # RGB image - warp each channel
        warped = np.zeros(output_shape, dtype=image.dtype)
        for c in range(3):
            warped[:, :, c] = warp(
                image[:, :, c],
                inverse_affine,
                output_shape=output_shape[:2],
                preserve_range=True,
                order=1,
            )
        result = warped.astype(image.dtype)
        if return_mask:
            return result, warped_mask
        return result

    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def align_rgb_frames(
    frame_paths: list[Path],
    reference_luma: np.ndarray,
    reference_path: str,
    debayer_func: Callable[[np.ndarray], np.ndarray],
    luma_func: Callable[[np.ndarray], np.ndarray],
    max_failures: int = 50,
    show_progress: bool = True,
) -> tuple[list[np.ndarray], list[AlignmentResult]]:
    """
    Align multiple RGB frames to a reference using luminance-based registration.

    This function:
    1. Loads each raw Bayer frame
    2. Debayers to RGB
    3. Extracts luminance for alignment
    4. Finds transform against reference luminance
    5. Applies transform to full RGB

    Parameters
    ----------
    frame_paths : list[Path]
        Paths to raw Bayer frames.
    reference_luma : np.ndarray
        Reference luminance image.
    reference_path : str
        Path to reference frame (for logging).
    debayer_func : callable
        Function to debayer raw Bayer to RGB: f(bayer) -> rgb
    luma_func : callable
        Function to extract luminance from RGB: f(rgb) -> luma
    max_failures : int, default 50
        Maximum consecutive failures before aborting.
    show_progress : bool, default True
        Show progress bar.

    Returns
    -------
    tuple[list[np.ndarray], list[AlignmentResult]]
        (aligned_rgb_frames, all_results)
    """
    from .cli_output import create_progress_bar

    aligned_frames = []
    all_results = []
    consecutive_failures = 0

    ref_shape = reference_luma.shape

    pbar = create_progress_bar(
        total=len(frame_paths),
        desc="Aligning RGB",
        unit="frame",
        disable=not show_progress,
    )

    with pbar:
        for fpath in frame_paths:
            # Read and debayer
            try:
                raw_data = read_fits(fpath)
                rgb_data = debayer_func(raw_data)
                source_luma = luma_func(rgb_data)
            except Exception as e:
                logger.error("Failed to read/debayer %s: %s", fpath, e)
                result = AlignmentResult(
                    path=str(fpath),
                    success=False,
                    aligned_data=None,
                    transform=AlignmentTransform(
                        source_path=str(fpath),
                        reference_path=reference_path,
                        success=False,
                        error_message=f"Read/debayer error: {e}",
                    ),
                )
                all_results.append(result)
                consecutive_failures += 1
                pbar.update(1)
                continue

            # Find transform using luminance
            transform = find_transform(
                source_luma,
                reference_luma,
                source_path=str(fpath),
                reference_path=reference_path,
            )

            if transform.success:
                # Apply transform to RGB
                aligned_rgb = apply_transform_to_image(
                    rgb_data,
                    transform.matrix,
                    output_shape=rgb_data.shape,
                )

                result = AlignmentResult(
                    path=str(fpath),
                    success=True,
                    aligned_data=aligned_rgb,
                    transform=transform,
                )
                aligned_frames.append(aligned_rgb)
                consecutive_failures = 0

            else:
                result = AlignmentResult(
                    path=str(fpath),
                    success=False,
                    aligned_data=None,
                    transform=transform,
                )
                consecutive_failures += 1

            all_results.append(result)

            # Update progress bar
            n_success = len(aligned_frames)
            n_total = len(all_results)
            pbar.set_postfix(
                ok=f"{n_success}/{n_total}",
                rate=f"{100*n_success/n_total:.0f}%",
            )
            pbar.update(1)

            # Check abort condition
            if consecutive_failures >= max_failures:
                logger.error(
                    "Aborting alignment: %d consecutive failures",
                    consecutive_failures,
                )
                break

    n_success = len(aligned_frames)
    n_failed = sum(1 for r in all_results if not r.success)
    logger.info(
        "RGB alignment complete: %d successful, %d failed",
        n_success,
        n_failed,
    )

    return aligned_frames, all_results


def scale_transform_for_resolution(
    transform_matrix: np.ndarray,
    scale_factor: float = 2.0,
) -> np.ndarray:
    """
    Scale an affine transform matrix for different resolution.

    When a transform is computed at one resolution and needs to be
    applied at a different resolution, the translation component
    must be scaled accordingly.

    Parameters
    ----------
    transform_matrix : np.ndarray
        3x3 affine transformation matrix.
    scale_factor : float, default 2.0
        Factor to scale translation by.
        Use 2.0 when transform was computed at half-res
        and needs to be applied to full-res.

    Returns
    -------
    np.ndarray
        Scaled 3x3 affine transformation matrix.

    Notes
    -----
    For an affine transform matrix:
        [a  b  tx]
        [c  d  ty]
        [0  0  1 ]

    - a, b, c, d (rotation/scale) remain unchanged
    - tx, ty (translation) are multiplied by scale_factor

    Example
    -------
    >>> # Transform computed in half-res (1080x1920)
    >>> transform = find_transform(luma_half, ref_luma_half)
    >>> # Scale for full-res Bayer (2160x3840)
    >>> scaled_matrix = scale_transform_for_resolution(transform.matrix, 2.0)
    >>> aligned_bayer = apply_transform_to_image(bayer, scaled_matrix)
    """
    scaled = transform_matrix.copy()
    # Scale translation components
    scaled[0, 2] *= scale_factor  # tx
    scaled[1, 2] *= scale_factor  # ty
    return scaled


# =============================================================================
# INTEGER-ONLY PHASE-PRESERVING ALIGNMENT (for Bayer color fidelity)
# =============================================================================


def estimate_integer_shift(
    source_luma: np.ndarray,
    reference_luma: np.ndarray,
    force_even: bool = True,
) -> tuple[int, int]:
    """
    Estimate integer pixel shift using phase cross-correlation.

    This is the correct approach for Bayer mosaic alignment: compute
    the shift without any sub-pixel interpolation, preserving the
    RGGB phase relationship.

    Parameters
    ----------
    source_luma : np.ndarray
        Source luminance image (typically from superpixel debayer).
    reference_luma : np.ndarray
        Reference luminance image.
    force_even : bool, default True
        If True, round shifts to even integers to preserve RGGB phase.
        This is essential for Bayer mosaic alignment.

    Returns
    -------
    tuple[int, int]
        (dy, dx) integer shift to apply to source to align with reference.
        Apply with np.roll or apply_integer_shift.

    Notes
    -----
    When force_even=True, shifts are rounded to multiples of 2. This ensures
    that after shifting, R pixels still align with R pixels, G with G, B with B.

    An odd shift would swap the Bayer phase:
    - Even shift: R→R, G→G, B→B (correct)
    - Odd shift: R→G, G→R/B (color destroyed)
    """
    from skimage.registration import phase_cross_correlation

    # Estimate shift using phase correlation
    shift, error, diffphase = phase_cross_correlation(
        reference_luma, source_luma, upsample_factor=1
    )

    dy, dx = int(round(shift[0])), int(round(shift[1]))

    if force_even:
        # Round to even integers to preserve RGGB phase
        dy = 2 * round(dy / 2)
        dx = 2 * round(dx / 2)

    logger.debug(
        "Integer shift: raw=(%.1f, %.1f) -> (%d, %d) even=%s",
        shift[0], shift[1], dy, dx, force_even
    )

    return dy, dx


def apply_integer_shift(
    image: np.ndarray,
    dy: int,
    dx: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Apply integer pixel shift without interpolation.

    This preserves the exact pixel values and Bayer phase relationship,
    unlike warp() which interpolates and destroys color information.

    Parameters
    ----------
    image : np.ndarray
        Image to shift (2D or 3D).
    dy : int
        Vertical shift (positive = shift down in output).
    dx : int
        Horizontal shift (positive = shift right in output).
    fill_value : float, default 0.0
        Value to fill exposed edges.

    Returns
    -------
    np.ndarray
        Shifted image with same shape as input.
    """
    result = np.full_like(image, fill_value)

    # Compute source and destination slices
    if dy >= 0:
        src_y = slice(0, image.shape[0] - dy)
        dst_y = slice(dy, image.shape[0])
    else:
        src_y = slice(-dy, image.shape[0])
        dst_y = slice(0, image.shape[0] + dy)

    if dx >= 0:
        src_x = slice(0, image.shape[1] - dx)
        dst_x = slice(dx, image.shape[1])
    else:
        src_x = slice(-dx, image.shape[1])
        dst_x = slice(0, image.shape[1] + dx)

    if image.ndim == 2:
        result[dst_y, dst_x] = image[src_y, src_x]
    else:
        result[dst_y, dst_x, :] = image[src_y, src_x, :]

    return result


def align_bayer_integer(
    frame_paths: list[Path],
    reference_bayer: np.ndarray,
    reference_path: str,
    max_failures: int = 50,
    show_progress: bool = True,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Align raw Bayer frames using integer-only, phase-preserving shifts.

    This is the CORRECT approach for OSC sensors when color fidelity is
    paramount. By using only even integer shifts and no interpolation,
    the RGGB phase is preserved exactly.

    Parameters
    ----------
    frame_paths : list[Path]
        Paths to raw Bayer frames.
    reference_bayer : np.ndarray
        Reference Bayer frame (2D array).
    reference_path : str
        Path to reference frame (for logging).
    max_failures : int, default 50
        Maximum consecutive failures before aborting.
    show_progress : bool, default True
        Show progress bar.

    Returns
    -------
    tuple[list[np.ndarray], list[dict]]
        (aligned_bayer_frames, shift_info)
        shift_info contains dict with 'path', 'dy', 'dx', 'success' for each.

    Notes
    -----
    The alignment process:
    1. Build superpixel luminance from each Bayer frame (half-res)
    2. Estimate shift using phase cross-correlation
    3. Convert to full-res coordinates and force even parity
    4. Apply shift using integer slicing (NO interpolation)

    After stacking, debayer ONCE at the end. This preserves the R/G/B
    channel ratios from the original sensor data.

    IMPORTANT: This method only handles translation (no rotation/scale).
    For DWARF in EQ mode, translation dominates. If rotation is significant,
    use align_rgb_debayer_first() instead.

    Example
    -------
    >>> ref_bayer = read_fits(ref_path)
    >>> aligned, info = align_bayer_integer(frame_paths, ref_bayer, ref_path)
    >>> stacked_bayer = sigma_clip_mean(np.array(aligned))
    >>> master_rgb = debayer_rggb(stacked_bayer, mode="bilinear")
    >>> # R/G and B/G ratios will match single-frame ratios!
    """
    from .cli_output import create_progress_bar

    aligned_frames = []
    shift_info = []

    # Build reference superpixel luminance for alignment
    ref_luma = bayer_luma_rggb(reference_bayer)

    pbar = create_progress_bar(
        total=len(frame_paths),
        desc="Aligning Bayer (integer)",
        unit="frame",
        disable=not show_progress,
    )

    consecutive_failures = 0

    with pbar:
        for fpath in frame_paths:
            try:
                source_bayer = read_fits(fpath)
            except Exception as e:
                logger.error("Failed to read %s: %s", fpath, e)
                shift_info.append({
                    'path': str(fpath),
                    'dy': 0, 'dx': 0,
                    'success': False,
                    'error': str(e)
                })
                consecutive_failures += 1
                pbar.update(1)
                continue

            # Build source superpixel luminance
            source_luma = bayer_luma_rggb(source_bayer)

            try:
                # Estimate shift in half-res space
                dy_half, dx_half = estimate_integer_shift(
                    source_luma, ref_luma, force_even=True
                )

                # Convert to full-res (multiply by 2)
                dy_full = dy_half * 2
                dx_full = dx_half * 2

                # Apply integer shift (NO interpolation!)
                aligned = apply_integer_shift(source_bayer, dy_full, dx_full)

                aligned_frames.append(aligned)
                shift_info.append({
                    'path': str(fpath),
                    'dy': dy_full, 'dx': dx_full,
                    'success': True
                })
                consecutive_failures = 0

            except Exception as e:
                logger.warning("Alignment failed for %s: %s", fpath.name, e)
                shift_info.append({
                    'path': str(fpath),
                    'dy': 0, 'dx': 0,
                    'success': False,
                    'error': str(e)
                })
                consecutive_failures += 1

            pbar.set_postfix(ok=len(aligned_frames), fail=consecutive_failures)
            pbar.update(1)

            if consecutive_failures >= max_failures:
                logger.error("Aborting: %d consecutive failures", consecutive_failures)
                break

    n_ok = len(aligned_frames)
    n_fail = sum(1 for s in shift_info if not s['success'])
    logger.info("Bayer integer alignment: %d ok, %d failed", n_ok, n_fail)

    return aligned_frames, shift_info


def align_rgb_debayer_first(
    frame_paths: list[Path],
    reference_rgb: np.ndarray,
    reference_path: str,
    debayer_mode: str = "superpixel",
    max_failures: int = 50,
    show_progress: bool = True,
    return_masks: bool = True,
) -> tuple[list[np.ndarray], list[AlignmentResult], list[np.ndarray] | None]:
    """
    Align frames by debayering first, then applying affine transforms to RGB.

    This is the correct approach when rotation/scale correction is needed:
    1. Debayer each frame to RGB (superpixel or bilinear)
    2. Compute luma and find affine transform via astroalign
    3. Apply transform to RGB channels (interpolation is OK on RGB!)
    4. Return aligned frames WITH validity masks for proper stacking

    Parameters
    ----------
    frame_paths : list[Path]
        Paths to raw Bayer frames.
    reference_rgb : np.ndarray
        Reference RGB image (H, W, 3).
    reference_path : str
        Path to reference frame (for logging).
    debayer_mode : str, default "superpixel"
        Debayer mode: "superpixel" (half-res, robust) or "bilinear" (full-res).
    max_failures : int, default 50
        Maximum consecutive failures before aborting.
    show_progress : bool, default True
        Show progress bar.
    return_masks : bool, default True
        If True, return validity masks for each aligned frame. These masks
        are warped with nearest-neighbor interpolation to preserve sharp
        boundaries at the frame edges.

    Returns
    -------
    tuple[list[np.ndarray], list[AlignmentResult], list[np.ndarray] | None]
        (aligned_rgb_frames, all_results, validity_masks)
        validity_masks is None if return_masks=False, otherwise a list of
        float32 masks (1.0=valid, 0.0=invalid) for each aligned frame.

    Notes
    -----
    Unlike align_bayer_frames(), this method:
    - DEBAYERS before alignment (color is preserved in RGB space)
    - Allows full affine transforms (rotation, scale, translation)
    - Interpolation on RGB is safe (doesn't mix color channels)
    - Returns validity masks for mask-aware stacking

    The key insight: interpolation on a DEBAYERED RGB image is fine because
    adjacent pixels are the same "type" (all RGB). Interpolation on a BAYER
    mosaic destroys color because adjacent pixels are DIFFERENT colors.

    The validity masks are essential for proper stacking with rotated frames.
    Use sigma_clip_mask_aware_rgb() with these masks to avoid edge artifacts.
    """
    from .cli_output import create_progress_bar
    from .debayer import debayer_rggb, luminance_from_rgb

    aligned_frames = []
    validity_masks = [] if return_masks else None
    all_results = []
    consecutive_failures = 0

    # Reference luminance for alignment
    ref_luma = luminance_from_rgb(reference_rgb)

    pbar = create_progress_bar(
        total=len(frame_paths),
        desc=f"Aligning RGB ({debayer_mode})",
        unit="frame",
        disable=not show_progress,
    )

    with pbar:
        for fpath in frame_paths:
            # Read and debayer
            try:
                raw_data = read_fits(fpath)
                rgb_data = debayer_rggb(raw_data, mode=debayer_mode)
                source_luma = luminance_from_rgb(rgb_data)
            except Exception as e:
                logger.error("Failed to read/debayer %s: %s", fpath, e)
                result = AlignmentResult(
                    path=str(fpath),
                    success=False,
                    aligned_data=None,
                    transform=AlignmentTransform(
                        source_path=str(fpath),
                        reference_path=reference_path,
                        success=False,
                        error_message=f"Read/debayer error: {e}",
                    ),
                )
                all_results.append(result)
                consecutive_failures += 1
                pbar.update(1)
                continue

            # Find transform using luminance
            transform = find_transform(
                source_luma,
                ref_luma,
                source_path=str(fpath),
                reference_path=reference_path,
            )

            if transform.success:
                # Apply transform to RGB (this is SAFE - RGB, not Bayer!)
                # Get validity mask with nearest-neighbor warping
                if return_masks:
                    aligned_rgb, mask = apply_transform_to_image(
                        rgb_data,
                        transform.matrix,
                        output_shape=reference_rgb.shape,
                        return_mask=True,
                    )
                    validity_masks.append(mask)
                else:
                    aligned_rgb = apply_transform_to_image(
                        rgb_data,
                        transform.matrix,
                        output_shape=reference_rgb.shape,
                    )

                result = AlignmentResult(
                    path=str(fpath),
                    success=True,
                    aligned_data=aligned_rgb,
                    transform=transform,
                )
                aligned_frames.append(aligned_rgb)
                consecutive_failures = 0

            else:
                result = AlignmentResult(
                    path=str(fpath),
                    success=False,
                    aligned_data=None,
                    transform=transform,
                )
                consecutive_failures += 1

            all_results.append(result)

            n_success = len(aligned_frames)
            n_total = len(all_results)
            pbar.set_postfix(
                ok=f"{n_success}/{n_total}",
                rate=f"{100*n_success/n_total:.0f}%",
            )
            pbar.update(1)

            if consecutive_failures >= max_failures:
                logger.error("Aborting: %d consecutive failures", consecutive_failures)
                break

    n_success = len(aligned_frames)
    n_failed = sum(1 for r in all_results if not r.success)
    logger.info("RGB debayer-first alignment: %d ok, %d failed", n_success, n_failed)

    return aligned_frames, all_results, validity_masks


def align_bayer_frames(
    frame_paths: list[Path],
    reference_bayer: np.ndarray,
    reference_path: str,
    max_failures: int = 50,
    show_progress: bool = True,
    on_aligned: Callable[[np.ndarray, str], None] | None = None,
) -> tuple[list[np.ndarray], list[AlignmentResult]]:
    """
    Align raw Bayer frames using green luminance proxy.

    This is the correct approach for OSC (one-shot color) sensors:
    1. Extract green-based luminance (half-res) from each frame
    2. Compute alignment transforms in half-res space
    3. Scale transforms for full-res application
    4. Apply transforms to full-res Bayer mosaic

    This ensures geometric consistency: transforms are computed and
    applied in properly scaled coordinate systems.

    Parameters
    ----------
    frame_paths : list[Path]
        Paths to raw Bayer frames.
    reference_bayer : np.ndarray
        Reference Bayer frame (2D array).
    reference_path : str
        Path to reference frame (for logging).
    max_failures : int, default 50
        Maximum consecutive failures before aborting.
    show_progress : bool, default True
        Show progress bar.
    on_aligned : callable, optional
        Callback function(aligned_data, path) called after each
        successful alignment.

    Returns
    -------
    tuple[list[np.ndarray], list[AlignmentResult]]
        (aligned_bayer_frames, all_results)
        aligned_bayer_frames contains only successful alignments.
        all_results contains results for all attempted frames.

    Notes
    -----
    After alignment, stack the aligned Bayer frames with sigma-clip,
    then debayer ONCE at the end. This avoids resolution mismatches
    that cause color artifacts.

    Example
    -------
    >>> ref_bayer = read_fits(ref_path)
    >>> aligned, results = align_bayer_frames(frame_paths, ref_bayer, ref_path)
    >>> stacked_bayer = sigma_clip_mean(np.array(aligned))
    >>> master_rgb = debayer_rggb(stacked_bayer, mode="bilinear")
    """
    from .cli_output import create_progress_bar

    aligned_frames = []
    all_results = []
    consecutive_failures = 0

    # Extract reference luminance (half-res) for alignment
    ref_luma = bayer_luma_rggb(reference_bayer)
    ref_shape = reference_bayer.shape

    pbar = create_progress_bar(
        total=len(frame_paths),
        desc="Aligning Bayer",
        unit="frame",
        disable=not show_progress,
    )

    with pbar:
        for fpath in frame_paths:
            # Read raw Bayer frame
            try:
                source_bayer = read_fits(fpath)
            except Exception as e:
                logger.error("Failed to read %s: %s", fpath, e)
                result = AlignmentResult(
                    path=str(fpath),
                    success=False,
                    aligned_data=None,
                    transform=AlignmentTransform(
                        source_path=str(fpath),
                        reference_path=reference_path,
                        success=False,
                        error_message=f"Read error: {e}",
                    ),
                )
                all_results.append(result)
                consecutive_failures += 1
                pbar.update(1)
                continue

            # Extract source luminance (half-res)
            source_luma = bayer_luma_rggb(source_bayer)

            # Find transform in half-res space (consistent!)
            transform = find_transform(
                source_luma,
                ref_luma,
                source_path=str(fpath),
                reference_path=reference_path,
            )

            if transform.success:
                # Scale transform for full-res Bayer application
                scaled_matrix = scale_transform_for_resolution(
                    transform.matrix, scale_factor=2.0
                )

                # Apply scaled transform to full-res Bayer
                aligned_bayer = apply_transform_to_image(
                    source_bayer,
                    scaled_matrix,
                    output_shape=ref_shape,
                )

                result = AlignmentResult(
                    path=str(fpath),
                    success=True,
                    aligned_data=aligned_bayer,
                    transform=transform,
                )
                aligned_frames.append(aligned_bayer)
                consecutive_failures = 0

                if on_aligned is not None:
                    on_aligned(aligned_bayer, str(fpath))
            else:
                result = AlignmentResult(
                    path=str(fpath),
                    success=False,
                    aligned_data=None,
                    transform=transform,
                )
                consecutive_failures += 1

            all_results.append(result)

            # Update progress bar
            n_success = len(aligned_frames)
            n_total = len(all_results)
            pbar.set_postfix(
                ok=f"{n_success}/{n_total}",
                rate=f"{100*n_success/n_total:.0f}%",
            )
            pbar.update(1)

            # Check abort condition
            if consecutive_failures >= max_failures:
                logger.error(
                    "Aborting alignment: %d consecutive failures",
                    consecutive_failures,
                )
                break

    n_success = len(aligned_frames)
    n_failed = sum(1 for r in all_results if not r.success)
    logger.info(
        "Bayer alignment complete: %d successful, %d failed",
        n_success,
        n_failed,
    )

    return aligned_frames, all_results


def _align_rgb_debayer_single(
    args: tuple,
) -> tuple[AlignmentResult, np.ndarray | None, np.ndarray | None]:
    """
    Worker function for parallel RGB debayer-first alignment.

    Parameters
    ----------
    args : tuple
        (frame_path, reference_rgb, reference_path, debayer_mode)

    Returns
    -------
    tuple[AlignmentResult, np.ndarray | None, np.ndarray | None]
        (result, aligned_rgb_data, validity_mask)
        aligned_rgb_data and validity_mask are None if alignment failed.
    """
    from .debayer import debayer_rggb, luminance_from_rgb

    frame_path, reference_rgb, reference_path, debayer_mode = args

    try:
        # Read and debayer
        raw_data = read_fits(frame_path)
        rgb_data = debayer_rggb(raw_data, mode=debayer_mode)
        source_luma = luminance_from_rgb(rgb_data)

        # Reference luminance
        ref_luma = luminance_from_rgb(reference_rgb)

        # Find transform
        transform = find_transform(
            source_luma,
            ref_luma,
            source_path=str(frame_path),
            reference_path=reference_path,
        )

        if transform.success:
            # Apply transform to RGB with validity mask
            aligned_rgb, mask = apply_transform_to_image(
                rgb_data,
                transform.matrix,
                output_shape=reference_rgb.shape,
                return_mask=True,
            )
            result = AlignmentResult(
                path=str(frame_path),
                success=True,
                aligned_data=None,  # Don't store in result (memory)
                transform=transform,
            )
            return result, aligned_rgb, mask
        else:
            result = AlignmentResult(
                path=str(frame_path),
                success=False,
                aligned_data=None,
                transform=transform,
            )
            return result, None, None

    except Exception as e:
        logger.error("Failed to process %s: %s", frame_path, e)
        result = AlignmentResult(
            path=str(frame_path),
            success=False,
            aligned_data=None,
            transform=AlignmentTransform(
                source_path=str(frame_path),
                reference_path=reference_path,
                success=False,
                error_message=f"Processing error: {e}",
            ),
        )
        return result, None, None


def align_rgb_debayer_first_parallel(
    frame_paths: list[Path],
    reference_rgb: np.ndarray,
    reference_path: str,
    debayer_mode: str = "bilinear",
    max_failures: int = 50,
    show_progress: bool = True,
    workers: int | None = None,
) -> tuple[list[np.ndarray], list[AlignmentResult], list[np.ndarray]]:
    """
    Parallel version of align_rgb_debayer_first.

    Aligns frames by debayering first, then applying affine transforms to RGB.
    Uses ProcessPoolExecutor for parallel processing of independent frames.

    Parameters
    ----------
    frame_paths : list[Path]
        Paths to raw Bayer frames.
    reference_rgb : np.ndarray
        Reference RGB image (H, W, 3), already debayered.
    reference_path : str
        Path to reference frame (for logging).
    debayer_mode : str, default "bilinear"
        Debayer mode: "superpixel" (half-res) or "bilinear" (full-res).
    max_failures : int, default 50
        Maximum total failures before logging warning.
    show_progress : bool, default True
        Show progress bar.
    workers : int or None, default None
        Number of parallel workers. None uses auto-detection (CPU count - 1).

    Returns
    -------
    tuple[list[np.ndarray], list[AlignmentResult], list[np.ndarray]]
        (aligned_rgb_frames, all_results, validity_masks)
        validity_masks contains a float32 mask (1.0=valid, 0.0=invalid)
        for each successfully aligned frame.

    Notes
    -----
    This is the recommended approach for alt-az tracking with field rotation:
    1. Debayer each frame to RGB
    2. Find affine transform via luminance (handles rotation)
    3. Apply transform to RGB with nearest-neighbor mask warping
    4. Stack aligned RGB frames with mask-aware averaging

    The validity masks are essential for proper stacking with rotated frames.
    Use sigma_clip_mask_aware_rgb() with these masks to avoid edge artifacts.

    Memory consideration: Each worker loads one RGB frame (~50MB for DWARF 3).
    With 8 workers, expect ~400MB additional RAM usage.
    """
    from .cli_output import create_progress_bar

    if workers is None:
        workers = DEFAULT_WORKERS

    n_frames = len(frame_paths)

    # Fall back to sequential for small counts
    if workers <= 1 or n_frames < 10:
        return align_rgb_debayer_first(
            frame_paths,
            reference_rgb,
            reference_path,
            debayer_mode=debayer_mode,
            max_failures=max_failures,
            show_progress=show_progress,
            return_masks=True,
        )

    logger.info(
        "Parallel RGB debayer-first alignment: %d frames, %d workers",
        n_frames, workers
    )

    aligned_frames = []
    validity_masks = []
    all_results = []
    failed_count = 0

    # Prepare arguments
    args_list = [
        (path, reference_rgb, reference_path, debayer_mode)
        for path in frame_paths
    ]

    pbar = create_progress_bar(
        total=n_frames,
        desc=f"RGB Align ({workers} workers)",
        unit="frame",
        disable=not show_progress,
    )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(_align_rgb_debayer_single, args): args[0]
            for args in args_list
        }

        with pbar:
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result, aligned_rgb, mask = future.result()
                    all_results.append(result)

                    if result.success and aligned_rgb is not None:
                        aligned_frames.append(aligned_rgb)
                        if mask is not None:
                            validity_masks.append(mask)
                    else:
                        failed_count += 1

                except Exception as e:
                    logger.error("Alignment task failed for %s: %s", path, e)
                    failed_count += 1
                    result = AlignmentResult(
                        path=str(path),
                        success=False,
                        aligned_data=None,
                        transform=AlignmentTransform(
                            source_path=str(path),
                            reference_path=reference_path,
                            success=False,
                            error_message=f"Task error: {e}",
                        ),
                    )
                    all_results.append(result)

                # Update progress
                n_success = len(aligned_frames)
                n_total = len(all_results)
                pbar.set_postfix(
                    ok=f"{n_success}/{n_total}",
                    fail=failed_count,
                )
                pbar.update(1)

    n_success = len(aligned_frames)
    n_failed = sum(1 for r in all_results if not r.success)
    logger.info(
        "Parallel RGB alignment complete: %d successful, %d failed",
        n_success, n_failed
    )

    if failed_count > max_failures:
        logger.warning(
            "High failure rate: %d/%d frames failed alignment",
            failed_count, n_frames
        )

    return aligned_frames, all_results, validity_masks


# =============================================================================
# SHARED MEMORY PARALLEL ALIGNMENT (Zero-Pickle Overhead)
# =============================================================================

# Global references for worker processes (initialized by _init_shm_worker)
_shm_rgb = None
_shm_mask = None
_shm_ref_lum = None
_aligned_rgb_view = None
_masks_view = None
_ref_lum_view = None
_worker_rgb_shape = None
_worker_mask_shape = None
_worker_debayer_mode = None


def _init_shm_worker(
    shm_rgb_name: str,
    shm_mask_name: str,
    shm_ref_name: str,
    rgb_shape: tuple,
    mask_shape: tuple,
    ref_shape: tuple,
    debayer_mode: str,
):
    """Initialize shared memory for worker process."""
    global _shm_rgb, _shm_mask, _shm_ref_lum
    global _aligned_rgb_view, _masks_view, _ref_lum_view
    global _worker_rgb_shape, _worker_mask_shape, _worker_debayer_mode

    _worker_rgb_shape = rgb_shape
    _worker_mask_shape = mask_shape
    _worker_debayer_mode = debayer_mode

    # Connect to shared memory
    _shm_rgb = shared_memory.SharedMemory(name=shm_rgb_name)
    _shm_mask = shared_memory.SharedMemory(name=shm_mask_name)
    _shm_ref_lum = shared_memory.SharedMemory(name=shm_ref_name)

    # Create numpy views
    _aligned_rgb_view = np.ndarray(rgb_shape, dtype=np.float32, buffer=_shm_rgb.buf)
    _masks_view = np.ndarray(mask_shape, dtype=np.float32, buffer=_shm_mask.buf)
    _ref_lum_view = np.ndarray(ref_shape, dtype=np.float32, buffer=_shm_ref_lum.buf)


def _align_shm_worker(args: tuple) -> AlignmentResult:
    """Worker for shared memory alignment."""
    # Note: Using globals for SHM access
    frame_path, idx, reference_path = args

    from .debayer import debayer_rggb, luminance_from_rgb

    try:
        # Read and debayer
        bayer = read_fits(frame_path)
        rgb = debayer_rggb(bayer, mode=_worker_debayer_mode)
        source_luma = luminance_from_rgb(rgb)

        # Get reference from SHM
        ref_luma = _ref_lum_view

        # Find transform
        transform = find_transform(
            source_luma,
            ref_luma,
            source_path=str(frame_path),
            reference_path=reference_path,
        )

        if transform.success:
            # Apply transform to RGB and write directly to SHM
            # We need to warp each channel
            
            # Use skimage warp with output parameter if possible?
            # warp() returns a new array. We must copy it to SHM.
            # To optimize, we could assume align_rgb_debayer_first_shm
            # only needs the transform and we do the warp in main process?
            # NO, the whole point is parallel warping.
            
            # Warp RGB
            aligned_rgb, mask = apply_transform_to_image(
                rgb,
                transform.matrix,
                output_shape=_worker_rgb_shape[1:], # H, W, 3
                return_mask=True,
            )
            
            # Write to SHM slice
            _aligned_rgb_view[idx] = aligned_rgb
            _masks_view[idx] = mask
            
            # Don't return data in result
            return AlignmentResult(
                path=str(frame_path),
                success=True,
                aligned_data=None,
                transform=transform,
            )
        else:
            return AlignmentResult(
                path=str(frame_path),
                success=False,
                aligned_data=None,
                transform=transform,
            )

    except Exception as e:
        logger.error("SHM Worker failed for %s: %s", frame_path, e)
        return AlignmentResult(
            path=str(frame_path),
            success=False,
            aligned_data=None,
            transform=AlignmentTransform(
                source_path=str(frame_path),
                reference_path=reference_path,
                success=False,
                error_message=str(e),
            ),
        )


def align_rgb_debayer_first_shm(
    frame_paths: list[Path],
    reference_rgb: np.ndarray,
    reference_path: str,
    debayer_mode: str = "bilinear",
    max_failures: int = 50,
    show_progress: bool = True,
    workers: int | None = None,
) -> tuple[list[np.ndarray], list[AlignmentResult], list[np.ndarray]]:
    """
    Shared-memory optimized parallel alignment.
    
    This is significantly faster than align_rgb_debayer_first_parallel because:
    1. Reference image is put in shared memory (read once)
    2. Workers write aligned images directly to shared memory (no pickling)
    
    Parameters
    ----------
    frame_paths : list[Path]
        Paths to raw Bayer frames.
    reference_rgb : np.ndarray
        Reference RGB image (H, W, 3), already debayered.
    reference_path : str
        Path to reference frame (for logging).
    debayer_mode : str, default "bilinear"
        Debayer mode: "superpixel" (half-res) or "bilinear" (full-res).
    max_failures : int, default 50
        Maximum total failures before logging warning.
    show_progress : bool, default True
        Show progress bar.
    workers : int or None, default None
        Number of parallel workers. None uses auto-detection.

    Returns
    -------
    tuple[list[np.ndarray], list[AlignmentResult], list[np.ndarray]]
        (aligned_rgb_frames, all_results, validity_masks)
        Note: The returned lists contain copies from SHM, so they are safe to use.
    """
    from .cli_output import create_progress_bar
    from .debayer import luminance_from_rgb

    if workers is None:
        workers = DEFAULT_WORKERS

    n_frames = len(frame_paths)
    
    # Use standard method for small batches or if SHM fails
    if n_frames < 5:
        return align_rgb_debayer_first_parallel(
            frame_paths, reference_rgb, reference_path,
            debayer_mode, max_failures, show_progress, workers
        )

    # 1. Prepare shared memory
    ref_luma = luminance_from_rgb(reference_rgb)
    
    # Shapes
    rgb_shape = (n_frames, *reference_rgb.shape)
    mask_shape = (n_frames, *reference_rgb.shape[:2])
    ref_shape = ref_luma.shape
    
    # Calculate bytes
    rgb_bytes = int(np.prod(rgb_shape) * 4)  # float32
    mask_bytes = int(np.prod(mask_shape) * 4)
    ref_bytes = int(np.prod(ref_shape) * 4)
    
    total_mb = (rgb_bytes + mask_bytes + ref_bytes) / 1024 / 1024
    logger.info("Allocating %.1f MB shared memory for alignment", total_mb)
    
    try:
        shm_rgb = shared_memory.SharedMemory(create=True, size=rgb_bytes)
        shm_mask = shared_memory.SharedMemory(create=True, size=mask_bytes)
        shm_ref = shared_memory.SharedMemory(create=True, size=ref_bytes)
    except Exception as e:
        logger.warning("Failed to allocate shared memory: %s. Falling back to standard parallel.", e)
        return align_rgb_debayer_first_parallel(
            frame_paths, reference_rgb, reference_path,
            debayer_mode, max_failures, show_progress, workers
        )

    try:
        # Initialize ref in SHM
        ref_view = np.ndarray(ref_shape, dtype=np.float32, buffer=shm_ref.buf)
        ref_view[:] = ref_luma[:]
        
        # Prepare args
        args_list = [
            (path, i, reference_path) 
            for i, path in enumerate(frame_paths)
        ]
        
        all_results = []
        
        # Start pool with initializer
        from multiprocessing import Pool, get_context
        ctx = get_context('spawn')
        
        with ctx.Pool(
            processes=workers,
            initializer=_init_shm_worker,
            initargs=(
                shm_rgb.name, shm_mask.name, shm_ref.name,
                rgb_shape, mask_shape, ref_shape, debayer_mode
            )
        ) as pool:
            
            pbar = create_progress_bar(
                total=n_frames,
                desc=f"SHM Align ({workers})",
                unit="frame",
                disable=not show_progress
            )
            
            with pbar:
                for result in pool.imap_unordered(_align_shm_worker, args_list):
                    all_results.append(result)
                    pbar.update(1)
        
        # Collect successful frames from SHM
        # We need to sort results to match indices or just grab based on success
        # The SHM was written by index, so aligned_frames[i] corresponds to frame_paths[i]
        
        aligned_frames = []
        validity_masks = []
        
        # Read from SHM views (in main process)
        shm_rgb_view = np.ndarray(rgb_shape, dtype=np.float32, buffer=shm_rgb.buf)
        shm_mask_view = np.ndarray(mask_shape, dtype=np.float32, buffer=shm_mask.buf)
        
        # Map path back to result to check success
        result_map = {r.path: r for r in all_results}
        
        success_count = 0
        for i, path in enumerate(frame_paths):
            res = result_map.get(str(path))
            if res and res.success:
                # Copy from SHM to local memory
                aligned_frames.append(shm_rgb_view[i].copy())
                validity_masks.append(shm_mask_view[i].copy())
                success_count += 1
                
        logger.info("SHM alignment complete: %d/%d successful", success_count, n_frames)
        
        return aligned_frames, all_results, validity_masks

    finally:
        # cleanup
        shm_rgb.close()
        shm_rgb.unlink()
        shm_mask.close()
        shm_mask.unlink()
        shm_ref.close()
        shm_ref.unlink()
