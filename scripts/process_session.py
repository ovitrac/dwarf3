#!/usr/bin/env python3
"""
Universal processing script for DWARF 3 astrophotography sessions.

Supports both Alt-Az mode (with field rotation) and EQ mode (equatorial mount,
translation-only shifts) with appropriate alignment strategies.

Usage:
    # M31 (Alt-Az mode, full rotation handling)
    python scripts/process_session.py "rawData/DWARF_RAW_TELE_M 31_EXP_15_GAIN_60_2025-12-27-18-26-56-449"

    # M43 (EQ mode, integer-pixel shifts for color fidelity)
    python scripts/process_session.py "rawData/DWARF_RAW_TELE_M 43_EXP_60_GAIN_60_2025-12-28-21-44-39-627" --eq-mode

    # With color enhancement
    python scripts/process_session.py <session> --saturation 1.3 --scnr 0.3

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dwarf3.io import list_lights, read_fits
from dwarf3.quality import rank_frames, select_frames
from dwarf3.debayer import debayer_rggb, bayer_luma_rggb
from dwarf3.align import (
    select_reference_frame,
    align_frames_parallel,
    align_rgb_debayer_first_parallel,
    save_transforms,
    load_transforms,
    apply_integer_shift as dwarf3_apply_integer_shift,
    analyze_field_rotation,
)
from dwarf3.stack import (
    sigma_clip_mean,
    compute_stack_statistics,
    StackStatistics,
    stream_plane_stack,
)
from dwarf3.color import (
    apply_ccm,
    calibrate_rgb,
    scnr_green,
    combine_lrgb,
    stretch_rgb_linked,
    create_extended_object_mask,
)
from dwarf3.processing import (
    process_luminance,
    create_galaxy_mask,
    subtract_background,
)
from dwarf3.config import FrameScore
from dwarf3.utils import __version__, get_version_banner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def detect_mount_mode(session_path: Path) -> str:
    """
    Auto-detect mount mode from session data.

    EQ mode typically has minimal rotation between frames,
    while Alt-Az mode shows field rotation over time.

    Returns 'eq' or 'altaz'.
    """
    # For now, rely on user specification
    # TODO: Implement automatic detection by analyzing transform matrices
    return "altaz"


def round_transform_to_integer(matrix: np.ndarray) -> np.ndarray:
    """
    Round translation components to even integers (Bayer-safe).

    For EQ mode, rounding to even pixel values preserves Bayer grid
    alignment and prevents color smearing.

    Parameters
    ----------
    matrix : np.ndarray
        3x3 affine transform matrix.

    Returns
    -------
    np.ndarray
        Matrix with translations rounded to even integers.
    """
    result = matrix.copy()
    # Translation is in matrix[0, 2] (x) and matrix[1, 2] (y)
    # Round to nearest even integer to preserve RGGB alignment
    result[0, 2] = 2 * round(result[0, 2] / 2)
    result[1, 2] = 2 * round(result[1, 2] / 2)
    # Remove any small rotation/scale (should be ~identity for EQ mode)
    # Check if rotation is minimal (< 0.1 degrees)
    rot_angle = np.arctan2(matrix[1, 0], matrix[0, 0])
    if abs(rot_angle) < np.radians(0.1):
        # Set to pure translation
        result[0, 0] = 1.0
        result[0, 1] = 0.0
        result[1, 0] = 0.0
        result[1, 1] = 1.0
    return result


def apply_integer_shift(
    image: np.ndarray,
    tx: int,
    ty: int,
) -> np.ndarray:
    """
    Apply integer pixel shift (for EQ mode Bayer-safe alignment).

    This is faster and more accurate than interpolation for EQ mode.
    """
    h, w = image.shape[:2]
    shifted = np.zeros_like(image)

    # Compute source and destination slices
    src_y_start = max(0, -ty)
    src_y_end = min(h, h - ty)
    src_x_start = max(0, -tx)
    src_x_end = min(w, w - tx)

    dst_y_start = max(0, ty)
    dst_y_end = min(h, h + ty)
    dst_x_start = max(0, tx)
    dst_x_end = min(w, w + tx)

    if image.ndim == 2:
        shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            image[src_y_start:src_y_end, src_x_start:src_x_end]
    else:
        shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = \
            image[src_y_start:src_y_end, src_x_start:src_x_end, :]

    return shifted


def process_session(
    session_path: Path,
    output_root: Path,
    keep_fraction: float = 0.92,
    sigma: float = 3.0,
    eq_mode: bool = False,
    eq_robust: bool = False,
    saturation_boost: float = 1.0,
    scnr_strength: float = 0.0,
    wb_method: str = "stars",
    ccm_preset: str = "rich",
    workers: int | None = None,
    skip_alignment: bool = False,
) -> dict:
    """
    Process a DWARF 3 session with full color pipeline.

    Parameters
    ----------
    session_path : Path
        Path to raw session folder.
    output_root : Path
        Root output directory.
    keep_fraction : float
        Fraction of frames to keep after quality scoring.
    sigma : float
        Sigma clipping threshold.
    eq_mode : bool
        If True, use EQ mode (auto-switches to robust if needed).
    eq_robust : bool
        If True, force robust plane-based alignment even for EQ.
    saturation_boost : float
        Saturation multiplier (1.0 = unchanged, 1.3 = 30% boost).
    scnr_strength : float
        SCNR green reduction (0 = off, 0.3-0.5 = typical).
    wb_method : str
        White balance method ('stars', 'gray_world', 'background', 'none').
    ccm_preset : str
        Color Correction Matrix preset ('none', 'neutral', 'rich', 'vivid', 'ha_emission').
    workers : int, optional
        Number of parallel workers.
    skip_alignment : bool
        Skip alignment if transforms are cached.

    Returns
    -------
    dict
        Processing statistics and output paths.
    """
    session_name = session_path.name
    output_dir = output_root / session_name
    output_dir.mkdir(parents=True, exist_ok=True)

    stacked_dir = output_dir / "stacked"
    stacked_dir.mkdir(exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info(get_version_banner())
    logger.info("=" * 60)
    logger.info("Processing session: %s", session_name)
    logger.info("Mode: %s", "EQ (auto)" if eq_mode else "Alt-Az (full affine)")
    logger.info("=" * 60)

    start_time = time.time()

    # Step 1: Discover and score frames
    logger.info("\n[1/6] Discovering frames...")
    frame_paths = list_lights(session_path)
    n_discovered = len(frame_paths)
    logger.info("Found %d valid frames", n_discovered)

    if n_discovered == 0:
        raise ValueError(f"No valid frames found in {session_path}")

    # Step 2: Quality scoring
    logger.info("\n[2/6] Scoring frame quality...")
    scores_file = cache_dir / "scores.json"

    scores_list = None
    if scores_file.exists():
        logger.info("Loading cached scores...")
        with open(scores_file) as f:
            scores_data = json.load(f)
        # Reconstruct FrameScore objects
        scores_list = [
            FrameScore(
                path=d["path"],
                background_median=d.get("background_median", 0),
                background_mad=d.get("background_mad", 0),
                noise_proxy=d.get("noise_proxy", 0),
                saturation_fraction=d.get("saturation_fraction", 0),
                composite_score=d.get("composite_score", d.get("score", 0)),
            )
            for d in scores_data["scores"]
        ]
    else:
        # Compute scores using rank_frames
        scores_list = rank_frames(frame_paths, workers=workers)
        # Cache scores
        scores_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_frames": len(scores_list),
            "scores": [
                {
                    "path": s.path,
                    "composite_score": s.composite_score,
                    "background_median": s.background_median,
                    "background_mad": s.background_mad,
                    "noise_proxy": s.noise_proxy,
                    "saturation_fraction": s.saturation_fraction,
                }
                for s in scores_list
            ],
        }
        with open(scores_file, "w") as f:
            json.dump(scores_data, f, indent=2)

    # Select best frames using select_frames
    kept_scores, rejected_scores = select_frames(scores_list, keep_fraction=keep_fraction)
    kept_frames = [s.path for s in kept_scores]
    rejected_frames = [s.path for s in rejected_scores]

    # Build score lookup
    scores = {s.path: s.composite_score for s in scores_list}

    logger.info("Keeping %d frames (%.0f%% of %d)",
                len(kept_frames), keep_fraction * 100, n_discovered)

    # Step 3: Find reference and compute transforms
    logger.info("\n[3/6] Computing alignment transforms...")
    transforms_file = cache_dir / "transforms.json"

    # Select reference frame (best quality)
    kept_paths = [Path(p) for p in kept_frames]
    ref_path, ref_data_bayer = select_reference_frame(kept_paths, scores=kept_scores, method="best")
    ref_path = str(ref_path)
    logger.info("Reference frame: %s", Path(ref_path).name)

    # Determine alignment strategy
    detected_rotation = 0.0
    use_robust_eq = False

    if eq_mode:
        if eq_robust:
            logger.info("EQ-Robust mode requested: forcing plane-based affine alignment")
            use_robust_eq = True
        else:
            # Auto-detect rotation
            logger.info("Analyzing field rotation (sample size=5)...")
            detected_rotation = analyze_field_rotation(kept_paths, ref_path)
            logger.info("Detected max rotation: %.3f°", detected_rotation)
            
            if detected_rotation > 0.1:
                logger.warning("Rotation > 0.1° detected! Switching to Robust EQ (Plane-Based Affine).")
                use_robust_eq = True
            else:
                logger.info("Rotation negligible (< 0.1°). Using fast Integer EQ.")
                use_robust_eq = False

    if transforms_file.exists() and skip_alignment:
        logger.info("Loading cached transforms...")
        transforms_data, _, _ = load_transforms(transforms_file)
        # Filter to kept frames
        transforms = [t for t in transforms_data if t.source_path in kept_frames or not t.source_path]
    else:
        # Pre-debayer reference for robust alignment (superpixel is fine for alignment)
        ref_rgb_super = debayer_rggb(ref_data_bayer, mode="superpixel")

        # Compute transforms using robust RGB alignment (superpixel proxy)
        # This handles rotation correctly even if we later round to integers
        aligned_frames, all_results, _ = align_rgb_debayer_first_parallel(
            [Path(p) for p in kept_frames],
            reference_rgb=ref_rgb_super,
            reference_path=ref_path,
            debayer_mode="superpixel", # Fast and robust
            workers=workers,
        )

        # Extract transforms from results
        transforms = [r.transform for r in all_results if r.success]
        failures = [r for r in all_results if not r.success]

        # Save transforms
        save_transforms(transforms, transforms_file, ref_path)
        logger.info("Alignment: %d succeeded, %d failed",
                    len(transforms), len(failures))

    # Apply EQ-specific rounding if NOT using robust mode
    if eq_mode and not use_robust_eq:
        logger.info("EQ mode (Standard): rounding transforms to even-pixel shifts...")
        for t in transforms:
            if t.matrix is not None:
                t.matrix = round_transform_to_integer(t.matrix)
    elif eq_mode and use_robust_eq:
        logger.info("EQ mode (Robust): preserving affine rotation/scale")

    # Step 4: Stack using plane-based approach (preserves color)
    logger.info("\n[4/6] Stacking frames with plane-based approach...")

    # Filter to successful transforms
    valid_transforms = [t for t in transforms if t.success and t.matrix is not None]
    valid_paths = [t.source_path for t in valid_transforms]

    logger.info("Stacking %d frames...", len(valid_paths))

    # Use stream_plane_stack for memory-efficient stacking
    stacked_rgb, coverage = stream_plane_stack(
        frame_paths=valid_paths,
        transforms=valid_transforms,
        reference_path=ref_path,
        sigma=sigma,
        show_progress=True,
        coverage_threshold=0.5,
        chroma_cleanup=True,
    )

    # Get exposure time for statistics
    with fits.open(ref_path) as hdul:
        exptime = hdul[0].header.get("EXPTIME", 15.0)

    # Compute statistics
    stats = StackStatistics(
        n_frames=len(valid_paths),
        total_exposure_s=len(valid_paths) * exptime,
        mean_clipped_fraction=0.03,  # Approximate from stream_plane_stack
        snr_proxy=float(np.median(stacked_rgb) / (1.4826 * np.median(np.abs(stacked_rgb - np.median(stacked_rgb))))),
    )

    logger.info("Stack complete: %d frames, %.1fs total exposure",
                stats.n_frames, stats.total_exposure_s)

    # Step 5: Color calibration and enhancement
    logger.info("\n[5/6] Color calibration and enhancement...")

    # Create galaxy mask for background protection
    galaxy_mask = create_extended_object_mask(stacked_rgb)

    # Handle different white balance methods
    if wb_method == "background":
        # Background-based WB: sample sky background and neutralize green bias
        # Use corner regions to avoid nebula/galaxy
        h, w = stacked_rgb.shape[:2]
        bg_regions = [
            stacked_rgb[50:200, 50:300, :],      # top-left
            stacked_rgb[50:200, w-300:w-50, :],  # top-right
        ]
        bg_sample = np.concatenate([r.reshape(-1, 3) for r in bg_regions], axis=0)
        # Exclude zeros (masked pixels)
        valid_bg = bg_sample[bg_sample.sum(axis=1) > 0]

        if len(valid_bg) > 100:
            r_mean = np.median(valid_bg[:, 0])
            g_mean = np.median(valid_bg[:, 1])
            b_mean = np.median(valid_bg[:, 2])
            target = (r_mean + b_mean) / 2
            g_correction = target / g_mean if g_mean > 0 else 1.0

            logger.info("Background WB: R=%.1f, G=%.1f, B=%.1f -> G_corr=%.3f",
                        r_mean, g_mean, b_mean, g_correction)

            calibrated_rgb = stacked_rgb.copy()
            calibrated_rgb[:, :, 1] *= g_correction

            # Create dummy cal_params for reporting
            from dataclasses import dataclass
            @dataclass
            class DummyCalParams:
                wb_gain_r: float = 1.0
                wb_gain_g: float = 1.0
                wb_gain_b: float = 1.0
            cal_params = DummyCalParams(1.0, g_correction, 1.0)
        else:
            logger.warning("Not enough background pixels, using no WB")
            calibrated_rgb = stacked_rgb.copy()
            from dataclasses import dataclass
            @dataclass
            class DummyCalParams:
                wb_gain_r: float = 1.0
                wb_gain_g: float = 1.0
                wb_gain_b: float = 1.0
            cal_params = DummyCalParams()
    else:
        # Use standard calibrate_rgb for other methods
        calibrated_rgb, cal_params = calibrate_rgb(
            stacked_rgb,
            object_mask=galaxy_mask,
            wb_method=wb_method,
            bg_mode="none",  # Preserve galaxy halo
        )

    logger.info("White balance: R=%.3f, G=%.3f, B=%.3f",
                cal_params.wb_gain_r, cal_params.wb_gain_g, cal_params.wb_gain_b)

    # Apply Color Correction Matrix (CCM) to restore color saturation
    # CCM transforms Camera RGB to sRGB, correcting for sensor spectral response
    if ccm_preset and ccm_preset.lower() != "none":
        logger.info("Applying Color Correction Matrix: %s", ccm_preset)
        calibrated_rgb = apply_ccm(
            calibrated_rgb,
            matrix=ccm_preset,
            clip_negatives=True,
            preserve_luminance=True,
        )

    # Apply SCNR if requested
    if scnr_strength > 0:
        logger.info("Applying SCNR green reduction (strength=%.2f)...", scnr_strength)
        calibrated_rgb = scnr_green(calibrated_rgb, strength=scnr_strength)

    # Stretch for visualization
    # IMPORTANT: Exclude zero pixels (masked regions) when computing percentiles
    # Otherwise black_point becomes 0 and sky background appears gray
    luma = 0.299 * calibrated_rgb[:, :, 0] + 0.587 * calibrated_rgb[:, :, 1] + 0.114 * calibrated_rgb[:, :, 2]
    valid_mask = luma > 0
    valid_luma = luma[valid_mask]

    black_point = np.percentile(valid_luma, 5.0)  # Use 5% to get sky background (skip edge artifacts)
    white_point = np.percentile(valid_luma, 99.9)

    logger.info("Stretch: black=%.1f, white=%.1f (from valid pixels only)", black_point, white_point)

    # Manual stretch excluding zeros
    normalized = (calibrated_rgb - black_point) / (white_point - black_point)
    normalized = np.clip(normalized, 0, None)

    # Asinh stretch
    asinh_a = 0.1
    stretched_rgb = np.arcsinh(normalized / asinh_a) / np.arcsinh(1.0 / asinh_a)
    stretched_rgb = stretched_rgb / np.max(stretched_rgb)
    stretched_rgb = np.clip(stretched_rgb, 0, 1).astype(np.float32)

    # Apply saturation boost if requested
    if saturation_boost != 1.0:
        logger.info("Boosting saturation by %.0f%%...", (saturation_boost - 1) * 100)
        # Convert to HLS and boost saturation
        from skimage.color import rgb2hsv, hsv2rgb
        hsv = rgb2hsv(stretched_rgb)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 1)
        stretched_rgb = hsv2rgb(hsv).astype(np.float32)

    # Step 6: Save outputs
    logger.info("\n[6/6] Saving outputs...")

    # Save linear FITS (science master)
    linear_fits_path = stacked_dir / "master_linear.fits"
    hdu = fits.PrimaryHDU(stacked_rgb.astype(np.float32))
    hdu.header["NFRAMES"] = len(valid_paths)
    hdu.header["EXPTIME"] = stats.total_exposure_s
    hdu.header["HISTORY"] = f"Stacked with dwarf3 ({datetime.now().isoformat()})"
    hdu.writeto(linear_fits_path, overwrite=True)
    logger.info("Saved: %s", linear_fits_path)

    # Save calibrated RGB FITS
    rgb_fits_path = stacked_dir / "master_lrgb.fits"
    hdu = fits.PrimaryHDU(calibrated_rgb.astype(np.float32))
    hdu.header["NFRAMES"] = len(valid_paths)
    hdu.header["WB_R"] = cal_params.wb_gain_r
    hdu.header["WB_G"] = cal_params.wb_gain_g
    hdu.header["WB_B"] = cal_params.wb_gain_b
    hdu.writeto(rgb_fits_path, overwrite=True)

    # Save quicklook PNG
    png_path = stacked_dir / "master_lrgb_galaxy_v2.png"
    from PIL import Image
    img_uint8 = (np.clip(stretched_rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(png_path)
    logger.info("Saved: %s", png_path)

    # Save 16-bit TIF (use tifffile for 16-bit RGB support)
    tif_path = stacked_dir / "master_lrgb_galaxy_v2.tif"
    img_uint16 = (np.clip(stretched_rgb, 0, 1) * 65535).astype(np.uint16)
    try:
        import tifffile
        tifffile.imwrite(tif_path, img_uint16)
        logger.info("Saved: %s", tif_path)
    except ImportError:
        # Fallback: save as 8-bit TIF via PIL
        logger.warning("tifffile not available, saving 8-bit TIF instead")
        Image.fromarray(img_uint8).save(tif_path)

    # Generate report
    elapsed = time.time() - start_time
    report_data = {
        "version": __version__,
        "session_id": session_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "eq" if eq_mode else "altaz",
        "alignment": {
            "rotation_deg": detected_rotation,
            "strategy": "robust_affine" if use_robust_eq else "integer_shift",
        },
        "summary": {
            "frames_discovered": n_discovered,
            "frames_kept": len(kept_frames),
            "frames_stacked": len(valid_paths),
            "total_exposure_s": float(stats.total_exposure_s),
            "processing_time_s": float(elapsed),
            "snr_proxy": float(stats.snr_proxy),
        },
        "color": {
            "wb_method": wb_method,
            "wb_r": float(cal_params.wb_gain_r),
            "wb_g": float(cal_params.wb_gain_g),
            "wb_b": float(cal_params.wb_gain_b),
            "ccm_preset": ccm_preset if ccm_preset else "none",
            "saturation_boost": float(saturation_boost),
            "scnr_strength": float(scnr_strength),
        },
        "outputs": {
            "master_linear": str(linear_fits_path),
            "master_lrgb": str(rgb_fits_path),
            "quicklook_png": str(png_path),
            "quicklook_tif": str(tif_path),
        },
    }

    report_json = output_dir / "report.json"
    with open(report_json, "w") as f:
        json.dump(report_data, f, indent=2)

    # Generate simple markdown report
    report_md = output_dir / "report.md"
    with open(report_md, "w") as f:
        f.write(f"# Processing Report: {session_name}\n\n")
        f.write(f"**dwarf3 version:** {__version__}\n\n")
        f.write(f"**Generated:** {report_data['timestamp']}\n\n")
        f.write(f"**Mode:** {report_data['mode'].upper()}\n\n")
        f.write("## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Frames discovered | {n_discovered} |\n")
        f.write(f"| Frames kept | {len(kept_frames)} |\n")
        f.write(f"| Frames stacked | {len(valid_paths)} |\n")
        f.write(f"| Alignment strategy | {report_data['alignment']['strategy']} |\n")
        f.write(f"| Detected rotation | {detected_rotation:.3f}° |\n")
        f.write(f"| Total exposure | {stats.total_exposure_s:.0f}s ({stats.total_exposure_s/60:.1f} min) |\n")
        f.write(f"| Processing time | {elapsed:.1f}s |\n")
        f.write(f"| SNR proxy | {stats.snr_proxy:.1f} |\n\n")
        f.write("## Color Calibration\n\n")
        f.write(f"| Parameter | Value |\n")
        f.write(f"|-----------|-------|\n")
        f.write(f"| White balance method | {wb_method} |\n")
        f.write(f"| WB R gain | {cal_params.wb_gain_r:.3f} |\n")
        f.write(f"| WB G gain | {cal_params.wb_gain_g:.3f} |\n")
        f.write(f"| WB B gain | {cal_params.wb_gain_b:.3f} |\n")
        f.write(f"| Saturation boost | {saturation_boost:.2f} |\n")
        f.write(f"| SCNR strength | {scnr_strength:.2f} |\n\n")
        f.write("## Outputs\n\n")
        for name, path in report_data["outputs"].items():
            f.write(f"- **{name}:** `{path}`\n")

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE!")
    logger.info("  Frames stacked: %d", len(valid_paths))
    logger.info("  Total exposure: %.0f min", stats.total_exposure_s / 60)
    logger.info("  Processing time: %.1f min", elapsed / 60)
    logger.info("  SNR proxy: %.1f", stats.snr_proxy)
    logger.info("=" * 60)

    return report_data


def main():
    parser = argparse.ArgumentParser(
        description="Process DWARF 3 astrophotography session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process M31 (Alt-Az mode, rotation handling)
  python scripts/process_session.py rawData/DWARF_RAW_TELE_M\ 31_*

  # Process M43 (EQ mode, auto-switches if rotation > 0.1 deg)
  python scripts/process_session.py rawData/DWARF_RAW_TELE_M\ 43_* --eq-mode

  # Force robust alignment for EQ (always checks rotation)
  python scripts/process_session.py <session> --eq-mode --eq-robust

  # Enhanced colors
  python scripts/process_session.py <session> --saturation 1.3 --scnr 0.3

Mount Modes:
  Alt-Az:  Significant field rotation. Uses debayer-first + affine transforms.
  EQ:      Minimal rotation. Uses integer-pixel Bayer-safe shifts (or robust if needed).
        """,
    )
    parser.add_argument("session", type=Path, help="Path to raw session folder")
    parser.add_argument("--out", type=Path, default=Path("processedData"),
                        help="Output root directory (default: processedData)")
    parser.add_argument("--keep", type=float, default=0.92,
                        help="Fraction of frames to keep (default: 0.92)")
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="Sigma clipping threshold (default: 3.0)")
    parser.add_argument("--eq-mode", action="store_true",
                        help="Use EQ mode (integer-pixel alignment)")
    parser.add_argument("--eq-robust", action="store_true",
                        help="Force robust alignment for EQ mode (handles residual rotation)")
    parser.add_argument("--saturation", type=float, default=1.0,
                        help="Saturation boost factor (default: 1.0)")
    parser.add_argument("--scnr", type=float, default=0.0,
                        help="SCNR green reduction strength (default: 0, off)")
    parser.add_argument("--wb", choices=["stars", "gray_world", "background", "none"],
                        default="background", help="White balance method (default: background)")
    parser.add_argument("--ccm", choices=["none", "neutral", "rich", "vivid", "ha_emission"],
                        default="rich", help="Color Correction Matrix preset (default: rich)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto)")
    parser.add_argument("--skip-align", action="store_true",
                        help="Skip alignment if cached transforms exist")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.session.exists():
        print(f"Error: Session path does not exist: {args.session}")
        sys.exit(1)

    try:
        result = process_session(
            session_path=args.session,
            output_root=args.out,
            keep_fraction=args.keep,
            sigma=args.sigma,
            eq_mode=args.eq_mode,
            eq_robust=args.eq_robust,
            saturation_boost=args.saturation,
            scnr_strength=args.scnr,
            wb_method=args.wb,
            ccm_preset=args.ccm,
            workers=args.workers,
            skip_alignment=args.skip_align,
        )

        print(f"\nOutputs saved to: {args.out / args.session.name}")
        print(f"  - {result['outputs']['quicklook_png']}")

    except Exception as e:
        logger.exception("Processing failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()