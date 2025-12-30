"""
Command-line interface for dwarf3 stacking pipeline.

Usage:
    python -m dwarf3 stack <session_path> [options]
    dwarf3 stack <session_path> [options]

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from functools import partial
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from .align import (
    align_bayer_frames,
    align_bayer_integer,
    align_frames,
    align_frames_parallel,
    align_rgb_debayer_first,
    align_rgb_debayer_first_shm,
    align_rgb_frames,
    apply_cached_transforms,
    load_transforms,
    save_transforms,
    select_reference_frame,
)
from .cli_output import (
    Colors,
    PipelineProgress,
    Symbols,
    create_progress_bar,
    format_duration as fmt_duration,
    print_banner,
    print_error,
    print_header,
    print_info,
    print_metric,
    print_path,
    print_stage,
    print_success,
    print_summary_box,
    print_warning,
    setup_terminal,
)
from .config import RejectedFrame, RejectionReason, StackConfig, StackResult
from .debayer import debayer_rggb, luminance_from_rgb
from .io import list_lights, read_fits, read_header, write_fits
from .quality import rank_frames, select_frames
from .report import write_all_reports
from .stack import (
    compute_stack_statistics,
    compute_stack_statistics_rgb,
    sigma_clip_mean,
    sigma_clip_mean_rgb,
    sigma_clip_mask_aware_rgb_streaming,
    stream_fullres_stack,
    stream_plane_stack,
)
from .backend import get_backend_summary, is_gpu_available
from .cache import get_session_cache, CACHE_DIR
from .utils import (
    asinh_stretch,
    ensure_output_dir,
    format_duration,
    get_platform_info,
    get_timestamp_iso,
    get_version,
    to_uint8,
    to_uint16,
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_frame_list(frame_list_path: str | Path, session_path: Path) -> list[Path] | None:
    """
    Load frame list from file.

    Parameters
    ----------
    frame_list_path : str or Path
        Path to file containing frame names/paths (one per line).
    session_path : Path
        Session directory for resolving relative paths.

    Returns
    -------
    list[Path] or None
        List of frame paths, or None if file not found.
    """
    frame_list_path = Path(frame_list_path)
    if not frame_list_path.exists():
        logger.warning("Frame list file not found: %s", frame_list_path)
        return None

    frames = []
    with open(frame_list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Try as absolute path first
            frame_path = Path(line)
            if not frame_path.is_absolute():
                # Try relative to session
                frame_path = session_path / line

            if frame_path.exists():
                frames.append(frame_path)
            else:
                logger.warning("Frame not found: %s", line)

    logger.info("Loaded %d frames from list: %s", len(frames), frame_list_path)
    return frames


def stack_session(
    session_path: str | Path,
    output_root: str | Path = "processedData",
    config: StackConfig | None = None,
    dry_run: bool = False,
    quiet: bool = False,
    save_transforms: bool = False,
    load_transforms_path: str | Path | None = None,
    no_cache: bool = False,
    cache_refresh: bool = False,
    frame_list_path: str | Path | None = None,
) -> StackResult:
    """
    Execute the full stacking pipeline for a session.

    Parameters
    ----------
    session_path : str or Path
        Path to the raw session folder.
    output_root : str or Path, default "processedData"
        Root directory for outputs.
    config : StackConfig, optional
        Configuration. Uses defaults if not provided.
    dry_run : bool, default False
        If True, only score and report, no actual stacking.
    quiet : bool, default False
        If True, suppress colored output (use logging only).
    save_transforms : bool, default False
        If True, save alignment transforms to transforms.json for reuse.
    load_transforms_path : str or Path, optional
        Path to cached transforms JSON file. If provided, skips alignment
        computation and uses cached transforms instead.
    no_cache : bool, default False
        If True, disable all caching (compute fresh, don't save).
    cache_refresh : bool, default False
        If True, clear existing cache before starting.
    frame_list_path : str or Path, optional
        Path to file containing frame names to include.

    Returns
    -------
    StackResult
        Complete result with paths to outputs and statistics.
    """
    start_time = time.time()
    session_path = Path(session_path).resolve()
    output_root = Path(output_root).resolve()

    if config is None:
        config = StackConfig()
    config.validate()

    session_id = session_path.name

    # Setup terminal and show banner
    if not quiet:
        setup_terminal()
        print_banner(get_version())
        print_header(f"Session: {session_id}")
        print_metric("Mode", config.debayer if config.debayer != "none" else "mono (Bayer)")
        print_metric("Keep fraction", f"{config.keep_fraction:.0%}")
        print_metric("Sigma clip", config.sigma)
        if config.use_gpu:
            if is_gpu_available():
                print_metric("Backend", get_backend_summary())
            else:
                print_warning("GPU requested but not available, using CPU")

    logger.info("Processing session: %s", session_id)

    # Initialize result
    result = StackResult(
        session_id=session_id,
        session_path=str(session_path),
        config=config,
        version=get_version(),
        platform=get_platform_info(),
    )

    # Initialize cache
    output_dir = ensure_output_dir(output_root, session_id)
    cache = get_session_cache(output_dir, session_id)

    # Handle cache refresh
    if cache_refresh and cache.cache_dir.exists():
        cache.clear()
        logger.info("Cache cleared (--cache-refresh)")
        if not quiet:
            print_info("Cache cleared")

    # Initialize progress tracker
    progress = PipelineProgress(total_stages=7, quiet=quiet)

    # --- Stage 1: Discovery ---
    progress.start_stage(1, "Frame Discovery", Symbols.FOLDER)
    lights = list_lights(
        session_path,
        exclude_failed=config.exclude_failed_prefix,
        exclude_stacked=config.exclude_stacked_prefix,
    )

    # Filter by frame list if provided
    if frame_list_path:
        frame_list = load_frame_list(frame_list_path, session_path)
        if frame_list:
            original_count = len(lights)
            frame_names = {p.name for p in frame_list}
            lights = [p for p in lights if p.name in frame_names]
            progress.update_detail(f"Filtered: {len(lights)}/{original_count} from frame list")
            logger.info("Frame list filter: %d -> %d frames", original_count, len(lights))

    result.inputs = [str(p) for p in lights]

    if len(lights) == 0:
        progress.fail_stage("No light frames found!")
        print_error("No light frames found in session")
        result.timestamp = get_timestamp_iso()
        return result

    # Record excluded frames as rejected
    all_fits = sorted(session_path.glob("*.fits"))
    n_excluded = 0
    for fpath in all_fits:
        if fpath not in lights:
            n_excluded += 1
            if fpath.name.startswith("failed_"):
                result.rejected.append(
                    RejectedFrame(str(fpath), RejectionReason.FAILED_PREFIX)
                )
            elif fpath.name.startswith("stacked-"):
                result.rejected.append(
                    RejectedFrame(str(fpath), RejectionReason.STACKED_PREFIX)
                )

    progress.update_detail(f"Found {len(lights)} light frames")
    if n_excluded > 0:
        progress.update_detail(f"Excluded {n_excluded} (failed/stacked)")
    progress.complete_stage(f"{len(lights)} frames ready")

    # --- Stage 2: Quality Assessment ---
    progress.start_stage(2, "Quality Scoring", Symbols.CHART)

    # Try to load cached scores (unless no_cache)
    scores = None
    if not no_cache and cache.has_scores():
        scores = cache.load_scores()
        if scores:
            # Validate cache matches current frame list
            cached_paths = {Path(s.path).name for s in scores}
            current_paths = {p.name for p in lights}
            if cached_paths == current_paths:
                progress.update_detail("Using cached scores")
                logger.info("Loaded %d cached scores", len(scores))
            else:
                logger.info("Cache invalidated: frame list changed")
                scores = None

    # Compute scores if not cached
    if scores is None:
        scores = rank_frames(
            lights,
            saturation_threshold=config.saturation_threshold,
            workers=config.workers,
            show_progress=not quiet,
        )
        # Save to cache (unless no_cache)
        if not no_cache:
            cache.save_scores(scores)
            progress.update_detail("Saved scores to cache")

    result.scores = scores

    if scores:
        progress.update_detail(f"Best score: {scores[0].composite_score:.4f}")
        progress.update_detail(f"Worst score: {scores[-1].composite_score:.4f}")
    progress.complete_stage(f"Scored {len(scores)} frames")

    # --- Stage 3: Frame Selection ---
    progress.start_stage(3, "Frame Selection", Symbols.STAR)
    kept_scores, rejected_scores = select_frames(scores, config.keep_fraction)

    # Record quality-rejected frames
    for score in rejected_scores:
        result.rejected.append(
            RejectedFrame(
                score.path,
                RejectionReason.LOW_QUALITY_SCORE,
                f"score={score.composite_score:.4f}",
            )
        )

    kept_paths = [Path(s.path) for s in kept_scores]
    progress.update_detail(f"Keeping {len(kept_paths)} frames ({config.keep_fraction:.0%})")
    if rejected_scores:
        progress.update_detail(f"Rejected {len(rejected_scores)} low-quality frames")
    progress.complete_stage(f"{len(kept_paths)} frames selected")

    if dry_run:
        print_warning("Dry run - skipping alignment and stacking")
        result.kept = [str(p) for p in kept_paths]
        result.timestamp = get_timestamp_iso()

        # Write reports even for dry run
        output_dir = ensure_output_dir(output_root, session_id)
        report_paths = write_all_reports(result, output_dir)
        result.outputs.update({k: str(v) for k, v in report_paths.items()})

        return result

    # Get header info
    header = read_header(kept_paths[0])
    exptime = header.get("EXPTIME", 15.0)

    # Branch based on debayer mode
    if config.debayer == "none":
        # --- MONO PIPELINE (original) ---
        result = _stack_mono_pipeline(
            kept_paths, kept_scores, config, result, header, exptime, progress, quiet,
            save_transforms_flag=save_transforms,
            load_transforms_path=load_transforms_path,
        )
    elif config.debayer == "bayer-first":
        # --- BAYER-FIRST RGB PIPELINE (recommended for OSC) ---
        # Align Bayer frames, stack Bayer, debayer ONCE at end
        result = _stack_bayer_first_pipeline(
            kept_paths, kept_scores, config, result, header, exptime, progress, quiet
        )
    else:
        # --- RGB PIPELINE (legacy - has resolution mismatch issues) ---
        result = _stack_rgb_pipeline(
            kept_paths, kept_scores, config, result, header, exptime, progress, quiet
        )

    # --- Write Reports ---
    output_dir = ensure_output_dir(output_root, session_id)
    report_paths = write_all_reports(result, output_dir)
    result.outputs.update({f"report_{k}": str(v) for k, v in report_paths.items()})

    # Finalize
    elapsed = time.time() - start_time
    result.timestamp = get_timestamp_iso()
    result.stats["processing_time_s"] = elapsed

    # Print final summary
    if not quiet:
        summary_lines = [
            f"Frames stacked: {result.stats.get('n_frames_stacked', 0)}",
            f"Total exposure: {fmt_duration(result.stats.get('total_exposure_s', 0))}",
            f"Processing time: {fmt_duration(elapsed)}",
        ]
        if "snr_proxy" in result.stats:
            summary_lines.append(f"SNR proxy: {result.stats['snr_proxy']:.1f}")

        print_summary_box(summary_lines, title=f"{Symbols.SPARKLE} Complete {Symbols.SPARKLE}")

        if "master_linear" in result.outputs:
            print_path("Output", result.outputs["master_linear"])
        elif "master_rgb" in result.outputs:
            print_path("Output", result.outputs["master_rgb"])

    logger.info("Processing complete in %s", format_duration(elapsed))

    return result


def _stack_mono_pipeline(
    kept_paths: list[Path],
    kept_scores: list,
    config: StackConfig,
    result: StackResult,
    header: dict,
    exptime: float,
    progress: PipelineProgress,
    quiet: bool = False,
    save_transforms_flag: bool = False,
    load_transforms_path: str | Path | None = None,
) -> StackResult:
    """Execute mono (Bayer mosaic) stacking pipeline."""

    # --- Stage 4: Reference Selection ---
    progress.start_stage(4, "Reference Selection", Symbols.STAR)
    ref_path, ref_data = select_reference_frame(
        kept_paths,
        scores=kept_scores,
        method=config.reference,
    )
    result.reference_frame = str(ref_path)
    progress.update_detail(f"Reference: {ref_path.name}")
    progress.update_detail(f"Method: {config.reference}")
    progress.complete_stage()

    # --- Stage 5: Registration ---
    progress.start_stage(5, "Frame Registration", Symbols.WRENCH)

    # Exclude reference from alignment (it's already aligned to itself)
    frames_to_align = [p for p in kept_paths if p != ref_path]

    # Check if using cached transforms
    if load_transforms_path is not None:
        progress.update_detail("Using cached transforms")
        progress.update_detail(f"Loading from: {Path(load_transforms_path).name}")

        cached_transforms, cached_ref, _ = load_transforms(load_transforms_path)
        aligned_frames, align_results = apply_cached_transforms(
            frames_to_align,
            cached_transforms,
            ref_data,
            str(ref_path),
            show_progress=not quiet,
        )
    else:
        # Compute alignment
        progress.update_detail(f"Aligning {len(frames_to_align)} frames to reference")

        # Use parallel alignment by default (unless workers=1)
        if config.workers is None or config.workers > 1:
            import os
            n_workers = config.workers if config.workers else max(1, os.cpu_count() - 1)
            progress.update_detail(f"Using {n_workers} parallel workers")
            aligned_frames, align_results = align_frames_parallel(
                frames_to_align,
                ref_data,
                str(ref_path),
                max_failures=config.max_align_fail,
                show_progress=not quiet,
                workers=config.workers,  # None will auto-detect in the function
            )
        else:
            aligned_frames, align_results = align_frames(
                frames_to_align,
                ref_data,
                str(ref_path),
                max_failures=config.max_align_fail,
                show_progress=not quiet,
            )

    # Save transforms if requested (only if we computed them)
    if save_transforms_flag and load_transforms_path is None:
        output_dir = ensure_output_dir(
            Path(result.session_path).parent.parent / "processedData",
            result.session_id,
        )
        transforms_path = output_dir / "transforms.json"
        all_transforms = [ar.transform for ar in align_results]
        save_transforms(
            all_transforms,
            transforms_path,
            reference_path=str(ref_path),
            metadata={"session_id": result.session_id},
        )
        result.outputs["transforms"] = str(transforms_path)
        progress.update_detail(f"Saved transforms to {transforms_path.name}")

    # Add reference frame to aligned list
    aligned_frames.insert(0, ref_data)

    # Record alignment failures
    n_failures = 0
    for ar in align_results:
        if not ar.success:
            n_failures += 1
            result.alignment_failures.append(ar.path)
            result.rejected.append(
                RejectedFrame(
                    ar.path,
                    RejectionReason.ALIGNMENT_FAILED,
                    ar.transform.error_message,
                )
            )

    # Update kept list (only successfully aligned)
    result.kept = [str(ref_path)] + [ar.path for ar in align_results if ar.success]

    if n_failures > 0:
        progress.warn(f"{n_failures} frames failed alignment")
    progress.complete_stage(f"{len(aligned_frames)} frames aligned")

    if len(aligned_frames) < 2:
        progress.fail_stage("Insufficient frames for stacking (need at least 2)")
        print_error("Insufficient frames for stacking")
        result.timestamp = get_timestamp_iso()
        return result

    # --- Stage 6: Stacking ---
    progress.start_stage(6, "Sigma-Clip Stacking", Symbols.GALAXY)
    progress.update_detail(f"Stacking {len(aligned_frames)} frames")
    progress.update_detail(f"Sigma={config.sigma}, maxiters={config.maxiters}")

    stacked, mask_count = sigma_clip_mean(
        aligned_frames,
        sigma=config.sigma,
        maxiters=config.maxiters,
        use_gpu=config.use_gpu,
    )

    stack_stats = compute_stack_statistics(
        stacked, mask_count, len(aligned_frames), exptime
    )
    result.stats = {
        "n_frames_stacked": stack_stats.n_frames,
        "total_exposure_s": stack_stats.total_exposure_s,
        "mean_clipped_fraction": stack_stats.mean_clipped_fraction,
        "snr_proxy": stack_stats.snr_proxy,
    }

    progress.update_detail(f"Clipped fraction: {stack_stats.mean_clipped_fraction:.1%}")
    progress.update_detail(f"SNR proxy: {stack_stats.snr_proxy:.1f}")
    progress.complete_stage(f"Total exposure: {fmt_duration(stack_stats.total_exposure_s)}")

    # --- Stage 7: Write Outputs ---
    progress.start_stage(7, "Writing Outputs", Symbols.FILE)
    output_dir = ensure_output_dir(Path(result.session_path).parent.parent / "processedData", result.session_id)
    stacked_dir = output_dir / "stacked"

    # Write linear FITS master
    master_fits_path = stacked_dir / "master_linear.fits"
    header["NCOMBINE"] = len(aligned_frames)
    header["EXPTIME"] = stack_stats.total_exposure_s
    header["COMMENT"] = f"Stacked with dwarf3 v{get_version()}"
    header["COMMENT"] = f"Sigma-clipped mean (sigma={config.sigma}, maxiters={config.maxiters})"
    write_fits(master_fits_path, stacked, header=header, overwrite=True)
    result.outputs["master_linear"] = str(master_fits_path)
    progress.update_detail(f"FITS: {master_fits_path.name}")

    # Write quicklook images
    if config.write_quicklook:
        stretched = asinh_stretch(
            stacked,
            a=config.asinh_a,
            percentiles=config.quicklook_percentiles,
        )

        # PNG (8-bit) - grayscale
        png_path = stacked_dir / "master_quicklook.png"
        iio.imwrite(png_path, to_uint8(stretched))
        result.outputs["quicklook_png"] = str(png_path)
        progress.update_detail(f"PNG: {png_path.name}")

        # TIFF (16-bit) - grayscale
        tif_path = stacked_dir / "master_quicklook.tif"
        iio.imwrite(tif_path, to_uint16(stretched))
        result.outputs["quicklook_tif"] = str(tif_path)
        progress.update_detail(f"TIFF: {tif_path.name}")

    progress.complete_stage()

    return result


def _stack_bayer_first_pipeline(
    kept_paths: list[Path],
    kept_scores: list,
    config: StackConfig,
    result: StackResult,
    header: dict,
    exptime: float,
    progress: PipelineProgress,
    quiet: bool = False,
) -> StackResult:
    """
    Execute Bayer-first RGB stacking pipeline.

    This is the recommended approach for OSC (one-shot color) sensors:
    - align_mode='integer': Align using integer-only shifts (preserves RGGB phase)
    - align_mode='rgb_affine': Debayer first, then apply affine transforms to RGB

    Integer mode is preferred for equatorial tracking (minimal rotation).
    RGB affine mode is needed for alt-az tracking with field rotation.
    """
    # Determine alignment mode
    align_mode = config.align_mode
    if align_mode == "auto":
        align_mode = "integer"  # Default for bayer-first: use integer shifts

    # Determine final debayer mode (for output)
    final_debayer = "bilinear"  # Full resolution output by default

    # --- Stage 4: Reference Selection ---
    progress.start_stage(4, "Reference Selection (Bayer)", Symbols.STAR)
    ref_path = Path(kept_scores[0].path) if config.reference == "best" else kept_paths[0]
    result.reference_frame = str(ref_path)

    # Read reference Bayer frame
    ref_bayer = read_fits(ref_path)

    progress.update_detail(f"Reference: {ref_path.name}")
    progress.update_detail(f"Bayer shape: {ref_bayer.shape}")
    progress.update_detail(f"Align mode: {align_mode}")
    progress.complete_stage()

    # Exclude reference from alignment
    frames_to_align = [p for p in kept_paths if p != ref_path]

    if align_mode == "integer":
        # --- Stage 5: Integer-Only Bayer Registration (Color-Preserving) ---
        progress.start_stage(5, "Integer-Only Bayer Alignment", Symbols.WRENCH)
        progress.update_detail(f"Aligning {len(frames_to_align)} frames")
        progress.update_detail("Method: phase correlation, even-integer shifts only")
        progress.update_detail("Color fidelity: PRESERVED (no interpolation)")

        aligned_bayer_frames, shift_info = align_bayer_integer(
            frames_to_align,
            ref_bayer,
            str(ref_path),
            max_failures=config.max_align_fail,
            show_progress=not quiet,
        )

        # Add reference Bayer to aligned list (it's already aligned to itself)
        aligned_bayer_frames.insert(0, ref_bayer)

        # Record alignment info
        n_failures = sum(1 for s in shift_info if not s['success'])
        for s in shift_info:
            if not s['success']:
                result.alignment_failures.append(s['path'])
                result.rejected.append(
                    RejectedFrame(
                        s['path'],
                        RejectionReason.ALIGNMENT_FAILED,
                        s.get('error', 'Alignment failed'),
                    )
                )

        # Update kept list
        result.kept = [str(ref_path)] + [s['path'] for s in shift_info if s['success']]

        if n_failures > 0:
            progress.warn(f"{n_failures} frames failed alignment")
        progress.complete_stage(f"{len(aligned_bayer_frames)} Bayer frames aligned")

    else:  # rgb_affine mode
        # --- Stage 5: Debayer-First RGB Registration ---
        progress.start_stage(5, "Debayer-First RGB Alignment", Symbols.WRENCH)
        progress.update_detail(f"Aligning {len(frames_to_align)} frames")
        progress.update_detail("Method: debayer → affine transform on RGB")
        progress.update_detail("Supports: rotation, scale, translation")

        # Debayer reference for alignment
        ref_rgb = debayer_rggb(ref_bayer, mode="superpixel")

        # Use shared memory parallel alignment for best performance (default)
        if config.workers is None or config.workers > 1:
            import os
            n_workers = config.workers if config.workers else max(1, os.cpu_count() - 1)
            progress.update_detail(f"Using SHM parallel ({n_workers} workers)")
            aligned_rgb_frames, align_results, validity_masks = align_rgb_debayer_first_shm(
                frames_to_align,
                ref_rgb,
                str(ref_path),
                debayer_mode="superpixel",  # Half-res for alignment, faster
                max_failures=config.max_align_fail,
                show_progress=not quiet,
                workers=config.workers,  # None will auto-detect in the function
            )
        else:
            aligned_rgb_frames, align_results, validity_masks = align_rgb_debayer_first(
                frames_to_align,
                ref_rgb,
                str(ref_path),
                debayer_mode="superpixel",  # Half-res for alignment, faster
                max_failures=config.max_align_fail,
                show_progress=not quiet,
                return_masks=True,  # Get validity masks for proper stacking
            )

        # Add reference RGB and its mask (all ones) to aligned lists
        aligned_rgb_frames.insert(0, ref_rgb)
        ref_mask = np.ones((ref_rgb.shape[0], ref_rgb.shape[1]), dtype=np.float32)
        validity_masks.insert(0, ref_mask)

        # Record alignment failures
        n_failures = 0
        for ar in align_results:
            if not ar.success:
                n_failures += 1
                result.alignment_failures.append(ar.path)
                result.rejected.append(
                    RejectedFrame(
                        ar.path,
                        RejectionReason.ALIGNMENT_FAILED,
                        ar.transform.error_message,
                    )
                )

        # Update kept list
        result.kept = [str(ref_path)] + [ar.path for ar in align_results if ar.success]

        if n_failures > 0:
            progress.warn(f"{n_failures} frames failed alignment")
        progress.complete_stage(f"{len(aligned_rgb_frames)} RGB frames aligned")

    # Branch based on alignment mode for stacking
    if align_mode == "integer":
        # INTEGER MODE: Stack Bayer, then debayer once
        if len(aligned_bayer_frames) < 2:
            progress.fail_stage("Insufficient frames for stacking (need at least 2)")
            print_error("Insufficient frames for stacking")
            result.timestamp = get_timestamp_iso()
            return result

        # --- Stage 6: Bayer Stacking ---
        progress.start_stage(6, "Bayer Sigma-Clip Stacking", Symbols.GALAXY)
        progress.update_detail(f"Stacking {len(aligned_bayer_frames)} Bayer frames")
        progress.update_detail(f"Sigma={config.sigma}, maxiters={config.maxiters}")

        stacked_bayer, mask_count = sigma_clip_mean(
            aligned_bayer_frames,
            sigma=config.sigma,
            maxiters=config.maxiters,
            use_gpu=config.use_gpu,
        )

        stack_stats = compute_stack_statistics(
            stacked_bayer, mask_count, len(aligned_bayer_frames), exptime
        )
        n_frames_stacked = len(aligned_bayer_frames)

        progress.update_detail(f"Clipped fraction: {stack_stats.mean_clipped_fraction:.1%}")
        progress.update_detail(f"SNR proxy: {stack_stats.snr_proxy:.1f}")
        progress.complete_stage(f"Total exposure: {fmt_duration(stack_stats.total_exposure_s)}")

        # --- Stage 6b: Final Debayer ---
        progress.start_stage(6, "Final Debayer", Symbols.STAR)
        progress.update_detail(f"Debayering stacked Bayer using {final_debayer} mode")

        stacked_rgb = debayer_rggb(stacked_bayer, mode=final_debayer)

        progress.update_detail(f"RGB output shape: {stacked_rgb.shape}")
        progress.complete_stage("Single debayer complete")

        # Update stats
        result.stats = {
            "n_frames_stacked": stack_stats.n_frames,
            "total_exposure_s": stack_stats.total_exposure_s,
            "mean_clipped_fraction": stack_stats.mean_clipped_fraction,
            "snr_proxy": stack_stats.snr_proxy,
            "align_mode": "integer",
            "debayer_mode": f"bayer-first-{final_debayer}",
            "output_shape": list(stacked_rgb.shape),
        }

        # --- Stage 7: Write Outputs ---
        progress.start_stage(7, "Writing RGB Outputs", Symbols.FILE)
        output_dir = ensure_output_dir(Path(result.session_path).parent.parent / "processedData", result.session_id)
        stacked_dir = output_dir / "stacked"

        # Write linear Bayer FITS (intermediate, for debugging)
        bayer_fits_path = stacked_dir / "master_bayer.fits"
        header_bayer = header.copy()
        header_bayer["NCOMBINE"] = n_frames_stacked
        header_bayer["EXPTIME"] = stack_stats.total_exposure_s
        header_bayer["COMMENT"] = f"Stacked with dwarf3 v{get_version()}"
        header_bayer["COMMENT"] = "Integer-only alignment: RGGB phase preserved"
        write_fits(bayer_fits_path, stacked_bayer, header=header_bayer, overwrite=True)
        result.outputs["master_bayer"] = str(bayer_fits_path)
        progress.update_detail(f"FITS (Bayer): {bayer_fits_path.name}")

    else:
        # RGB_AFFINE MODE: Stack already-debayered RGB frames
        if len(aligned_rgb_frames) < 2:
            progress.fail_stage("Insufficient frames for stacking (need at least 2)")
            print_error("Insufficient frames for stacking")
            result.timestamp = get_timestamp_iso()
            return result

        # --- Stage 6: RGB Stacking ---
        if config.full_res:
            # FULL-RES MODE: Plane-based stacking (Option B1 from TODO2.md)
            # Warp Bayer planes separately, then demosaic once at the end
            progress.start_stage(6, "Full-Res Plane-Based Stack", Symbols.GALAXY)
            progress.update_detail("Warp planes → stack → demosaic once")
            progress.update_detail("Lanczos4 interpolation (sharper)")
            progress.update_detail("No double interpolation artifacts")

            # Extract successful transforms (skip failures)
            successful_paths = []
            successful_transforms = []
            for ar in align_results:
                if ar.success and ar.transform.matrix is not None:
                    successful_paths.append(Path(ar.path))
                    successful_transforms.append(ar.transform)

            # Free 2K aligned frames memory before 4K processing
            del aligned_rgb_frames
            del validity_masks

            stacked_rgb, coverage = stream_plane_stack(
                successful_paths,
                successful_transforms,
                str(ref_path),
                sigma=config.sigma,
                show_progress=not quiet,
            )
            n_frames_stacked = len(successful_paths) + 1  # +1 for reference

            # Convert coverage to mask_count format for stats
            mask_count = (coverage > 0).astype(np.float32) * n_frames_stacked
            debayer_mode_str = "plane-stack-lanczos4"
        else:
            # STANDARD 2K MODE: Stack pre-aligned frames
            progress.start_stage(6, "RGB Sigma-Clip Stacking (Streaming)", Symbols.GALAXY)
            progress.update_detail(f"Stacking {len(aligned_rgb_frames)} RGB frames")
            progress.update_detail(f"Sigma={config.sigma} (two-pass streaming)")
            progress.update_detail("Memory-efficient mask-aware normalization")

            stacked_rgb, coverage = sigma_clip_mask_aware_rgb_streaming(
                aligned_rgb_frames,
                validity_masks,
                sigma=config.sigma,
            )
            n_frames_stacked = len(aligned_rgb_frames)

            # Convert coverage to mask_count format for stats
            mask_count = (coverage > 0).astype(np.float32) * n_frames_stacked
            debayer_mode_str = "debayer-first-superpixel"

        stack_stats = compute_stack_statistics_rgb(
            stacked_rgb, mask_count, n_frames_stacked, exptime
        )

        progress.update_detail(f"Clipped fraction: {stack_stats.mean_clipped_fraction:.1%}")
        progress.update_detail(f"SNR proxy: {stack_stats.snr_proxy:.1f}")
        progress.complete_stage(f"Total exposure: {fmt_duration(stack_stats.total_exposure_s)}")

        # Update stats
        result.stats = {
            "n_frames_stacked": stack_stats.n_frames,
            "total_exposure_s": stack_stats.total_exposure_s,
            "mean_clipped_fraction": stack_stats.mean_clipped_fraction,
            "snr_proxy": stack_stats.snr_proxy,
            "align_mode": "rgb_affine",
            "debayer_mode": debayer_mode_str,
            "output_shape": list(stacked_rgb.shape),
            "full_res": config.full_res,
        }

        # --- Stage 7: Write Outputs ---
        progress.start_stage(7, "Writing RGB Outputs", Symbols.FILE)
        output_dir = ensure_output_dir(Path(result.session_path).parent.parent / "processedData", result.session_id)
        stacked_dir = output_dir / "stacked"

    # Common output writing for both modes
    # Write linear RGB FITS master
    master_fits_path = stacked_dir / "master_rgb.fits"
    header["NCOMBINE"] = n_frames_stacked
    header["EXPTIME"] = stack_stats.total_exposure_s
    header["NAXIS"] = 3
    header["NAXIS3"] = 3
    header["COMMENT"] = f"Stacked with dwarf3 v{get_version()}"
    header["COMMENT"] = f"Align mode: {align_mode}"
    header["COMMENT"] = f"Sigma-clipped mean (sigma={config.sigma}, maxiters={config.maxiters})"

    # For RGB FITS, reorder to (3, H, W) for FITS convention
    stacked_fits = np.transpose(stacked_rgb, (2, 0, 1))
    write_fits(master_fits_path, stacked_fits, header=header, overwrite=True)
    result.outputs["master_rgb"] = str(master_fits_path)
    progress.update_detail(f"FITS (RGB): {master_fits_path.name}")

    # Write quicklook images
    if config.write_quicklook:
        # Stretch each channel independently for visualization
        stretched_rgb = np.zeros_like(stacked_rgb)
        for c in range(3):
            stretched_rgb[:, :, c] = asinh_stretch(
                stacked_rgb[:, :, c],
                a=config.asinh_a,
                percentiles=config.quicklook_percentiles,
            )

        # PNG (8-bit RGB)
        png_path = stacked_dir / "master_quicklook_rgb.png"
        iio.imwrite(png_path, to_uint8(stretched_rgb))
        result.outputs["quicklook_png"] = str(png_path)
        progress.update_detail(f"PNG: {png_path.name}")

        # TIFF (16-bit RGB)
        tif_path = stacked_dir / "master_quicklook_rgb.tif"
        iio.imwrite(tif_path, to_uint16(stretched_rgb))
        result.outputs["quicklook_tif"] = str(tif_path)
        progress.update_detail(f"TIFF: {tif_path.name}")

    progress.complete_stage()

    return result


def _stack_rgb_pipeline(
    kept_paths: list[Path],
    kept_scores: list,
    config: StackConfig,
    result: StackResult,
    header: dict,
    exptime: float,
    progress: PipelineProgress,
    quiet: bool = False,
) -> StackResult:
    """Execute RGB (debayered) stacking pipeline."""

    # Determine debayer mode
    debayer_mode = "superpixel" if config.debayer == "superpixel" else "bilinear"
    progress.update_detail(f"Using {debayer_mode} debayer mode")

    # Create debayer and luminance functions
    debayer_func = partial(debayer_rggb, mode=debayer_mode)
    luma_func = partial(luminance_from_rgb, method="weighted")

    # --- Stage 4: Reference Selection ---
    progress.start_stage(4, "Reference Selection + Debayer", Symbols.STAR)
    ref_path = Path(kept_scores[0].path) if config.reference == "best" else kept_paths[0]
    result.reference_frame = str(ref_path)

    # Read, debayer, and extract luminance from reference
    ref_raw = read_fits(ref_path)
    ref_rgb = debayer_func(ref_raw)
    ref_luma = luma_func(ref_rgb)

    progress.update_detail(f"Reference: {ref_path.name}")
    progress.update_detail(f"RGB shape: {ref_rgb.shape}")
    progress.complete_stage()

    # --- Stage 5: RGB Registration ---
    progress.start_stage(5, "RGB Frame Registration", Symbols.WRENCH)
    progress.update_detail(f"Aligning {len(kept_paths) - 1} frames to reference")

    # Exclude reference from alignment
    frames_to_align = [p for p in kept_paths if p != ref_path]

    aligned_rgb_frames, align_results = align_rgb_frames(
        frames_to_align,
        ref_luma,
        str(ref_path),
        debayer_func=debayer_func,
        luma_func=luma_func,
        max_failures=config.max_align_fail,
        show_progress=not quiet,
    )

    # Add reference RGB to aligned list
    aligned_rgb_frames.insert(0, ref_rgb)

    # Record alignment failures
    n_failures = 0
    for ar in align_results:
        if not ar.success:
            n_failures += 1
            result.alignment_failures.append(ar.path)
            result.rejected.append(
                RejectedFrame(
                    ar.path,
                    RejectionReason.ALIGNMENT_FAILED,
                    ar.transform.error_message,
                )
            )

    # Update kept list
    result.kept = [str(ref_path)] + [ar.path for ar in align_results if ar.success]

    if n_failures > 0:
        progress.warn(f"{n_failures} frames failed alignment")
    progress.complete_stage(f"{len(aligned_rgb_frames)} RGB frames aligned")

    if len(aligned_rgb_frames) < 2:
        progress.fail_stage("Insufficient frames for stacking (need at least 2)")
        print_error("Insufficient frames for stacking")
        result.timestamp = get_timestamp_iso()
        return result

    # --- Stage 6: RGB Stacking ---
    progress.start_stage(6, "RGB Sigma-Clip Stacking", Symbols.GALAXY)
    progress.update_detail(f"Stacking {len(aligned_rgb_frames)} RGB frames")
    progress.update_detail(f"Sigma={config.sigma}, maxiters={config.maxiters}")

    stacked_rgb, mask_count = sigma_clip_mean_rgb(
        aligned_rgb_frames,
        sigma=config.sigma,
        maxiters=config.maxiters,
        use_gpu=config.use_gpu,
    )

    stack_stats = compute_stack_statistics_rgb(
        stacked_rgb, mask_count, len(aligned_rgb_frames), exptime
    )
    result.stats = {
        "n_frames_stacked": stack_stats.n_frames,
        "total_exposure_s": stack_stats.total_exposure_s,
        "mean_clipped_fraction": stack_stats.mean_clipped_fraction,
        "snr_proxy": stack_stats.snr_proxy,
        "debayer_mode": debayer_mode,
        "output_shape": list(stacked_rgb.shape),
    }

    progress.update_detail(f"Clipped fraction: {stack_stats.mean_clipped_fraction:.1%}")
    progress.update_detail(f"SNR proxy: {stack_stats.snr_proxy:.1f}")
    progress.complete_stage(f"Total exposure: {fmt_duration(stack_stats.total_exposure_s)}")

    # --- Stage 7: Write Outputs ---
    progress.start_stage(7, "Writing RGB Outputs", Symbols.FILE)
    output_dir = ensure_output_dir(Path(result.session_path).parent.parent / "processedData", result.session_id)
    stacked_dir = output_dir / "stacked"

    # Write linear RGB FITS master
    master_fits_path = stacked_dir / "master_rgb.fits"
    header["NCOMBINE"] = len(aligned_rgb_frames)
    header["EXPTIME"] = stack_stats.total_exposure_s
    header["NAXIS"] = 3
    header["NAXIS3"] = 3
    header["COMMENT"] = f"Stacked with dwarf3 v{get_version()}"
    header["COMMENT"] = f"Debayer mode: {debayer_mode}"
    header["COMMENT"] = f"Sigma-clipped mean (sigma={config.sigma}, maxiters={config.maxiters})"

    # For RGB FITS, we need to reorder to (3, H, W) for FITS convention
    stacked_fits = np.transpose(stacked_rgb, (2, 0, 1))
    write_fits(master_fits_path, stacked_fits, header=header, overwrite=True)
    result.outputs["master_rgb"] = str(master_fits_path)
    progress.update_detail(f"FITS: {master_fits_path.name}")

    # Write quicklook images
    if config.write_quicklook:
        # Stretch each channel independently for better visualization
        stretched_rgb = np.zeros_like(stacked_rgb)
        for c in range(3):
            stretched_rgb[:, :, c] = asinh_stretch(
                stacked_rgb[:, :, c],
                a=config.asinh_a,
                percentiles=config.quicklook_percentiles,
            )

        # PNG (8-bit RGB)
        png_path = stacked_dir / "master_quicklook_rgb.png"
        iio.imwrite(png_path, to_uint8(stretched_rgb))
        result.outputs["quicklook_png"] = str(png_path)
        progress.update_detail(f"PNG: {png_path.name}")

        # TIFF (16-bit RGB)
        tif_path = stacked_dir / "master_quicklook_rgb.tif"
        iio.imwrite(tif_path, to_uint16(stretched_rgb))
        result.outputs["quicklook_tif"] = str(tif_path)
        progress.update_detail(f"TIFF: {tif_path.name}")

    progress.complete_stage()

    return result


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="dwarf3",
        description="Reproducible stacking pipeline for DWARF 3 telescope FITS data",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"dwarf3 {get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Stack command
    stack_parser = subparsers.add_parser(
        "stack",
        help="Stack frames from a session",
    )
    stack_parser.add_argument(
        "session",
        type=str,
        help="Path to the raw session folder",
    )
    stack_parser.add_argument(
        "--out",
        type=str,
        default="processedData",
        help="Output root directory (default: processedData)",
    )
    stack_parser.add_argument(
        "--keep",
        type=float,
        default=0.92,
        help="Fraction of frames to keep (default: 0.92)",
    )
    stack_parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Sigma for sigma-clipped mean (default: 3.0)",
    )
    stack_parser.add_argument(
        "--maxiters",
        type=int,
        default=5,
        help="Max iterations for sigma clipping (default: 5)",
    )
    stack_parser.add_argument(
        "--reference",
        type=str,
        choices=["best", "first"],
        default="best",
        help="Reference frame selection method (default: best)",
    )
    stack_parser.add_argument(
        "--debayer",
        type=str,
        choices=["none", "rgb", "superpixel", "bayer-first"],
        default="none",
        help="Debayer mode: none (mono), bayer-first (recommended for RGB), rgb, superpixel",
    )
    stack_parser.add_argument(
        "--align-mode",
        type=str,
        choices=["integer", "rgb_affine", "auto"],
        default="auto",
        help="Alignment mode: integer (phase-preserving shifts, best for EQ tracking), "
             "rgb_affine (debayer first, full affine, for alt-az), auto (default)",
    )
    stack_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for scoring (default: auto = CPU count - 1)",
    )
    stack_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score and report only, no stacking",
    )
    stack_parser.add_argument(
        "--no-quicklook",
        action="store_true",
        help="Skip quicklook PNG/TIFF generation",
    )
    stack_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    stack_parser.add_argument(
        "--save-transforms",
        action="store_true",
        help="Save alignment transforms to transforms.json for reuse",
    )
    stack_parser.add_argument(
        "--load-transforms",
        type=str,
        default=None,
        metavar="FILE",
        help="Load cached transforms from JSON file (skip alignment computation)",
    )
    stack_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress colored output (use logging only)",
    )
    stack_parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration (CuPy) for stacking if available",
    )
    stack_parser.add_argument(
        "--full-res",
        action="store_true",
        help="Output full 4K resolution (streaming mode). Aligns at 2K, stacks at 4K. "
             "Slower but preserves full sensor resolution. Only for bayer-first mode.",
    )
    stack_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (compute everything fresh, don't save cache)",
    )
    stack_parser.add_argument(
        "--cache-refresh",
        action="store_true",
        help="Clear existing cache before starting (force recomputation)",
    )
    stack_parser.add_argument(
        "--frame-list",
        type=str,
        default=None,
        metavar="FILE",
        help="File containing frame paths/names to include (one per line)",
    )

    # Cache management command
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage pipeline cache (status, clear)",
    )
    cache_parser.add_argument(
        "session",
        type=str,
        help="Path to the raw session folder or processed output folder",
    )
    cache_parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status (default if no action specified)",
    )
    cache_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all cached artifacts",
    )
    cache_parser.add_argument(
        "--clear-transforms",
        action="store_true",
        help="Clear only alignment transforms cache",
    )
    cache_parser.add_argument(
        "--clear-stack",
        action="store_true",
        help="Clear only stacked image cache",
    )
    cache_parser.add_argument(
        "--out",
        type=str,
        default="processedData",
        help="Output root directory (default: processedData)",
    )

    # Frames listing command
    frames_parser = subparsers.add_parser(
        "frames",
        help="List and manage frames in a session",
    )
    frames_parser.add_argument(
        "session",
        type=str,
        help="Path to the raw session folder",
    )
    frames_parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="FILE",
        help="Save frame list to file (for use with --frame-list)",
    )
    frames_parser.add_argument(
        "--scored",
        action="store_true",
        help="Show quality scores if available in cache",
    )
    frames_parser.add_argument(
        "--keep",
        type=float,
        default=None,
        help="Show which frames would be kept at this fraction",
    )
    frames_parser.add_argument(
        "--out",
        type=str,
        default="processedData",
        help="Output root directory for cache lookup (default: processedData)",
    )

    # Process command (post-stack luminance processing)
    process_parser = subparsers.add_parser(
        "process",
        help="Apply post-stack luminance processing to a FITS master",
    )
    process_parser.add_argument(
        "input",
        type=str,
        nargs="?",  # Makes input optional when --list-colormaps is used
        default=None,
        help="Path to the stacked FITS master",
    )
    process_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: <input>_processed.png)",
    )
    process_parser.add_argument(
        "--no-bg-subtract",
        action="store_true",
        help="Skip background subtraction",
    )
    process_parser.add_argument(
        "--bg-cell-size",
        type=int,
        default=128,
        help="Cell size for background estimation (default: 128)",
    )
    process_parser.add_argument(
        "--stretch",
        type=str,
        choices=["single", "two_stage"],
        default="two_stage",
        help="Stretch mode (default: two_stage)",
    )
    process_parser.add_argument(
        "--no-contrast",
        action="store_true",
        help="Skip local contrast enhancement",
    )
    process_parser.add_argument(
        "--contrast-scale",
        type=int,
        default=100,
        help="Scale for contrast enhancement (default: 100)",
    )
    process_parser.add_argument(
        "--contrast-strength",
        type=float,
        default=0.3,
        help="Strength of contrast enhancement (default: 0.3)",
    )
    process_parser.add_argument(
        "--reduce-noise",
        action="store_true",
        help="Apply masked noise reduction to background only",
    )
    process_parser.add_argument(
        "--noise-method",
        type=str,
        choices=["median", "gaussian", "bilateral"],
        default="median",
        help="Noise reduction filter type (default: median)",
    )
    process_parser.add_argument(
        "--noise-size",
        type=int,
        default=3,
        help="Noise filter kernel size (default: 3)",
    )
    process_parser.add_argument(
        "--colormap",
        type=str,
        default=None,
        help="Apply pseudo-color palette (e.g., viridis, inferno, cool, warm, h_alpha)",
    )
    process_parser.add_argument(
        "--list-colormaps",
        action="store_true",
        help="List available colormap palettes and exit",
    )
    process_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Color command (LRGB color processing)
    color_parser = subparsers.add_parser(
        "color",
        help="Apply proper RGB calibration and LRGB combination",
    )
    color_parser.add_argument(
        "input",
        type=str,
        help="Path to linear RGB FITS (from stack --debayer rgb) or session folder",
    )
    color_parser.add_argument(
        "--luminance",
        type=str,
        default=None,
        help="Path to processed luminance image (PNG/TIFF/FITS) for LRGB combination",
    )
    color_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: same as input or processedData/<session>)",
    )
    color_parser.add_argument(
        "--wb-method",
        type=str,
        choices=["stars", "gray_world", "none"],
        default="stars",
        help="White balance method (default: stars)",
    )
    color_parser.add_argument(
        "--scnr",
        type=float,
        default=0.0,
        metavar="STRENGTH",
        help="SCNR green reduction strength 0-1 (default: 0 = disabled)",
    )
    color_parser.add_argument(
        "--bg",
        type=str,
        choices=["none", "mild", "strong"],
        default="none",
        help="Background subtraction mode (default: none - safest for galaxies)",
    )
    color_parser.add_argument(
        "--chroma-boost",
        type=float,
        default=1.2,
        help="Chrominance/saturation boost factor (default: 1.2)",
    )
    color_parser.add_argument(
        "--stretch",
        type=float,
        default=0.1,
        help="Asinh stretch parameter for RGB (default: 0.1)",
    )
    color_parser.add_argument(
        "--lrgb-method",
        type=str,
        choices=["lab", "hsv", "direct"],
        default="lab",
        help="LRGB combination method (default: lab)",
    )
    color_parser.add_argument(
        "--no-stretch",
        action="store_true",
        help="Skip stretching (output linear calibrated RGB only)",
    )
    color_parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "galaxy"],
        default="default",
        help="Processing mode: 'galaxy' disables BG subtraction in calibration (safest for extended objects)",
    )
    color_parser.add_argument(
        "--rgb-gains",
        type=str,
        default=None,
        metavar="R,G,B",
        help="Manual RGB gains, e.g. '1.2,0.7,0.9' (overrides WB computation)",
    )
    color_parser.add_argument(
        "--saturation",
        type=float,
        default=1.0,
        help="Saturation boost factor (default: 1.0)",
    )
    color_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def main() -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "stack":
        setup_logging(args.verbose)

        config = StackConfig(
            keep_fraction=args.keep,
            sigma=args.sigma,
            maxiters=args.maxiters,
            reference=args.reference,
            debayer=args.debayer,
            align_mode=args.align_mode,
            workers=args.workers,
            write_quicklook=not args.no_quicklook,
            use_gpu=args.use_gpu,
            full_res=args.full_res,
        )

        try:
            result = stack_session(
                args.session,
                output_root=args.out,
                config=config,
                dry_run=args.dry_run,
                quiet=args.quiet,
                save_transforms=args.save_transforms,
                load_transforms_path=args.load_transforms,
                no_cache=args.no_cache,
                cache_refresh=args.cache_refresh,
                frame_list_path=args.frame_list,
            )

            if len(result.kept) == 0:
                print_error("No frames were successfully processed")
                return 1

            return 0

        except Exception as e:
            print_error(f"Stacking failed: {e}")
            logger.exception("Stacking failed: %s", e)
            return 1

    elif args.command == "process":
        setup_logging(args.verbose)

        from .processing import process_luminance
        from .colormap import apply_colormap, list_palettes

        # Handle --list-colormaps
        if args.list_colormaps:
            print_header("Available Colormap Palettes")
            palettes = list_palettes()
            print_info("Custom astronomy palettes:")
            for p in ["grayscale", "cool", "warm", "blue_gold", "h_alpha", "oiii"]:
                if p in palettes:
                    print(f"  {Colors.VALUE}{p}{Colors.RESET}")
            print_info("Standard matplotlib palettes:")
            for p in palettes:
                if p not in ["grayscale", "cool", "warm", "blue_gold", "h_alpha", "oiii"]:
                    print(f"  {Colors.VALUE}{p}{Colors.RESET}")
            return 0

        # Setup terminal
        setup_terminal()

        # Check input is provided (required unless --list-colormaps)
        if args.input is None:
            print_error("Input file is required. Use 'dwarf3 process <input.fits>'")
            return 1

        input_path = Path(args.input)
        if not input_path.exists():
            print_error(f"Input file not found: {input_path}")
            return 1

        # Determine output path
        if args.out:
            output_path = Path(args.out)
        else:
            output_path = input_path.with_name(
                input_path.stem + "_processed.png"
            )

        print_header(f"Post-Stack Processing: {input_path.name}")
        print_metric("Background subtraction", "enabled" if not args.no_bg_subtract else "disabled")
        print_metric("Noise reduction", f"{args.noise_method} (size={args.noise_size})" if args.reduce_noise else "disabled")
        print_metric("Stretch mode", args.stretch)
        print_metric("Contrast enhancement", "enabled" if not args.no_contrast else "disabled")
        print_metric("Colormap", args.colormap if args.colormap else "grayscale")

        try:
            # Load FITS
            progress = PipelineProgress(total_stages=3, quiet=False)
            progress.start_stage(1, "Loading FITS", Symbols.FILE)
            data = read_fits(input_path)
            progress.update_detail(f"Shape: {data.shape}")
            progress.update_detail(f"Dtype: {data.dtype}")
            progress.complete_stage()

            # Apply processing
            progress.start_stage(2, "Luminance Processing", Symbols.WRENCH)
            progress.update_detail(f"Cell size: {args.bg_cell_size}")
            if args.reduce_noise:
                progress.update_detail(f"Noise reduction: {args.noise_method} (size={args.noise_size})")
            if not args.no_contrast:
                progress.update_detail(f"Contrast scale: {args.contrast_scale}")
                progress.update_detail(f"Contrast strength: {args.contrast_strength}")

            processed = process_luminance(
                data,
                subtract_bg=not args.no_bg_subtract,
                bg_cell_size=args.bg_cell_size,
                stretch_mode=args.stretch,
                enhance_contrast=not args.no_contrast,
                contrast_scale=args.contrast_scale,
                contrast_strength=args.contrast_strength,
                protect_stars=True,
                reduce_noise=args.reduce_noise,
                noise_method=args.noise_method,
                noise_filter_size=args.noise_size,
            )
            progress.complete_stage()

            # Apply colormap if requested
            if args.colormap:
                progress.update_detail(f"Applying colormap: {args.colormap}")
                processed = apply_colormap(
                    processed,
                    palette=args.colormap,
                    percentile_clip=(0.5, 99.5),
                )

            # Save output
            progress.start_stage(3, "Writing Outputs", Symbols.FILE)

            # Convert to output format
            if args.colormap:
                # RGB output (colormap applied)
                output_uint8 = (np.clip(processed, 0, 1) * 255).astype(np.uint8)
                iio.imwrite(output_path, output_uint8)
                progress.update_detail(f"PNG (color): {output_path.name}")

                # Also save TIFF
                if output_path.suffix.lower() == ".png":
                    tiff_path = output_path.with_suffix(".tif")
                    output_uint16 = (np.clip(processed, 0, 1) * 65535).astype(np.uint16)
                    iio.imwrite(tiff_path, output_uint16)
                    progress.update_detail(f"TIFF (color): {tiff_path.name}")
            else:
                # Grayscale output
                output_uint8 = to_uint8(processed)
                iio.imwrite(output_path, output_uint8)
                progress.update_detail(f"PNG: {output_path.name}")

                # Also save TIFF if PNG output
                if output_path.suffix.lower() == ".png":
                    tiff_path = output_path.with_suffix(".tif")
                    output_uint16 = to_uint16(processed)
                    iio.imwrite(tiff_path, output_uint16)
                    progress.update_detail(f"TIFF: {tiff_path.name}")

            progress.complete_stage()

            print_summary_box(
                [f"Output: {output_path}"],
                title=f"{Symbols.SPARKLE} Processing Complete {Symbols.SPARKLE}",
            )

            return 0

        except Exception as e:
            print_error(f"Processing failed: {e}")
            logger.exception("Processing failed: %s", e)
            return 1

    elif args.command == "color":
        setup_logging(args.verbose)

        from .color import (
            calibrate_rgb,
            combine_lrgb,
            create_extended_object_mask,
            scnr_green,
            stretch_rgb_linked,
        )
        from .processing import create_galaxy_mask, subtract_background

        # Setup terminal
        setup_terminal()

        input_path = Path(args.input)
        if not input_path.exists():
            print_error(f"Input not found: {input_path}")
            return 1

        # Determine if input is FITS or session folder
        if input_path.is_dir():
            # Session folder - look for RGB stack in cache or stacked
            rgb_candidates = [
                input_path / "stacked" / "master_rgb.fits",
                input_path / "cache" / "stack_rgb_linear.fits",
            ]
            rgb_path = None
            for candidate in rgb_candidates:
                if candidate.exists():
                    rgb_path = candidate
                    break
            if rgb_path is None:
                print_error(f"No RGB stack found in session: {input_path}")
                print_info("Run 'dwarf3 stack <session> --debayer rgb' first")
                return 1
            output_dir = input_path / "stacked"
        else:
            # Direct FITS file
            rgb_path = input_path
            output_dir = Path(args.out) if args.out else rgb_path.parent

        output_dir.mkdir(parents=True, exist_ok=True)

        print_header(f"Color Processing: {rgb_path.name}")
        print_metric("Mode", args.mode.upper())
        print_metric("Background (pre-calib)", args.bg)
        print_metric("White balance", args.wb_method if not args.rgb_gains else f"manual ({args.rgb_gains})")
        print_metric("SCNR green", f"{args.scnr:.2f}" if args.scnr > 0 else "disabled")
        print_metric("Chroma boost", f"{args.chroma_boost:.1f}x")
        print_metric("Saturation", f"{args.saturation:.1f}x" if args.saturation != 1.0 else "default")
        print_metric("LRGB method", args.lrgb_method if args.luminance else "N/A (no luminance)")

        try:
            # Adjust stage count based on bg mode
            total_stages = 6 if args.bg != "none" else 5
            progress = PipelineProgress(total_stages=total_stages, quiet=False)
            stage_num = 0

            # Stage 1: Load RGB
            stage_num += 1
            progress.start_stage(stage_num, "Loading RGB Stack", Symbols.FILE)
            rgb_data = read_fits(rgb_path)

            # Handle different FITS layouts
            if rgb_data.ndim == 3:
                if rgb_data.shape[0] == 3:
                    # (3, H, W) -> (H, W, 3)
                    rgb_data = np.transpose(rgb_data, (1, 2, 0))
                elif rgb_data.shape[2] != 3:
                    print_error(f"Unexpected RGB shape: {rgb_data.shape}")
                    return 1
            else:
                print_error(f"Expected 3D RGB array, got shape: {rgb_data.shape}")
                return 1

            progress.update_detail(f"Shape: {rgb_data.shape}")
            progress.update_detail(f"Range: [{rgb_data.min():.1f}, {rgb_data.max():.1f}]")
            progress.complete_stage()

            # Stage 2: Create galaxy mask for background protection
            stage_num += 1
            progress.start_stage(stage_num, "Creating Galaxy Mask", Symbols.STAR)

            # Use luminance channel for galaxy detection
            luminance_for_mask = 0.299 * rgb_data[:, :, 0] + 0.587 * rgb_data[:, :, 1] + 0.114 * rgb_data[:, :, 2]
            galaxy_mask = create_galaxy_mask(
                luminance_for_mask,
                threshold_sigma=1.5,
                smoothing_scales=(50, 100, 200),
                dilation_radius=50,
            )
            mask_pct = 100 * np.mean(galaxy_mask)
            progress.update_detail(f"Galaxy protection: {mask_pct:.1f}% of image")
            progress.complete_stage()

            # Stage 3 (optional): Background subtraction
            if args.bg != "none":
                stage_num += 1
                progress.start_stage(stage_num, "Background Subtraction", Symbols.WRENCH)

                # Configure based on mode
                if args.bg == "mild":
                    # Polynomial plane only (tilt correction)
                    bg_params = {"cell_size": 512, "bg_model": "polynomial", "poly_order": 1}
                    progress.update_detail("Mode: mild (plane correction)")
                else:  # strong
                    # Quadratic polynomial (vignetting correction)
                    bg_params = {"cell_size": 256, "bg_model": "polynomial", "poly_order": 2}
                    progress.update_detail("Mode: strong (quadratic correction)")

                # Apply background subtraction per channel with galaxy mask protection
                for i, ch_name in enumerate(["R", "G", "B"]):
                    rgb_data[:, :, i] = subtract_background(
                        rgb_data[:, :, i],
                        object_mask=galaxy_mask,
                        mode="subtract",
                        **bg_params,
                    )
                    progress.update_detail(f"  {ch_name} channel corrected")

                progress.complete_stage()

            # Create object mask for calibration (stars + extended)
            object_mask = create_extended_object_mask(rgb_data)

            # Parse manual RGB gains if provided
            rgb_gains = None
            if args.rgb_gains:
                try:
                    parts = [float(x.strip()) for x in args.rgb_gains.split(",")]
                    if len(parts) != 3:
                        raise ValueError("Expected 3 values")
                    rgb_gains = tuple(parts)
                except ValueError as e:
                    print_error(f"Invalid --rgb-gains format: {args.rgb_gains} ({e})")
                    return 1

            # Determine background mode for calibration
            # Galaxy mode uses "none" to preserve faint halo signal
            if args.mode == "galaxy":
                bg_mode = "none"
            else:
                bg_mode = "per_channel"  # Default for point sources / star fields

            # Next stage: Calibrate RGB
            stage_num += 1
            progress.start_stage(stage_num, "Calibrating Colors", Symbols.WRENCH)
            if args.mode == "galaxy":
                progress.update_detail("Mode: galaxy (no BG subtraction in calibration)")
            calibrated_rgb, calibration = calibrate_rgb(
                rgb_data,
                object_mask=object_mask,
                wb_method=args.wb_method,
                bg_mode=bg_mode,
                rgb_gains=rgb_gains,
            )
            if rgb_gains:
                progress.update_detail(f"Manual gains: R={rgb_gains[0]:.3f}, G={rgb_gains[1]:.3f}, B={rgb_gains[2]:.3f}")
            else:
                progress.update_detail(f"BG offset: R={calibration.bg_offset_r:.1f}, G={calibration.bg_offset_g:.1f}, B={calibration.bg_offset_b:.1f}")
                progress.update_detail(f"WB gains: R={calibration.wb_gain_r:.3f}, G={calibration.wb_gain_g:.3f}, B={calibration.wb_gain_b:.3f}")

            # Apply SCNR if requested
            if args.scnr > 0:
                progress.update_detail(f"Applying SCNR (strength={args.scnr:.2f})")
                calibrated_rgb = scnr_green(calibrated_rgb, strength=args.scnr)

            # Apply saturation boost if requested
            if args.saturation != 1.0:
                from skimage import color as skcolor
                # Convert to HSV, boost saturation, convert back
                # Normalize to 0-1 first
                rgb_norm = calibrated_rgb / max(calibrated_rgb.max(), 1e-6)
                rgb_norm = np.clip(rgb_norm, 0, 1)
                hsv = skcolor.rgb2hsv(rgb_norm)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * args.saturation, 0, 1)
                calibrated_rgb = skcolor.hsv2rgb(hsv) * calibrated_rgb.max()
                progress.update_detail(f"Saturation boost: {args.saturation:.2f}x")

            progress.complete_stage()

            # Next stage: Stretch or LRGB combination
            stage_num += 1
            if args.luminance:
                progress.start_stage(stage_num, "LRGB Combination", Symbols.SPARKLE)

                # Load luminance
                lum_path = Path(args.luminance)
                if not lum_path.exists():
                    print_error(f"Luminance file not found: {lum_path}")
                    return 1

                if lum_path.suffix.lower() in [".fits", ".fit"]:
                    luminance = read_fits(lum_path)
                else:
                    # PNG/TIFF
                    luminance = iio.imread(lum_path)
                    if luminance.ndim == 3:
                        luminance = luminance[:, :, 0]  # Take first channel
                    luminance = luminance.astype(np.float32) / luminance.max()

                progress.update_detail(f"Luminance shape: {luminance.shape}")

                # Stretch RGB mildly for chrominance
                if not args.no_stretch:
                    stretched_rgb = stretch_rgb_linked(
                        calibrated_rgb,
                        asinh_stretch=args.stretch * 2,  # Gentler stretch for chroma
                    )
                else:
                    stretched_rgb = calibrated_rgb / calibrated_rgb.max()

                # Combine LRGB
                lrgb_result = combine_lrgb(
                    luminance,
                    stretched_rgb,
                    method=args.lrgb_method,
                    chroma_boost=args.chroma_boost,
                )
                final_rgb = lrgb_result.lrgb
                progress.update_detail(f"Method: {args.lrgb_method}")
                progress.complete_stage()

            else:
                progress.start_stage(stage_num, "Stretching RGB", Symbols.SPARKLE)

                if not args.no_stretch:
                    final_rgb = stretch_rgb_linked(
                        calibrated_rgb,
                        asinh_stretch=args.stretch,
                    )
                    progress.update_detail(f"Asinh stretch: {args.stretch}")
                else:
                    final_rgb = calibrated_rgb / max(calibrated_rgb.max(), 1e-6)
                    progress.update_detail("No stretch (linear output)")

                progress.complete_stage()

            # Final stage: Save outputs
            stage_num += 1
            progress.start_stage(stage_num, "Writing Outputs", Symbols.FILE)

            # Save calibrated linear RGB FITS
            calib_fits_path = output_dir / "master_rgb_calibrated.fits"
            write_fits(calib_fits_path, calibrated_rgb, overwrite=True)
            progress.update_detail(f"FITS (linear): {calib_fits_path.name}")

            # Save final RGB PNG
            output_name = "master_lrgb" if args.luminance else "master_rgb_color"
            png_path = output_dir / f"{output_name}.png"
            output_uint8 = (np.clip(final_rgb, 0, 1) * 255).astype(np.uint8)
            iio.imwrite(png_path, output_uint8)
            progress.update_detail(f"PNG: {png_path.name}")

            # Save TIFF
            tiff_path = output_dir / f"{output_name}.tif"
            output_uint16 = (np.clip(final_rgb, 0, 1) * 65535).astype(np.uint16)
            iio.imwrite(tiff_path, output_uint16)
            progress.update_detail(f"TIFF: {tiff_path.name}")

            # Save calibration report
            import json
            from dataclasses import asdict
            calib_report_path = output_dir / "color_calibration.json"
            with open(calib_report_path, "w") as f:
                # Convert numpy types to Python types for JSON serialization
                calib_dict = asdict(calibration)
                for key, val in calib_dict.items():
                    if hasattr(val, 'item'):  # numpy scalar
                        calib_dict[key] = val.item()
                json.dump(calib_dict, f, indent=2)
            progress.update_detail(f"Report: {calib_report_path.name}")

            progress.complete_stage()

            print_summary_box(
                [
                    f"Calibrated FITS: {calib_fits_path}",
                    f"Final PNG: {png_path}",
                    f"Final TIFF: {tiff_path}",
                ],
                title=f"{Symbols.SPARKLE} Color Processing Complete {Symbols.SPARKLE}",
            )

            return 0

        except Exception as e:
            print_error(f"Color processing failed: {e}")
            logger.exception("Color processing failed: %s", e)
            return 1

    elif args.command == "cache":
        # Cache management command
        setup_terminal()

        session_path = Path(args.session).resolve()
        output_root = Path(args.out).resolve()

        # Determine session ID and output dir
        if session_path.name == CACHE_DIR:
            # User pointed directly to cache dir
            output_dir = session_path.parent
            session_id = output_dir.name
        elif (session_path / CACHE_DIR).exists():
            # User pointed to processed output folder
            output_dir = session_path
            session_id = session_path.name
        else:
            # User pointed to raw session folder
            session_id = session_path.name
            output_dir = output_root / session_id

        cache = get_session_cache(output_dir, session_id)

        # Perform requested action
        if args.clear:
            if cache.cache_dir.exists():
                cache.clear()
                print_success(f"Cleared all cache for: {session_id}")
            else:
                print_warning(f"No cache found for: {session_id}")
            return 0

        if args.clear_transforms:
            transforms_path = cache.cache_dir / "transforms.json"
            if transforms_path.exists():
                transforms_path.unlink()
                print_success("Cleared transforms cache")
            else:
                print_warning("No transforms cache found")
            return 0

        if args.clear_stack:
            stack_files = ["stack_mono.fits", "stack_rgb_linear.fits",
                           "stack_r.fits", "stack_g.fits", "stack_b.fits"]
            cleared = 0
            for fname in stack_files:
                fpath = cache.cache_dir / fname
                if fpath.exists():
                    fpath.unlink()
                    cleared += 1
            if cleared > 0:
                print_success(f"Cleared {cleared} stack cache files")
            else:
                print_warning("No stack cache found")
            return 0

        # Default: show status
        print_header(f"Cache Status: {session_id}")
        status = cache.get_status()

        if not cache.cache_dir.exists():
            print_info("No cache directory found")
            print_path("Expected location", str(cache.cache_dir))
            return 0

        print_path("Cache directory", str(cache.cache_dir))

        # Show what's cached
        items = [
            ("Quality scores", status["has_scores"], "scores.json"),
            ("Alignment transforms", status["has_transforms"], "transforms.json"),
            ("Mono stack", status["has_stack_mono"], "stack_mono.fits"),
            ("RGB stack", status["has_stack_rgb"], "stack_rgb_linear.fits"),
            ("Channel stacks", status["has_stack_channels"], "stack_r/g/b.fits"),
            ("Calibration", status["has_calibration"], "calibration.json"),
        ]

        print()
        for label, cached, filename in items:
            if cached:
                print(f"  {Colors.GREEN}{Symbols.CHECK}{Colors.RESET} {label}: {Colors.VALUE}{filename}{Colors.RESET}")
            else:
                print(f"  {Colors.DIM}{Symbols.CROSS} {label}: not cached{Colors.RESET}")

        # Show cache size
        cache_size = sum(f.stat().st_size for f in cache.cache_dir.iterdir() if f.is_file())
        print()
        print_metric("Total cache size", f"{cache_size / 1024 / 1024:.1f} MB")

        return 0

    elif args.command == "frames":
        # Frames listing command
        setup_terminal()

        session_path = Path(args.session).resolve()
        output_root = Path(args.out).resolve()

        if not session_path.exists():
            print_error(f"Session not found: {session_path}")
            return 1

        # Discover frames
        lights = list_lights(session_path)
        if len(lights) == 0:
            print_warning(f"No light frames found in: {session_path}")
            return 1

        session_id = session_path.name
        output_dir = output_root / session_id

        # Check for cached scores
        cache = get_session_cache(output_dir, session_id)
        scores = None
        if args.scored or args.keep:
            scores = cache.load_scores()
            if scores is None and (args.scored or args.keep):
                print_info("No cached scores found. Computing scores...")
                scores = rank_frames(lights, workers=None)

        # Apply keep filter if requested
        kept_paths = None
        if args.keep and scores:
            n_keep = int(len(scores) * args.keep)
            kept_scores = scores[:n_keep]
            kept_paths = {Path(s.path) for s in kept_scores}

        # Save to file if requested
        if args.save:
            save_path = Path(args.save)
            with open(save_path, "w") as f:
                for light in lights:
                    if kept_paths is None or light in kept_paths:
                        f.write(f"{light.name}\n")
            print_success(f"Saved {len(kept_paths) if kept_paths else len(lights)} frames to: {save_path}")
            return 0

        # Print frame list
        print_header(f"Frames in: {session_id}")
        print_metric("Total frames", len(lights))

        if scores:
            # Create path -> score mapping
            score_map = {Path(s.path): s for s in scores}

            print()
            print(f"{'#':>4}  {'Frame':<60}  {'Score':>8}  {'Status':<10}")
            print("-" * 90)

            for i, light in enumerate(lights):
                score = score_map.get(light)
                score_val = f"{score.total_score:.3f}" if score else "N/A"

                if kept_paths is not None:
                    if light in kept_paths:
                        status = f"{Colors.GREEN}KEEP{Colors.RESET}"
                    else:
                        status = f"{Colors.DIM}reject{Colors.RESET}"
                else:
                    status = ""

                print(f"{i+1:>4}  {light.name:<60}  {score_val:>8}  {status:<10}")

            if args.keep:
                print()
                print_metric("Keep fraction", f"{args.keep:.0%}")
                print_metric("Frames to keep", len(kept_paths))
                print_metric("Frames to reject", len(lights) - len(kept_paths))
        else:
            print()
            for i, light in enumerate(lights):
                print(f"  {i+1:>4}. {light.name}")

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
