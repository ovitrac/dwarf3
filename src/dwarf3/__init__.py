"""
dwarf3 - Reproducible stacking pipeline for DWARF 3 smart telescope.

A scientific, scriptable, reproducible processing library for DWARF 3
smart telescope FITS acquisitions.

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com

Example
-------
>>> from dwarf3 import stack_session, StackConfig
>>> config = StackConfig(keep_fraction=0.92, sigma=3.0)
>>> result = stack_session("rawData/DWARF_RAW_TELE_M31_...", config=config)
>>> print(result.outputs["master_linear"])

Example (RGB mode)
------------------
>>> config = StackConfig(keep_fraction=0.92, debayer="superpixel")
>>> result = stack_session("rawData/DWARF_RAW_TELE_M31_...", config=config)
>>> print(result.outputs["master_rgb"])  # RGB FITS output
"""

from .config import (
    FrameScore,
    ProcessConfig,
    RejectedFrame,
    RejectionReason,
    StackConfig,
    StackResult,
)
from .utils import __version__, __version_info__, get_version_banner

# Primary entry point
from .cli import stack_session

# I/O functions
from .io import list_lights, read_fits, read_header, write_fits

# Quality assessment
from .quality import rank_frames, score_frame, select_frames

# Alignment
from .align import (
    align_bayer_frames,
    align_bayer_integer,
    align_frames,
    align_frames_parallel,
    align_rgb_debayer_first,
    align_rgb_debayer_first_parallel,
    align_rgb_debayer_first_shm,
    align_rgb_frames,
    analyze_field_rotation,
    apply_cached_transforms,
    apply_integer_shift,
    apply_transform_to_image,
    estimate_integer_shift,
    find_transform,
    load_transforms,
    register_to_reference,
    save_transforms,
    scale_transform_for_resolution,
    select_reference_frame,
)

# Stacking
from .stack import (
    compute_stack_statistics,
    compute_stack_statistics_rgb,
    crop_to_coverage,
    feather_mask,
    get_coverage_bounds,
    mask_aware_mean_rgb,
    sigma_clip_mask_aware_rgb,
    sigma_clip_mean,
    sigma_clip_mean_rgb,
)

# Debayer
from .debayer import (
    bayer_luma_rggb,
    debayer_rggb,
    extract_luminance,
    get_channel,
    luminance_from_rgb,
)

# Visualization helpers
from .utils import asinh_stretch, linear_stretch

# Post-stack processing
from .processing import (
    create_galaxy_mask,
    create_object_mask,
    create_soft_mask,
    create_star_mask,
    estimate_background,
    local_contrast_enhancement,
    process_luminance,
    reduce_background_noise,
    subtract_background,
    two_stage_stretch,
)

# Colormap / pseudo-color
from .colormap import (
    apply_bicolor,
    apply_colormap,
    blend_with_luminance,
    create_colorbar,
    list_palettes,
)

# Backend / GPU acceleration
from .backend import (
    ArrayBackend,
    get_array_module,
    get_backend_summary,
    get_device_info,
    is_gpu_available,
    sigma_clip_mean_gpu,
)

# Cache management
from .cache import (
    PipelineCache,
    CacheManifest,
    get_session_cache,
)

# Color processing (LRGB workflow)
from .color import (
    CCM_PRESETS,
    ColorCalibration,
    LRGBResult,
    apply_bayer_compensation,
    apply_ccm,
    calibrate_rgb,
    combine_lrgb,
    compute_white_balance,
    create_extended_object_mask,
    scnr_green,
    stretch_rgb_linked,
)

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    "get_version_banner",
    # Config
    "StackConfig",
    "StackResult",
    "ProcessConfig",
    "FrameScore",
    "RejectedFrame",
    "RejectionReason",
    # Main entry point
    "stack_session",
    # I/O
    "list_lights",
    "read_fits",
    "read_header",
    "write_fits",
    # Quality
    "score_frame",
    "rank_frames",
    "select_frames",
    # Alignment
    "register_to_reference",
    "align_bayer_frames",
    "align_bayer_integer",
    "align_frames",
    "align_frames_parallel",
    "align_rgb_debayer_first",
    "align_rgb_debayer_first_parallel",
    "align_rgb_debayer_first_shm",
    "align_rgb_frames",
    "analyze_field_rotation",
    "apply_cached_transforms",
    "apply_integer_shift",
    "estimate_integer_shift",
    "select_reference_frame",
    "find_transform",
    "apply_transform_to_image",
    "scale_transform_for_resolution",
    "save_transforms",
    "load_transforms",
    # Stacking
    "sigma_clip_mean",
    "sigma_clip_mean_rgb",
    "sigma_clip_mask_aware_rgb",
    "mask_aware_mean_rgb",
    "compute_stack_statistics",
    "compute_stack_statistics_rgb",
    "feather_mask",
    "crop_to_coverage",
    "get_coverage_bounds",
    # Debayer
    "bayer_luma_rggb",
    "debayer_rggb",
    "extract_luminance",
    "luminance_from_rgb",
    "get_channel",
    # Visualization
    "asinh_stretch",
    "linear_stretch",
    # Processing
    "estimate_background",
    "subtract_background",
    "two_stage_stretch",
    "create_star_mask",
    "create_object_mask",
    "create_galaxy_mask",
    "create_soft_mask",
    "local_contrast_enhancement",
    "reduce_background_noise",
    "process_luminance",
    # Colormap
    "apply_colormap",
    "apply_bicolor",
    "blend_with_luminance",
    "create_colorbar",
    "list_palettes",
    # Backend / GPU
    "is_gpu_available",
    "get_device_info",
    "get_backend_summary",
    "get_array_module",
    "ArrayBackend",
    "sigma_clip_mean_gpu",
    # Cache
    "PipelineCache",
    "CacheManifest",
    "get_session_cache",
    # Color / LRGB
    "CCM_PRESETS",
    "ColorCalibration",
    "LRGBResult",
    "apply_bayer_compensation",
    "apply_ccm",
    "calibrate_rgb",
    "combine_lrgb",
    "compute_white_balance",
    "create_extended_object_mask",
    "scnr_green",
    "stretch_rgb_linked",
]
