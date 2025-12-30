"""
Configuration dataclasses for dwarf3 stacking pipeline.

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal


class RejectionReason(Enum):
    """Reason codes for frame rejection."""

    FAILED_PREFIX = "failed_prefix"  # Device-flagged failure
    STACKED_PREFIX = "stacked_prefix"  # Device-produced stack (not a light)
    LOW_QUALITY_SCORE = "low_quality_score"  # Below keep threshold
    ALIGNMENT_FAILED = "alignment_failed"  # Could not align to reference
    HEADER_MISMATCH = "header_mismatch"  # Inconsistent dimensions/binning


@dataclass
class RejectedFrame:
    """Record of a rejected frame with reason."""

    path: str
    reason: RejectionReason
    detail: str = ""  # Optional additional info (e.g., score value)


@dataclass
class FrameScore:
    """Quality metrics for a single frame."""

    path: str
    background_median: float
    background_mad: float  # Median absolute deviation
    noise_proxy: float  # MAD-based noise estimate
    saturation_fraction: float  # Fraction of saturated pixels
    composite_score: float  # Combined ranking score (higher = better)


@dataclass
class StackConfig:
    """
    Configuration for the stacking pipeline.

    All parameters are explicitly documented and have sensible defaults.
    """

    # --- Frame selection ---
    keep_fraction: float = 0.92
    """Fraction of frames to keep after quality ranking (0.0-1.0)."""

    exclude_failed_prefix: bool = True
    """Exclude files matching 'failed_*.fits' pattern."""

    exclude_stacked_prefix: bool = True
    """Exclude files matching 'stacked-*.fits' pattern."""

    # --- Quality scoring ---
    saturation_threshold: float = 0.95
    """Pixel values above this fraction of max are considered saturated."""

    # --- Registration ---
    reference: Literal["best", "first"] = "best"
    """Reference frame selection: 'best' (highest score) or 'first' (deterministic)."""

    backend_align: Literal["astroalign", "ecc", "auto"] = "astroalign"
    """Alignment backend: 'astroalign' (star-based), 'ecc' (intensity), or 'auto'."""

    max_align_fail: int = 50
    """Maximum alignment failures before aborting."""

    downsample_for_align: int | None = None
    """Downsample factor for alignment (None = full resolution)."""

    # --- Stacking ---
    sigma: float = 3.0
    """Sigma threshold for sigma-clipped mean."""

    maxiters: int = 5
    """Maximum iterations for sigma clipping."""

    use_gpu: bool = False
    """Use GPU acceleration (CuPy) for stacking if available."""

    full_res: bool = False
    """Output full 4K resolution (streaming mode). Aligns at 2K, stacks at 4K."""

    # --- Debayer ---
    debayer: Literal["none", "rgb", "superpixel", "bayer-first"] = "none"
    """Debayer mode: 'none' (keep mosaic), 'bayer-first' (recommended for OSC),
    'rgb' (legacy per-frame debayer), or 'superpixel' (half-res)."""

    # --- Alignment mode (for Bayer data) ---
    align_mode: Literal["integer", "rgb_affine", "auto"] = "auto"
    """Alignment mode for Bayer/RGB stacking:
    - 'integer': Phase-preserving integer shifts only (translation, no rotation).
                 Best for color fidelity with equatorial tracking (minimal rotation).
    - 'rgb_affine': Debayer first, then apply affine transforms to RGB.
                    Required when rotation/scale correction is needed.
    - 'auto': Use 'integer' for bayer-first debayer mode, 'rgb_affine' otherwise.
    """

    # --- Parallelism ---
    workers: int | None = None
    """Number of parallel workers for scoring/alignment. None = auto-detect (CPU count - 1)."""

    # --- Output ---
    cache_aligned: bool = False
    """Cache aligned frames to disk (increases disk usage)."""

    write_quicklook: bool = True
    """Generate quicklook PNG/TIFF previews."""

    quicklook_percentiles: tuple[float, float] = (1.0, 99.5)
    """Percentiles for quicklook stretch normalization."""

    asinh_a: float = 0.1
    """Asinh stretch parameter for quicklook generation."""

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 < self.keep_fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in (0, 1], got {self.keep_fraction}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.maxiters < 1:
            raise ValueError(f"maxiters must be >= 1, got {self.maxiters}")
        if not 0.0 < self.saturation_threshold <= 1.0:
            raise ValueError(
                f"saturation_threshold must be in (0, 1], got {self.saturation_threshold}"
            )


@dataclass
class ProcessConfig:
    """
    Configuration for post-stack luminance processing.

    Controls background subtraction, noise reduction, stretch, and enhancement.
    """

    # --- Background correction ---
    subtract_bg: bool = True
    """Apply background gradient subtraction."""

    bg_cell_size: int = 64
    """Cell size for background estimation in pixels."""

    # --- Noise reduction ---
    reduce_noise: bool = False
    """Apply masked noise reduction to background only."""

    noise_method: Literal["median", "gaussian", "bilateral"] = "median"
    """Noise reduction filter: 'median', 'gaussian', or 'bilateral'."""

    noise_filter_size: int = 3
    """Kernel size for noise reduction filter."""

    # --- Dynamic range stretch ---
    stretch_mode: Literal["single", "two_stage"] = "two_stage"
    """Stretch method: 'single' (asinh) or 'two_stage' (core protected)."""

    asinh_stretch: float = 0.1
    """Stretch parameter for single asinh mode (lower = more aggressive)."""

    # --- Local contrast enhancement ---
    enhance_contrast: bool = True
    """Apply local contrast enhancement."""

    contrast_scale: int = 50
    """Scale of contrast enhancement in pixels."""

    contrast_strength: float = 0.3
    """Strength of contrast enhancement (0-1)."""

    protect_stars: bool = True
    """Reduce contrast enhancement around stars."""

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.bg_cell_size < 16:
            raise ValueError(f"bg_cell_size must be >= 16, got {self.bg_cell_size}")
        if self.noise_filter_size < 1:
            raise ValueError(f"noise_filter_size must be >= 1, got {self.noise_filter_size}")
        if not 0.0 < self.asinh_stretch <= 1.0:
            raise ValueError(f"asinh_stretch must be in (0, 1], got {self.asinh_stretch}")
        if self.contrast_scale < 1:
            raise ValueError(f"contrast_scale must be >= 1, got {self.contrast_scale}")
        if not 0.0 <= self.contrast_strength <= 1.0:
            raise ValueError(
                f"contrast_strength must be in [0, 1], got {self.contrast_strength}"
            )


@dataclass
class StackResult:
    """
    Result of a stacking run.

    Contains all information needed to understand and reproduce the result.
    """

    # --- Session identification ---
    session_id: str
    """Session identifier (typically the raw folder name)."""

    session_path: str
    """Absolute path to the raw session folder."""

    # --- Frame accounting ---
    inputs: list[str] = field(default_factory=list)
    """All discovered FITS files."""

    kept: list[str] = field(default_factory=list)
    """Frames that passed selection and alignment."""

    rejected: list[RejectedFrame] = field(default_factory=list)
    """Frames rejected with reasons."""

    alignment_failures: list[str] = field(default_factory=list)
    """Frames that failed alignment (subset of rejected)."""

    # --- Quality data ---
    scores: list[FrameScore] = field(default_factory=list)
    """Quality scores for all evaluated frames."""

    reference_frame: str = ""
    """Path to the reference frame used for alignment."""

    # --- Outputs ---
    outputs: dict[str, str] = field(default_factory=dict)
    """Map of output type to path (e.g., 'master_linear' -> '/path/to/master_linear.fits')."""

    # --- Statistics ---
    stats: dict[str, float] = field(default_factory=dict)
    """Computed statistics (e.g., 'snr_proxy', 'total_exposure_s')."""

    # --- Configuration ---
    config: StackConfig | None = None
    """Configuration used for this run."""

    # --- Metadata ---
    version: str = ""
    """Library version."""

    timestamp: str = ""
    """ISO format timestamp of run completion."""

    platform: str = ""
    """Platform information."""
