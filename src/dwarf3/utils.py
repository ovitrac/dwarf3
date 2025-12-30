"""
Utility functions for dwarf3 pipeline.

Includes:
- Image stretching for visualization
- File/path helpers
- Version info

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

__version__ = "0.22.0-beta"
__version_info__ = {
    "major": 0,
    "minor": 22,
    "patch": 0,
    "status": "beta",
    "date": "2025-12-29",
}


def get_version_banner() -> str:
    """Return a formatted version banner for logging."""
    return f"dwarf3 v{__version__} | DWARF 3 Astrophotography Pipeline"


def get_version() -> str:
    """Return the library version string."""
    return __version__


def get_platform_info() -> str:
    """Return platform information string."""
    return f"{platform.system()} {platform.release()} / Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_timestamp_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def asinh_stretch(
    data: np.ndarray,
    a: float = 0.1,
    percentiles: tuple[float, float] = (1.0, 99.5),
) -> np.ndarray:
    """
    Apply asinh stretch for visualization.

    The asinh function compresses bright regions while preserving
    detail in faint structures. It approximates:
    - Linear for small values
    - Logarithmic for large values

    Parameters
    ----------
    data : np.ndarray
        Input image (linear scale).
    a : float, default 0.1
        Softening parameter. Smaller values = stronger compression.
    percentiles : tuple[float, float], default (1.0, 99.5)
        Percentiles for normalization (black point, white point).

    Returns
    -------
    np.ndarray
        Stretched image normalized to [0, 1] range (float32).

    Notes
    -----
    The formula is: stretched = asinh(a * normalized) / asinh(a)
    where normalized is the percentile-scaled input.
    """
    # Compute percentile bounds
    vmin = np.percentile(data, percentiles[0])
    vmax = np.percentile(data, percentiles[1])

    # Normalize to [0, 1]
    if vmax - vmin < 1e-10:
        return np.zeros_like(data, dtype=np.float32)

    normalized = (data - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    # Apply asinh stretch
    stretched = np.arcsinh(a * normalized) / np.arcsinh(a)

    return stretched.astype(np.float32)


def linear_stretch(
    data: np.ndarray,
    percentiles: tuple[float, float] = (1.0, 99.5),
) -> np.ndarray:
    """
    Apply simple linear percentile stretch.

    Parameters
    ----------
    data : np.ndarray
        Input image.
    percentiles : tuple[float, float], default (1.0, 99.5)
        Percentiles for black and white points.

    Returns
    -------
    np.ndarray
        Stretched image normalized to [0, 1] range (float32).
    """
    vmin = np.percentile(data, percentiles[0])
    vmax = np.percentile(data, percentiles[1])

    if vmax - vmin < 1e-10:
        return np.zeros_like(data, dtype=np.float32)

    stretched = (data - vmin) / (vmax - vmin)
    stretched = np.clip(stretched, 0, 1)

    return stretched.astype(np.float32)


def to_uint8(data: np.ndarray) -> np.ndarray:
    """
    Convert normalized [0,1] float array to uint8 [0,255].

    Parameters
    ----------
    data : np.ndarray
        Input array with values in [0, 1] range.

    Returns
    -------
    np.ndarray
        Output array with dtype uint8.
    """
    return (np.clip(data, 0, 1) * 255).astype(np.uint8)


def to_uint16(data: np.ndarray) -> np.ndarray:
    """
    Convert normalized [0,1] float array to uint16 [0,65535].

    Parameters
    ----------
    data : np.ndarray
        Input array with values in [0, 1] range.

    Returns
    -------
    np.ndarray
        Output array with dtype uint16.
    """
    return (np.clip(data, 0, 1) * 65535).astype(np.uint16)


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename.

    Replaces problematic characters with underscores.

    Parameters
    ----------
    name : str
        Input filename or path component.

    Returns
    -------
    str
        Sanitized string safe for use in filenames.
    """
    # Characters that are problematic on various filesystems
    bad_chars = '<>:"/\\|?*'
    result = name
    for char in bad_chars:
        result = result.replace(char, "_")
    return result


def ensure_output_dir(output_root: Path, session_id: str) -> Path:
    """
    Create and return the output directory for a session.

    Parameters
    ----------
    output_root : Path
        Root output directory (e.g., processedData/).
    session_id : str
        Session identifier (folder name).

    Returns
    -------
    Path
        Path to the session output directory.
    """
    session_dir = output_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (session_dir / "stacked").mkdir(exist_ok=True)

    return session_dir


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable form.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted string like "2h 15m 30s" or "45.2s".
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"
