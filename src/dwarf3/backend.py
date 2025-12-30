"""
Computational backend abstraction for dwarf3.

Provides NumPy-compatible array operations with optional GPU acceleration
via CuPy when available. Falls back to NumPy for CPU-only systems.

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Try to import CuPy
_CUPY_AVAILABLE = False
_cupy = None

try:
    import cupy as cp
    _cupy = cp
    _CUPY_AVAILABLE = True
    logger.debug("CuPy available: GPU acceleration enabled")
except ImportError:
    logger.debug("CuPy not available: using NumPy (CPU only)")


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.

    Returns
    -------
    bool
        True if CuPy is installed and a GPU is available.
    """
    if not _CUPY_AVAILABLE:
        return False

    try:
        # Try to access a GPU
        _cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def get_device_info() -> dict:
    """
    Get information about the compute device.

    Returns
    -------
    dict
        Device information including name, memory, and backend type.
    """
    if is_gpu_available():
        device = _cupy.cuda.Device()
        props = _cupy.cuda.runtime.getDeviceProperties(device.id)
        return {
            "backend": "cupy",
            "device_name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
            "device_id": device.id,
            "total_memory_gb": props["totalGlobalMem"] / (1024**3),
            "compute_capability": f"{props['major']}.{props['minor']}",
        }
    else:
        return {
            "backend": "numpy",
            "device_name": "CPU",
            "device_id": -1,
            "total_memory_gb": None,
            "compute_capability": None,
        }


class ArrayBackend:
    """
    Array computation backend abstraction.

    Provides a unified interface for NumPy and CuPy operations.
    Automatically transfers arrays between CPU and GPU as needed.

    Parameters
    ----------
    use_gpu : bool, default True
        Whether to use GPU if available.

    Examples
    --------
    >>> backend = ArrayBackend(use_gpu=True)
    >>> x = backend.asarray([1, 2, 3])
    >>> y = backend.sum(x)
    >>> result = backend.to_numpy(y)
    """

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and is_gpu_available()

        if self.use_gpu:
            self._xp = _cupy
            logger.info("Using CuPy backend (GPU)")
        else:
            self._xp = np
            logger.info("Using NumPy backend (CPU)")

    @property
    def xp(self):
        """Get the array module (NumPy or CuPy)."""
        return self._xp

    def asarray(self, data, dtype=None):
        """Convert data to array on the appropriate device."""
        return self._xp.asarray(data, dtype=dtype)

    def to_numpy(self, arr):
        """Convert array back to NumPy (CPU) array."""
        if self.use_gpu and hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    def zeros(self, shape, dtype=np.float32):
        """Create array of zeros."""
        return self._xp.zeros(shape, dtype=dtype)

    def zeros_like(self, arr, dtype=None):
        """Create array of zeros with same shape."""
        return self._xp.zeros_like(arr, dtype=dtype)

    def ones(self, shape, dtype=np.float32):
        """Create array of ones."""
        return self._xp.ones(shape, dtype=dtype)

    def empty(self, shape, dtype=np.float32):
        """Create uninitialized array."""
        return self._xp.empty(shape, dtype=dtype)

    def stack(self, arrays, axis=0):
        """Stack arrays along axis."""
        return self._xp.stack(arrays, axis=axis)

    def mean(self, arr, axis=None):
        """Compute mean."""
        return self._xp.mean(arr, axis=axis)

    def median(self, arr, axis=None):
        """Compute median."""
        return self._xp.median(arr, axis=axis)

    def std(self, arr, axis=None, ddof=0):
        """Compute standard deviation."""
        return self._xp.std(arr, axis=axis, ddof=ddof)

    def sum(self, arr, axis=None):
        """Compute sum."""
        return self._xp.sum(arr, axis=axis)

    def abs(self, arr):
        """Compute absolute value."""
        return self._xp.abs(arr)

    def clip(self, arr, a_min, a_max):
        """Clip array values."""
        return self._xp.clip(arr, a_min, a_max)

    def where(self, condition, x, y):
        """Element-wise selection."""
        return self._xp.where(condition, x, y)

    def percentile(self, arr, q, axis=None):
        """Compute percentile."""
        return self._xp.percentile(arr, q, axis=axis)

    def maximum(self, x1, x2):
        """Element-wise maximum."""
        return self._xp.maximum(x1, x2)

    def minimum(self, x1, x2):
        """Element-wise minimum."""
        return self._xp.minimum(x1, x2)

    def sqrt(self, arr):
        """Element-wise square root."""
        return self._xp.sqrt(arr)

    def exp(self, arr):
        """Element-wise exponential."""
        return self._xp.exp(arr)

    def log(self, arr):
        """Element-wise natural logarithm."""
        return self._xp.log(arr)

    def arcsinh(self, arr):
        """Element-wise inverse hyperbolic sine."""
        return self._xp.arcsinh(arr)


def sigma_clip_mean_gpu(
    frames: list[np.ndarray],
    sigma: float = 3.0,
    maxiters: int = 5,
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated sigma-clipped mean stacking.

    Parameters
    ----------
    frames : list of np.ndarray
        List of 2D images to stack.
    sigma : float, default 3.0
        Clipping threshold in standard deviations.
    maxiters : int, default 5
        Maximum number of clipping iterations.
    use_gpu : bool, default True
        Whether to use GPU if available.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (stacked_image, mask_count)
        stacked_image is the sigma-clipped mean.
        mask_count shows how many values were clipped at each pixel.
    """
    backend = ArrayBackend(use_gpu=use_gpu)
    xp = backend.xp

    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("No frames to stack")

    shape = frames[0].shape

    # Transfer frames to device
    cube = xp.stack([backend.asarray(f) for f in frames], axis=0)
    cube = cube.astype(xp.float32)

    # Initialize mask (True = valid, False = clipped)
    mask = xp.ones((n_frames, *shape), dtype=bool)

    for iteration in range(maxiters):
        # Compute mean and std of unmasked values
        masked_cube = xp.where(mask, cube, xp.nan)

        with np.errstate(all="ignore"):
            mean_img = xp.nanmean(masked_cube, axis=0)
            std_img = xp.nanstd(masked_cube, axis=0)

        # Compute deviations
        deviation = xp.abs(cube - mean_img)

        # Clip values beyond threshold
        threshold = sigma * std_img
        new_mask = deviation <= threshold

        # Combine with existing mask
        new_valid = mask & new_mask

        # Check convergence
        n_clipped_new = xp.sum(~new_valid) - xp.sum(~mask)
        if int(n_clipped_new) == 0:
            break

        mask = new_valid

    # Final mean
    masked_cube = xp.where(mask, cube, xp.nan)
    with np.errstate(all="ignore"):
        result = xp.nanmean(masked_cube, axis=0)

    # Replace NaN with 0 (shouldn't happen unless all values clipped)
    result = xp.where(xp.isnan(result), 0, result)

    # Count clipped values per pixel
    mask_count = xp.sum(~mask, axis=0).astype(xp.int32)

    # Transfer back to CPU
    result_cpu = backend.to_numpy(result)
    mask_count_cpu = backend.to_numpy(mask_count)

    return result_cpu, mask_count_cpu


def get_backend_summary() -> str:
    """
    Get a human-readable summary of the compute backend.

    Returns
    -------
    str
        Summary string describing the backend configuration.
    """
    info = get_device_info()
    if info["backend"] == "cupy":
        return (
            f"CuPy/GPU: {info['device_name']} "
            f"({info['total_memory_gb']:.1f} GB, "
            f"compute {info['compute_capability']})"
        )
    else:
        return "NumPy/CPU"


# Convenience function to get the appropriate array module
def get_array_module(use_gpu: bool = True):
    """
    Get the appropriate array module (NumPy or CuPy).

    Parameters
    ----------
    use_gpu : bool, default True
        Whether to use GPU if available.

    Returns
    -------
    module
        Either numpy or cupy module.
    """
    if use_gpu and is_gpu_available():
        return _cupy
    return np
