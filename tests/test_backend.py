"""
Tests for the backend module (GPU/CPU abstraction).

Tests cover:
- GPU availability detection
- Device info retrieval
- ArrayBackend class
- Sigma-clipped mean GPU function (CPU fallback)

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

import numpy as np
import pytest

from dwarf3.backend import (
    ArrayBackend,
    get_array_module,
    get_backend_summary,
    get_device_info,
    is_gpu_available,
    sigma_clip_mean_gpu,
)


class TestGPUAvailability:
    """Tests for GPU availability detection."""

    def test_is_gpu_available_returns_bool(self):
        """is_gpu_available should return a boolean."""
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_get_device_info_returns_dict(self):
        """get_device_info should return a dict with required keys."""
        info = get_device_info()
        assert isinstance(info, dict)
        assert "backend" in info
        assert "device_name" in info
        assert "device_id" in info
        assert info["backend"] in ("numpy", "cupy")

    def test_get_backend_summary_returns_string(self):
        """get_backend_summary should return a non-empty string."""
        summary = get_backend_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestArrayBackend:
    """Tests for ArrayBackend class."""

    def test_cpu_backend_creation(self):
        """ArrayBackend with use_gpu=False should use NumPy."""
        backend = ArrayBackend(use_gpu=False)
        assert backend.xp is np
        assert not backend.use_gpu

    def test_asarray(self):
        """asarray should convert Python list to array."""
        backend = ArrayBackend(use_gpu=False)
        data = [1, 2, 3]
        arr = backend.asarray(data)
        assert arr.shape == (3,)
        np.testing.assert_array_equal(arr, [1, 2, 3])

    def test_to_numpy_passthrough(self):
        """to_numpy should return NumPy array unchanged."""
        backend = ArrayBackend(use_gpu=False)
        arr = np.array([1, 2, 3])
        result = backend.to_numpy(arr)
        np.testing.assert_array_equal(result, arr)

    def test_zeros(self):
        """zeros should create array of zeros."""
        backend = ArrayBackend(use_gpu=False)
        arr = backend.zeros((3, 4), dtype=np.float32)
        assert arr.shape == (3, 4)
        assert arr.dtype == np.float32
        assert np.all(arr == 0)

    def test_ones(self):
        """ones should create array of ones."""
        backend = ArrayBackend(use_gpu=False)
        arr = backend.ones((2, 3), dtype=np.float32)
        assert arr.shape == (2, 3)
        assert np.all(arr == 1)

    def test_stack(self):
        """stack should stack arrays along axis."""
        backend = ArrayBackend(use_gpu=False)
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        stacked = backend.stack([a, b], axis=0)
        assert stacked.shape == (2, 3)
        np.testing.assert_array_equal(stacked[0], [1, 2, 3])
        np.testing.assert_array_equal(stacked[1], [4, 5, 6])

    def test_mean(self):
        """mean should compute mean correctly."""
        backend = ArrayBackend(use_gpu=False)
        arr = backend.asarray([[1, 2], [3, 4]], dtype=np.float32)
        assert backend.mean(arr) == pytest.approx(2.5)
        np.testing.assert_array_almost_equal(backend.mean(arr, axis=0), [2, 3])
        np.testing.assert_array_almost_equal(backend.mean(arr, axis=1), [1.5, 3.5])

    def test_std(self):
        """std should compute standard deviation."""
        backend = ArrayBackend(use_gpu=False)
        arr = backend.asarray([1, 2, 3, 4, 5], dtype=np.float32)
        assert backend.std(arr) == pytest.approx(np.std([1, 2, 3, 4, 5]))

    def test_clip(self):
        """clip should clip values."""
        backend = ArrayBackend(use_gpu=False)
        arr = backend.asarray([-1, 0, 1, 2, 3], dtype=np.float32)
        clipped = backend.clip(arr, 0, 2)
        np.testing.assert_array_equal(clipped, [0, 0, 1, 2, 2])

    def test_where(self):
        """where should select elements conditionally."""
        backend = ArrayBackend(use_gpu=False)
        arr = backend.asarray([1, 2, 3, 4, 5], dtype=np.float32)
        result = backend.where(arr > 3, arr, 0)
        np.testing.assert_array_equal(result, [0, 0, 0, 4, 5])

    def test_sqrt(self):
        """sqrt should compute square root."""
        backend = ArrayBackend(use_gpu=False)
        arr = backend.asarray([1, 4, 9, 16], dtype=np.float32)
        result = backend.sqrt(arr)
        np.testing.assert_array_equal(result, [1, 2, 3, 4])


class TestSigmaClipMeanGPU:
    """Tests for GPU-accelerated sigma-clipped mean (uses CPU fallback)."""

    def test_basic_stacking(self):
        """Basic stacking should work with CPU fallback."""
        frames = [np.full((10, 10), 100, dtype=np.float32) for _ in range(5)]
        stacked, mask_count = sigma_clip_mean_gpu(frames, sigma=3.0, maxiters=5, use_gpu=False)

        assert stacked.shape == (10, 10)
        assert np.allclose(stacked, 100)

    def test_outlier_rejection(self):
        """Outliers should be rejected."""
        # Create frames with one extreme outlier
        frames = [np.full((10, 10), 100, dtype=np.float32) for _ in range(9)]
        frames.append(np.full((10, 10), 10000, dtype=np.float32))

        stacked, mask_count = sigma_clip_mean_gpu(frames, sigma=3.0, maxiters=5, use_gpu=False)

        # Result should be close to 100 (outlier rejected)
        assert np.allclose(stacked, 100, atol=5)

    def test_empty_list_raises(self):
        """Empty frame list should raise ValueError."""
        with pytest.raises(ValueError, match="No frames"):
            sigma_clip_mean_gpu([], sigma=3.0, maxiters=5)

    def test_returns_correct_types(self):
        """Returns should be numpy arrays."""
        frames = [np.random.rand(5, 5).astype(np.float32) for _ in range(3)]
        stacked, mask_count = sigma_clip_mean_gpu(frames, sigma=3.0, use_gpu=False)

        assert isinstance(stacked, np.ndarray)
        assert isinstance(mask_count, np.ndarray)


class TestGetArrayModule:
    """Tests for get_array_module function."""

    def test_returns_numpy_without_gpu(self):
        """Should return numpy when GPU is not requested or available."""
        xp = get_array_module(use_gpu=False)
        assert xp is np
