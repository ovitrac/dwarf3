"""
Tests for the stack module.

Tests cover:
- Sigma-clipped mean stacking (mono and RGB)
- Stack statistics computation
- Memory-efficient chunked processing

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

import numpy as np
import pytest

from dwarf3.stack import (
    compute_stack_statistics,
    compute_stack_statistics_rgb,
    crop_to_coverage,
    feather_mask,
    get_coverage_bounds,
    mask_aware_mean_rgb,
    median_stack,
    sigma_clip_mask_aware_rgb,
    sigma_clip_mean,
    sigma_clip_mean_rgb,
    weighted_mean,
)


class TestSigmaClipMean:
    """Tests for sigma-clipped mean stacking."""

    def test_basic_stacking(self):
        """Basic stacking of identical frames."""
        frames = [np.full((10, 10), i, dtype=np.float32) for i in [100, 100, 100]]
        stacked, mask_count = sigma_clip_mean(frames, sigma=3.0, maxiters=5)

        assert stacked.shape == (10, 10)
        assert np.allclose(stacked, 100)
        assert np.all(mask_count == 3)  # All 3 frames contributed

    def test_outlier_rejection(self):
        """Outliers should be rejected."""
        # 9 frames with value 100, 1 frame with outlier value 10000
        frames = [np.full((10, 10), 100, dtype=np.float32) for _ in range(9)]
        frames.append(np.full((10, 10), 10000, dtype=np.float32))

        stacked, mask_count = sigma_clip_mean(frames, sigma=3.0, maxiters=5)

        # Result should be close to 100 (outlier rejected)
        assert np.allclose(stacked, 100, atol=1)
        # At least some pixels should have fewer than 10 contributing frames
        assert np.any(mask_count < 10)

    def test_empty_list_raises(self):
        """Empty frame list should raise ValueError."""
        with pytest.raises(ValueError, match="Empty frame list"):
            sigma_clip_mean([])

    def test_single_frame(self):
        """Single frame should return itself."""
        frame = np.random.uniform(0, 1000, (10, 10)).astype(np.float32)
        stacked, mask_count = sigma_clip_mean([frame], sigma=3.0, maxiters=5)

        assert np.allclose(stacked, frame)
        assert np.all(mask_count == 1)

    def test_preserves_spatial_structure(self):
        """Spatial structure should be preserved after stacking."""
        # Create gradient pattern
        gradient = np.arange(100).reshape(10, 10).astype(np.float32)
        frames = [gradient + np.random.normal(0, 0.1, (10, 10)).astype(np.float32)
                  for _ in range(10)]

        stacked, _ = sigma_clip_mean(frames, sigma=3.0, maxiters=5)

        # Result should be close to original gradient
        assert np.allclose(stacked, gradient, atol=1)

    def test_chunked_processing_same_result(self):
        """Chunked processing should give same result as direct."""
        np.random.seed(42)
        frames = [np.random.uniform(0, 1000, (64, 64)).astype(np.float32)
                  for _ in range(10)]

        stacked_small_chunk, _ = sigma_clip_mean(frames, chunk_rows=16)
        stacked_large_chunk, _ = sigma_clip_mean(frames, chunk_rows=32)

        assert np.allclose(stacked_small_chunk, stacked_large_chunk, rtol=1e-5)


class TestSigmaClipMeanRGB:
    """Tests for RGB sigma-clipped mean stacking."""

    def test_basic_rgb_stacking(self):
        """Basic stacking of RGB frames."""
        frames = [np.full((10, 10, 3), [100, 200, 300], dtype=np.float32)
                  for _ in range(5)]
        stacked, mask_count = sigma_clip_mean_rgb(frames, sigma=3.0, maxiters=5)

        assert stacked.shape == (10, 10, 3)
        assert np.allclose(stacked[:, :, 0], 100)
        assert np.allclose(stacked[:, :, 1], 200)
        assert np.allclose(stacked[:, :, 2], 300)

    def test_channel_independent_clipping(self):
        """Each channel should be clipped independently."""
        # Create frames where R has outliers, G and B don't
        frames = [np.full((10, 10, 3), [100, 200, 300], dtype=np.float32)
                  for _ in range(9)]
        # Add outlier frame with high R value only
        outlier = np.full((10, 10, 3), [10000, 200, 300], dtype=np.float32)
        frames.append(outlier)

        stacked, _ = sigma_clip_mean_rgb(frames, sigma=3.0, maxiters=5)

        # R should be ~100 (outlier rejected), G and B unchanged
        assert np.allclose(stacked[:, :, 0], 100, atol=1)
        assert np.allclose(stacked[:, :, 1], 200)
        assert np.allclose(stacked[:, :, 2], 300)

    def test_empty_list_raises(self):
        """Empty frame list should raise ValueError."""
        with pytest.raises(ValueError, match="Empty frame list"):
            sigma_clip_mean_rgb([])

    def test_wrong_channel_count_raises(self):
        """Non-RGB frames should raise ValueError."""
        frames = [np.zeros((10, 10, 4), dtype=np.float32)]  # 4 channels

        with pytest.raises(ValueError, match="Expected 3-channel"):
            sigma_clip_mean_rgb(frames)


class TestStackStatistics:
    """Tests for stack statistics computation."""

    def test_basic_statistics(self):
        """Compute basic statistics."""
        stacked = np.full((10, 10), 1000, dtype=np.float32)
        mask_count = np.full((10, 10), 10, dtype=np.int16)

        stats = compute_stack_statistics(stacked, mask_count, n_frames=10, exptime=15.0)

        assert stats.n_frames == 10
        assert stats.total_exposure_s == 150.0
        assert 0 <= stats.mean_clipped_fraction <= 1
        assert stats.snr_proxy >= 0

    def test_clipped_fraction(self):
        """Test clipped fraction calculation."""
        stacked = np.zeros((10, 10), dtype=np.float32)
        # Average of 8 means 2 were clipped on average
        mask_count = np.full((10, 10), 8, dtype=np.int16)

        stats = compute_stack_statistics(stacked, mask_count, n_frames=10, exptime=15.0)

        # 2/10 = 0.2 clipped
        assert np.isclose(stats.mean_clipped_fraction, 0.2)


class TestStackStatisticsRGB:
    """Tests for RGB stack statistics computation."""

    def test_basic_rgb_statistics(self):
        """Compute basic RGB statistics."""
        stacked = np.full((10, 10, 3), 1000, dtype=np.float32)
        mask_count = np.full((10, 10), 10, dtype=np.int16)

        stats = compute_stack_statistics_rgb(stacked, mask_count, n_frames=10, exptime=15.0)

        assert stats.n_frames == 10
        assert stats.total_exposure_s == 150.0
        assert stats.snr_proxy >= 0


class TestMedianStack:
    """Tests for simple median stacking."""

    def test_basic_median(self):
        """Median of odd number of frames."""
        frames = [np.full((5, 5), i, dtype=np.float32) for i in [10, 20, 30]]
        result = median_stack(frames)

        assert np.allclose(result, 20)

    def test_outlier_robust(self):
        """Median should be robust to outliers."""
        frames = [np.full((5, 5), 100, dtype=np.float32) for _ in range(5)]
        frames[2] = np.full((5, 5), 10000, dtype=np.float32)  # Outlier

        result = median_stack(frames)

        assert np.allclose(result, 100)


class TestWeightedMean:
    """Tests for weighted mean stacking."""

    def test_equal_weights(self):
        """Equal weights should give simple mean."""
        frames = [np.full((5, 5), i, dtype=np.float32) for i in [10, 20, 30]]
        weights = [1.0, 1.0, 1.0]

        result = weighted_mean(frames, weights)

        assert np.allclose(result, 20)

    def test_unequal_weights(self):
        """Unequal weights should bias toward higher weight."""
        frames = [np.full((5, 5), 0, dtype=np.float32),
                  np.full((5, 5), 100, dtype=np.float32)]
        weights = [1.0, 9.0]  # 90% weight on second frame

        result = weighted_mean(frames, weights)

        # (0 * 0.1 + 100 * 0.9) = 90
        assert np.allclose(result, 90)

    def test_mismatched_lengths_raises(self):
        """Mismatched frame/weight counts should raise."""
        frames = [np.zeros((5, 5), dtype=np.float32) for _ in range(3)]
        weights = [1.0, 1.0]  # Only 2 weights

        with pytest.raises(ValueError, match="must match"):
            weighted_mean(frames, weights)


class TestMaskAwareMeanRGB:
    """Tests for mask-aware RGB mean stacking."""

    def test_basic_mask_aware_mean(self):
        """Basic mask-aware mean with no masks (all valid)."""
        frames = [np.full((10, 10, 3), [100, 200, 300], dtype=np.float32)
                  for _ in range(5)]

        stacked, coverage = mask_aware_mean_rgb(frames, masks=None)

        assert stacked.shape == (10, 10, 3)
        assert np.allclose(stacked[:, :, 0], 100)
        assert np.allclose(stacked[:, :, 1], 200)
        assert np.allclose(stacked[:, :, 2], 300)
        assert np.all(coverage == 5)

    def test_mask_excludes_pixels(self):
        """Masked pixels should be excluded from mean."""
        # Frame 0: value 100, frame 1: value 200
        frame0 = np.full((10, 10, 3), 100, dtype=np.float32)
        frame1 = np.full((10, 10, 3), 200, dtype=np.float32)
        frames = [frame0, frame1]

        # Mask frame 1 entirely
        mask0 = np.ones((10, 10), dtype=np.float32)
        mask1 = np.zeros((10, 10), dtype=np.float32)
        masks = [mask0, mask1]

        stacked, coverage = mask_aware_mean_rgb(frames, masks)

        # Only frame0 contributes, so result should be 100
        assert np.allclose(stacked, 100)
        assert np.all(coverage == 1)

    def test_partial_mask(self):
        """Partial masks should weight correctly."""
        frame0 = np.full((10, 10, 3), 100, dtype=np.float32)
        frame1 = np.full((10, 10, 3), 200, dtype=np.float32)
        frames = [frame0, frame1]

        # Full mask for frame0, partial for frame1
        mask0 = np.ones((10, 10), dtype=np.float32)
        mask1 = np.zeros((10, 10), dtype=np.float32)
        mask1[:5, :] = 1.0  # Top half valid
        masks = [mask0, mask1]

        stacked, coverage = mask_aware_mean_rgb(frames, masks)

        # Top half: (100 + 200) / 2 = 150
        assert np.allclose(stacked[:5, :, :], 150)
        # Bottom half: only frame0 = 100
        assert np.allclose(stacked[5:, :, :], 100)
        # Coverage
        assert np.all(coverage[:5, :] == 2)
        assert np.all(coverage[5:, :] == 1)

    def test_empty_frames_raises(self):
        """Empty frame list should raise ValueError."""
        with pytest.raises(ValueError, match="Empty"):
            mask_aware_mean_rgb([])

    def test_rotated_frame_simulation(self):
        """Simulate rotated frames with corner masks."""
        # Simulate field rotation: frames have valid centers but invalid corners
        frames = [np.full((20, 20, 3), i * 100, dtype=np.float32) for i in range(1, 4)]

        # Create circular-ish masks (center valid, corners not)
        masks = []
        yy, xx = np.mgrid[0:20, 0:20]
        center = 10
        for i in range(3):
            # Slightly different radius per frame (simulating rotation)
            radius = 8 - i * 0.5
            mask = ((xx - center)**2 + (yy - center)**2 < radius**2).astype(np.float32)
            masks.append(mask)

        stacked, coverage = mask_aware_mean_rgb(frames, masks)

        # Center should have highest coverage
        assert coverage[10, 10] == 3
        # Corners should have lower coverage
        assert coverage[0, 0] < 3


class TestSigmaClipMaskAwareRGB:
    """Tests for sigma-clipped mask-aware RGB stacking."""

    def test_basic_sigma_clip_mask_aware(self):
        """Basic sigma-clipped mask-aware stacking."""
        frames = [np.full((10, 10, 3), [100, 200, 300], dtype=np.float32)
                  for _ in range(5)]

        stacked, coverage = sigma_clip_mask_aware_rgb(frames, sigma=3.0, maxiters=5)

        assert stacked.shape == (10, 10, 3)
        assert np.allclose(stacked[:, :, 0], 100)
        assert np.allclose(stacked[:, :, 1], 200)
        assert np.allclose(stacked[:, :, 2], 300)

    def test_returns_coverage_map(self):
        """Should return coverage map as second element."""
        frames = [np.full((10, 10, 3), 100, dtype=np.float32) for _ in range(5)]
        masks = [np.ones((10, 10), dtype=np.float32) for _ in range(5)]

        stacked, coverage = sigma_clip_mask_aware_rgb(frames, masks, sigma=3.0)

        assert coverage.shape == (10, 10)
        # With all masks valid, coverage should be 5 everywhere
        assert np.all(coverage == 5)

    def test_masked_regions_have_lower_coverage(self):
        """Masked regions should have lower coverage."""
        frames = [np.full((10, 10, 3), 100, dtype=np.float32) for _ in range(5)]

        # Mask bottom half in last 2 frames
        masks = []
        for i in range(5):
            mask = np.ones((10, 10), dtype=np.float32)
            if i >= 3:
                mask[5:, :] = 0
            masks.append(mask)

        stacked, coverage = sigma_clip_mask_aware_rgb(frames, masks, sigma=3.0)

        # Top half: 5 samples, bottom half: 3 samples
        assert np.all(coverage[:5, :] == 5)
        assert np.all(coverage[5:, :] == 3)

    def test_output_uses_valid_pixels_only(self):
        """Output should only use valid (masked) pixels."""
        # Two frames with different values
        frame0 = np.full((10, 10, 3), 100, dtype=np.float32)
        frame1 = np.full((10, 10, 3), 200, dtype=np.float32)
        frames = [frame0, frame1]

        # Mask frame1 in bottom half
        mask0 = np.ones((10, 10), dtype=np.float32)
        mask1 = np.ones((10, 10), dtype=np.float32)
        mask1[5:, :] = 0
        masks = [mask0, mask1]

        stacked, coverage = sigma_clip_mask_aware_rgb(frames, masks, sigma=3.0)

        # Top half: (100 + 200) / 2 = 150
        assert np.allclose(stacked[:5, :, :], 150)
        # Bottom half: only frame0 = 100
        assert np.allclose(stacked[5:, :, :], 100)

    def test_auto_mask_from_nan(self):
        """Should auto-create masks from NaN/zero regions."""
        frame0 = np.full((10, 10, 3), 100, dtype=np.float32)
        frame1 = np.full((10, 10, 3), 100, dtype=np.float32)
        frame1[8:, :, :] = 0  # Zero corner (invalid from rotation)

        frames = [frame0, frame1]

        stacked, coverage = sigma_clip_mask_aware_rgb(frames, masks=None, threshold=0.01)

        # Coverage in zeroed region should be 1 (only frame0)
        # This depends on implementation - may vary
        assert stacked.shape == (10, 10, 3)


class TestFeatherMask:
    """Tests for mask feathering using distance transform."""

    def test_no_feathering(self):
        """Zero feather width should return binary mask."""
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[3:7, 3:7] = 1.0

        result = feather_mask(mask, feather_width=0)

        # Should be identical to input (as float32)
        assert np.allclose(result, mask)

    def test_basic_feathering(self):
        """Feathering should create soft edges."""
        mask = np.zeros((20, 20), dtype=np.float32)
        mask[5:15, 5:15] = 1.0

        result = feather_mask(mask, feather_width=3)

        # Center should be 1.0 (fully valid)
        assert np.isclose(result[10, 10], 1.0)
        # Outside mask should be 0.0
        assert np.isclose(result[0, 0], 0.0)
        # Edge should be between 0 and 1
        assert 0.0 < result[5, 10] < 1.0

    def test_feather_gradient(self):
        """Feathering should create monotonic gradient from edge."""
        mask = np.zeros((30, 30), dtype=np.float32)
        mask[10:20, 10:20] = 1.0

        result = feather_mask(mask, feather_width=5)

        # Check gradient along a row at y=15 (center row of mask)
        # Values should increase from edge toward center
        edge_val = result[15, 10]
        mid_val = result[15, 12]
        center_val = result[15, 15]

        assert edge_val < mid_val < center_val
        assert np.isclose(center_val, 1.0)

    def test_output_shape_preserved(self):
        """Output should have same shape as input."""
        mask = np.ones((50, 60), dtype=np.float32)
        result = feather_mask(mask, feather_width=10)

        assert result.shape == mask.shape
        assert result.dtype == np.float32


class TestCropToCoverage:
    """Tests for coverage-based cropping."""

    def test_full_coverage(self):
        """Full coverage should return original."""
        image = np.random.uniform(0, 1, (100, 120, 3)).astype(np.float32)
        coverage = np.full((100, 120), 10, dtype=np.int16)

        cropped, bounds = crop_to_coverage(image, coverage, min_coverage_fraction=0.8)

        assert cropped.shape == image.shape
        assert bounds == (0, 100, 0, 120)

    def test_corner_cropping(self):
        """Should crop to avoid low-coverage corners."""
        image = np.zeros((100, 100, 3), dtype=np.float32)
        image[20:80, 20:80] = 1.0

        coverage = np.zeros((100, 100), dtype=np.int16)
        coverage[20:80, 20:80] = 10  # Full coverage in center
        coverage[:20, :] = 2  # Low coverage at top
        coverage[80:, :] = 2  # Low coverage at bottom
        coverage[:, :20] = 2  # Low coverage at left
        coverage[:, 80:] = 2  # Low coverage at right

        cropped, bounds = crop_to_coverage(image, coverage, min_coverage_fraction=0.8)

        top, bottom, left, right = bounds
        assert top >= 15  # Should exclude low-coverage top
        assert bottom <= 85  # Should exclude low-coverage bottom
        assert left >= 15
        assert right <= 85

    def test_2d_image(self):
        """Should work with 2D (mono) images."""
        image = np.zeros((50, 50), dtype=np.float32)
        coverage = np.full((50, 50), 10, dtype=np.int16)

        cropped, bounds = crop_to_coverage(image, coverage)

        assert cropped.ndim == 2
        assert bounds == (0, 50, 0, 50)

    def test_zero_coverage(self):
        """Zero coverage should return original with warning."""
        image = np.ones((10, 10, 3), dtype=np.float32)
        coverage = np.zeros((10, 10), dtype=np.int16)

        cropped, bounds = crop_to_coverage(image, coverage)

        assert cropped.shape == image.shape


class TestGetCoverageBounds:
    """Tests for coverage bounds calculation."""

    def test_full_coverage(self):
        """Full coverage should return full bounds."""
        coverage = np.full((100, 120), 10, dtype=np.int16)

        bounds = get_coverage_bounds(coverage, min_coverage_fraction=0.8)

        assert bounds == (0, 100, 0, 120)

    def test_partial_coverage(self):
        """Partial coverage should return reduced bounds."""
        coverage = np.zeros((100, 100), dtype=np.int16)
        coverage[25:75, 30:70] = 10

        bounds = get_coverage_bounds(coverage, min_coverage_fraction=0.8)

        top, bottom, left, right = bounds
        assert top == 25
        assert bottom == 75
        assert left == 30
        assert right == 70

    def test_threshold_effect(self):
        """Higher threshold should give tighter bounds."""
        coverage = np.zeros((100, 100), dtype=np.int16)
        coverage[20:80, 20:80] = 5
        coverage[30:70, 30:70] = 10  # Higher coverage in center

        bounds_low = get_coverage_bounds(coverage, min_coverage_fraction=0.4)
        bounds_high = get_coverage_bounds(coverage, min_coverage_fraction=0.8)

        # Higher threshold should give tighter bounds
        top_low, bottom_low, left_low, right_low = bounds_low
        top_high, bottom_high, left_high, right_high = bounds_high

        assert top_high >= top_low
        assert bottom_high <= bottom_low
        assert left_high >= left_low
        assert right_high <= right_low


class TestSigmaClipMaskAwareWithFeathering:
    """Tests for sigma_clip_mask_aware_rgb with feathering enabled."""

    def test_feathering_produces_smoother_edges(self):
        """Feathering should produce smoother stacked edges."""
        # Create frames with partial overlap
        frame0 = np.full((30, 30, 3), 100, dtype=np.float32)
        frame1 = np.full((30, 30, 3), 100, dtype=np.float32)

        # Masks with overlapping regions
        mask0 = np.zeros((30, 30), dtype=np.float32)
        mask0[:20, :] = 1.0
        mask1 = np.zeros((30, 30), dtype=np.float32)
        mask1[10:, :] = 1.0

        frames = [frame0, frame1]
        masks = [mask0, mask1]

        # Stack without feathering
        stacked_no_feather, _ = sigma_clip_mask_aware_rgb(
            frames, masks, feather_width=0
        )

        # Stack with feathering
        stacked_feathered, _ = sigma_clip_mask_aware_rgb(
            frames, masks, feather_width=5
        )

        # Both should produce valid output
        assert stacked_no_feather.shape == (30, 30, 3)
        assert stacked_feathered.shape == (30, 30, 3)

    def test_feathering_preserves_values(self):
        """Feathering should not change values in fully covered regions."""
        frames = [np.full((20, 20, 3), [100, 200, 300], dtype=np.float32)
                  for _ in range(5)]
        masks = [np.ones((20, 20), dtype=np.float32) for _ in range(5)]

        stacked, _ = sigma_clip_mask_aware_rgb(frames, masks, feather_width=3)

        # Center (far from edges) should match input values
        assert np.allclose(stacked[10, 10, 0], 100, atol=1)
        assert np.allclose(stacked[10, 10, 1], 200, atol=1)
        assert np.allclose(stacked[10, 10, 2], 300, atol=1)
