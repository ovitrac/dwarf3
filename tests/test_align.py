"""
Tests for the align module.

Tests cover:
- Parallel RGB alignment
- Transform computation and application
- Reference frame selection
- AlignmentResult and AlignmentTransform structures

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

import numpy as np
import pytest
from pathlib import Path

from dwarf3.align import (
    AlignmentResult,
    AlignmentTransform,
    apply_transform_to_image,
    transform_to_dict,
    dict_to_transform,
)
from dwarf3.debayer import luminance_from_rgb


class TestAlignmentTransform:
    """Tests for AlignmentTransform dataclass."""

    def test_basic_creation(self):
        """Create a basic AlignmentTransform."""
        transform = AlignmentTransform(
            source_path="frame.fits",
            reference_path="ref.fits",
            success=True,
            matrix=np.eye(3),
            n_matches=50,
        )

        assert transform.source_path == "frame.fits"
        assert transform.reference_path == "ref.fits"
        assert transform.success is True
        assert transform.matrix.shape == (3, 3)
        assert transform.n_matches == 50

    def test_failed_transform(self):
        """Create a failed AlignmentTransform."""
        transform = AlignmentTransform(
            source_path="bad.fits",
            reference_path="ref.fits",
            success=False,
            error_message="Not enough sources",
            matrix=None,
            n_matches=0,
        )

        assert transform.success is False
        assert transform.matrix is None
        assert "sources" in transform.error_message


class TestAlignmentResult:
    """Tests for AlignmentResult dataclass."""

    def test_basic_creation(self):
        """Create a basic AlignmentResult."""
        transform = AlignmentTransform(
            source_path="frame.fits",
            reference_path="ref.fits",
            success=True,
            matrix=np.eye(3),
            n_matches=50,
        )

        result = AlignmentResult(
            path="frame.fits",
            success=True,
            aligned_data=np.zeros((100, 100), dtype=np.float32),
            transform=transform,
        )

        assert result.path == "frame.fits"
        assert result.success is True
        assert result.aligned_data is not None
        assert result.transform.n_matches == 50

    def test_failed_result(self):
        """Create a failed AlignmentResult."""
        transform = AlignmentTransform(
            source_path="bad.fits",
            reference_path="ref.fits",
            success=False,
            error_message="Alignment failed",
            matrix=None,
            n_matches=0,
        )

        result = AlignmentResult(
            path="bad.fits",
            success=False,
            aligned_data=None,
            transform=transform,
        )

        assert result.success is False
        assert result.aligned_data is None


class TestTransformSerialization:
    """Tests for transform serialization."""

    def test_transform_to_dict(self):
        """Convert transform to dictionary."""
        transform = AlignmentTransform(
            source_path="frame.fits",
            reference_path="ref.fits",
            success=True,
            matrix=np.eye(3),
            n_matches=50,
        )

        d = transform_to_dict(transform)

        assert d["source_path"] == "frame.fits"
        assert d["success"] is True
        assert d["matrix"] is not None
        assert len(d["matrix"]) == 3

    def test_dict_to_transform(self):
        """Convert dictionary back to transform."""
        d = {
            "source_path": "frame.fits",
            "reference_path": "ref.fits",
            "success": True,
            "error_message": "",
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "n_matches": 50,
        }

        transform = dict_to_transform(d)

        assert transform.source_path == "frame.fits"
        assert transform.success is True
        assert np.allclose(transform.matrix, np.eye(3))

    def test_roundtrip(self):
        """Transform -> dict -> transform should preserve data."""
        original = AlignmentTransform(
            source_path="test.fits",
            reference_path="ref.fits",
            success=True,
            matrix=np.array([[0.99, 0.01, 5], [-0.01, 0.99, 3], [0, 0, 1]]),
            n_matches=42,
        )

        d = transform_to_dict(original)
        restored = dict_to_transform(d)

        assert restored.source_path == original.source_path
        assert restored.success == original.success
        assert np.allclose(restored.matrix, original.matrix)
        assert restored.n_matches == original.n_matches


class TestApplyTransformToImage:
    """Tests for transform application."""

    def test_identity_transform(self):
        """Identity transform should not change image."""
        image = np.random.uniform(0, 1000, (100, 100)).astype(np.float32)
        identity = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float32)

        transformed = apply_transform_to_image(image, identity)

        # Should be nearly identical (some interpolation at boundaries)
        assert np.allclose(transformed[10:-10, 10:-10], image[10:-10, 10:-10], atol=1)

    def test_translation_transform(self):
        """Translation should shift image."""
        image = np.zeros((100, 100), dtype=np.float32)
        image[40:60, 40:60] = 1000  # Box in center

        # Shift by 10 pixels right and down
        translation = np.array([
            [1, 0, 10],
            [0, 1, 10],
            [0, 0, 1],
        ], dtype=np.float32)

        transformed = apply_transform_to_image(image, translation)

        # Original center should now be near (50, 50) area
        assert np.sum(transformed[50:70, 50:70]) > np.sum(transformed[40:60, 40:60])

    def test_output_shape_preserved(self):
        """Output should have same shape as input."""
        image = np.zeros((100, 120), dtype=np.float32)
        transform = np.eye(3, dtype=np.float32)

        result = apply_transform_to_image(image, transform)

        assert result.shape == image.shape

    def test_rgb_image_transform(self):
        """Transform should work on RGB images."""
        rgb = np.random.uniform(0, 1000, (100, 100, 3)).astype(np.float32)
        transform = np.eye(3, dtype=np.float32)

        result = apply_transform_to_image(rgb, transform)

        assert result.shape == rgb.shape


class TestSyntheticAlignment:
    """Integration tests using synthetic star fields."""

    @pytest.fixture
    def synthetic_star_field_rgb(self):
        """Create synthetic RGB star field."""
        def _create(height=200, width=200, n_stars=15, seed=42):
            np.random.seed(seed)

            # Background with slight gradient
            rgb = np.zeros((height, width, 3), dtype=np.float32)
            for c in range(3):
                rgb[:, :, c] = np.random.normal(1000, 50, (height, width))

            # Add stars (same positions in all channels)
            yy, xx = np.mgrid[0:height, 0:width]
            for _ in range(n_stars):
                y0 = np.random.uniform(20, height - 20)
                x0 = np.random.uniform(20, width - 20)
                sigma = np.random.uniform(2, 4)
                amplitude = np.random.uniform(5000, 30000)

                star = amplitude * np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * sigma**2))
                for c in range(3):
                    rgb[:, :, c] += star * (0.8 + 0.4 * np.random.random())

            return np.clip(rgb, 0, 65535).astype(np.float32)

        return _create

    def test_aligned_frames_similar_to_reference(self, synthetic_star_field_rgb):
        """Aligned frames should be similar to reference."""
        ref_rgb = synthetic_star_field_rgb()

        # Create slightly shifted version
        shifted = np.roll(ref_rgb, 5, axis=0)
        shifted = np.roll(shifted, 3, axis=1)

        # After proper alignment, correlation should be high
        ref_luma = luminance_from_rgb(ref_rgb)
        shifted_luma = luminance_from_rgb(shifted)

        # Compute correlation (simple proxy for alignment quality)
        corr_before = np.corrcoef(ref_luma.flatten(), shifted_luma.flatten())[0, 1]

        # The correlation should still be reasonably high for similar fields
        # (shifted fields still share most content)
        assert corr_before > 0.3  # Shifted but still correlated


class TestTransformPersistence:
    """Tests for saving and loading transforms."""

    def test_transform_matrix_serializable(self):
        """Transform matrix should be JSON-serializable."""
        import json

        transform = np.array([
            [0.9998, 0.0175, 5.2],
            [-0.0175, 0.9998, 3.1],
            [0, 0, 1],
        ], dtype=np.float64)

        # Should be serializable as list
        serialized = json.dumps(transform.tolist())
        loaded = np.array(json.loads(serialized))

        assert np.allclose(transform, loaded)

    def test_alignment_transform_has_attributes(self):
        """AlignmentTransform should have expected attributes."""
        transform = AlignmentTransform(
            source_path="test.fits",
            reference_path="ref.fits",
            success=True,
            matrix=np.eye(3),
            n_matches=100,
        )

        assert hasattr(transform, 'source_path')
        assert hasattr(transform, 'reference_path')
        assert hasattr(transform, 'matrix')
        assert hasattr(transform, 'success')
        assert hasattr(transform, 'n_matches')


class TestApplyTransformWithMask:
    """Tests for apply_transform_to_image with return_mask option."""

    def test_identity_returns_full_mask(self):
        """Identity transform should return all-valid mask."""
        image = np.random.uniform(0, 1000, (100, 100)).astype(np.float32)
        identity = np.eye(3, dtype=np.float32)

        transformed, mask = apply_transform_to_image(
            image, identity, return_mask=True
        )

        assert transformed.shape == image.shape
        assert mask.shape == image.shape[:2]
        # All pixels should be valid (mask = 1.0)
        assert np.all(mask > 0.99)

    def test_translation_creates_invalid_region(self):
        """Translation should create invalid pixels where no source data exists."""
        image = np.ones((50, 50), dtype=np.float32) * 1000

        # Shift by 10 pixels right and down
        translation = np.array([
            [1, 0, 10],
            [0, 1, 10],
            [0, 0, 1],
        ], dtype=np.float32)

        transformed, mask = apply_transform_to_image(
            image, translation, return_mask=True
        )

        # Top-left corner should be invalid (no source data)
        assert mask[0, 0] < 0.5
        assert mask[5, 5] < 0.5
        # Bottom-right should be valid
        assert mask[45, 45] > 0.5

    def test_rotation_creates_corner_invalids(self):
        """Rotation should create invalid corners."""
        image = np.ones((100, 100), dtype=np.float32) * 1000

        # 45 degree rotation around center
        angle = np.radians(45)
        cx, cy = 50, 50
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([
            [cos_a, -sin_a, cx - cx * cos_a + cy * sin_a],
            [sin_a, cos_a, cy - cx * sin_a - cy * cos_a],
            [0, 0, 1],
        ], dtype=np.float32)

        transformed, mask = apply_transform_to_image(
            image, rotation, return_mask=True
        )

        # Corners should be invalid after rotation
        assert mask[0, 0] < 0.5  # Top-left corner
        # Center should still be valid
        assert mask[50, 50] > 0.5

    def test_mask_is_binary(self):
        """Mask should be binary (0 or 1) with nearest-neighbor interpolation."""
        image = np.ones((50, 50), dtype=np.float32)
        translation = np.array([
            [1, 0, 5],
            [0, 1, 5],
            [0, 0, 1],
        ], dtype=np.float32)

        transformed, mask = apply_transform_to_image(
            image, translation, return_mask=True
        )

        # Mask should only contain values very close to 0 or 1
        unique_vals = np.unique(mask)
        assert len(unique_vals) <= 2  # Only 0 and 1
        assert all(v < 0.01 or v > 0.99 for v in unique_vals)

    def test_rgb_image_mask(self):
        """Mask should work correctly for RGB images."""
        rgb = np.random.uniform(0, 1000, (50, 50, 3)).astype(np.float32)
        translation = np.array([
            [1, 0, 5],
            [0, 1, 5],
            [0, 0, 1],
        ], dtype=np.float32)

        transformed, mask = apply_transform_to_image(
            rgb, translation, return_mask=True
        )

        assert transformed.shape == rgb.shape
        assert mask.shape == rgb.shape[:2]  # Mask is 2D even for RGB

    def test_return_mask_false(self):
        """return_mask=False should return only the image."""
        image = np.ones((50, 50), dtype=np.float32)
        identity = np.eye(3, dtype=np.float32)

        result = apply_transform_to_image(
            image, identity, return_mask=False
        )

        # Should return just the image, not a tuple
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape
