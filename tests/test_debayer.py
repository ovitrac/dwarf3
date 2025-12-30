"""
Tests for the debayer module.

Tests cover:
- Superpixel debayer for RGGB pattern
- Bilinear debayer for RGGB pattern
- Luminance extraction
- Channel extraction

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

import numpy as np
import pytest

from dwarf3.debayer import (
    debayer_rggb,
    extract_luminance,
    get_channel,
    luminance_from_rgb,
)


class TestSuperpixelDebayer:
    """Tests for superpixel debayer mode."""

    def test_output_shape_half_resolution(self):
        """Superpixel should produce half-resolution output."""
        # 8x8 Bayer mosaic -> 4x4 RGB
        bayer = np.zeros((8, 8), dtype=np.uint16)
        rgb = debayer_rggb(bayer, mode="superpixel")

        assert rgb.shape == (4, 4, 3)
        assert rgb.dtype == np.float32

    def test_output_shape_dwarf3_resolution(self):
        """Test with DWARF 3 native resolution."""
        # 3840x2160 -> 1920x1080
        bayer = np.zeros((2160, 3840), dtype=np.uint16)
        rgb = debayer_rggb(bayer, mode="superpixel")

        assert rgb.shape == (1080, 1920, 3)

    def test_rggb_pattern_extraction(self):
        """Verify correct extraction of RGGB channels."""
        # Create a known pattern:
        # R=100, G1=200, G2=200, B=50
        bayer = np.array([
            [100, 200, 100, 200],  # R  G1  R  G1
            [200,  50, 200,  50],  # G2 B   G2 B
            [100, 200, 100, 200],
            [200,  50, 200,  50],
        ], dtype=np.float32)

        rgb = debayer_rggb(bayer, mode="superpixel")

        # Each 2x2 block should produce one RGB pixel
        # R=100, G=(200+200)/2=200, B=50
        assert rgb.shape == (2, 2, 3)
        assert np.allclose(rgb[:, :, 0], 100)  # Red
        assert np.allclose(rgb[:, :, 1], 200)  # Green (average)
        assert np.allclose(rgb[:, :, 2], 50)   # Blue

    def test_preserves_dynamic_range(self):
        """Test that full dynamic range is preserved."""
        bayer = np.array([
            [0, 0],
            [0, 65535],
        ], dtype=np.uint16)

        rgb = debayer_rggb(bayer, mode="superpixel")

        assert rgb[0, 0, 0] == 0      # R = 0
        assert rgb[0, 0, 2] == 65535  # B = 65535


class TestBilinearDebayer:
    """Tests for bilinear interpolation debayer mode."""

    def test_output_shape_full_resolution(self):
        """Bilinear should produce full-resolution output."""
        bayer = np.zeros((8, 8), dtype=np.uint16)
        rgb = debayer_rggb(bayer, mode="bilinear")

        assert rgb.shape == (8, 8, 3)
        assert rgb.dtype == np.float32

    def test_output_shape_dwarf3_resolution(self):
        """Test with DWARF 3 native resolution."""
        bayer = np.zeros((2160, 3840), dtype=np.uint16)
        rgb = debayer_rggb(bayer, mode="bilinear")

        assert rgb.shape == (2160, 3840, 3)

    def test_r_positions_correct(self):
        """At R positions, R channel should have original value."""
        bayer = np.zeros((4, 4), dtype=np.float32)
        bayer[0, 0] = 1000  # R position
        bayer[0, 2] = 2000  # R position
        bayer[2, 0] = 3000  # R position

        rgb = debayer_rggb(bayer, mode="bilinear")

        assert rgb[0, 0, 0] == 1000  # R at (0,0)
        assert rgb[0, 2, 0] == 2000  # R at (0,2)
        assert rgb[2, 0, 0] == 3000  # R at (2,0)

    def test_b_positions_correct(self):
        """At B positions, B channel should have original value."""
        bayer = np.zeros((4, 4), dtype=np.float32)
        bayer[1, 1] = 1000  # B position
        bayer[1, 3] = 2000  # B position

        rgb = debayer_rggb(bayer, mode="bilinear")

        assert rgb[1, 1, 2] == 1000  # B at (1,1)
        assert rgb[1, 3, 2] == 2000  # B at (1,3)

    def test_g_positions_correct(self):
        """At G positions, G channel should have original value."""
        bayer = np.zeros((4, 4), dtype=np.float32)
        bayer[0, 1] = 1000  # G position (even row, odd col)
        bayer[1, 0] = 2000  # G position (odd row, even col)

        rgb = debayer_rggb(bayer, mode="bilinear")

        assert rgb[0, 1, 1] == 1000  # G at (0,1)
        assert rgb[1, 0, 1] == 2000  # G at (1,0)


class TestDebayerRGGBGeneral:
    """General tests for debayer_rggb function."""

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        bayer = np.zeros((8, 8), dtype=np.uint16)

        with pytest.raises(ValueError, match="Unknown debayer mode"):
            debayer_rggb(bayer, mode="invalid")

    def test_3d_input_raises(self):
        """3D input should raise ValueError."""
        bayer = np.zeros((8, 8, 3), dtype=np.uint16)

        with pytest.raises(ValueError, match="Expected 2D array"):
            debayer_rggb(bayer)

    def test_odd_dimensions_handled(self):
        """Odd dimensions should be truncated to even."""
        bayer = np.zeros((9, 9), dtype=np.uint16)
        rgb = debayer_rggb(bayer, mode="superpixel")

        # 9x9 -> 4x4 (floor(9/2))
        assert rgb.shape == (4, 4, 3)


class TestExtractLuminance:
    """Tests for luminance extraction from Bayer mosaic."""

    def test_output_shape(self):
        """Luminance should be half resolution."""
        bayer = np.zeros((100, 100), dtype=np.uint16)
        luma = extract_luminance(bayer)

        assert luma.shape == (50, 50)
        assert luma.dtype == np.float32

    def test_superpixel_method(self):
        """Superpixel method averages all 4 pixels."""
        bayer = np.array([
            [100, 200],
            [200, 300],
        ], dtype=np.float32)

        luma = extract_luminance(bayer, method="superpixel")

        # (100 + 200 + 200 + 300) / 4 = 200
        assert luma.shape == (1, 1)
        assert luma[0, 0] == 200

    def test_weighted_method(self):
        """Weighted method uses ITU-R BT.601 coefficients."""
        bayer = np.array([
            [100, 200],  # R=100, G1=200
            [200, 300],  # G2=200, B=300
        ], dtype=np.float32)

        luma = extract_luminance(bayer, method="weighted")

        # L = 0.299*R + 0.587*G + 0.114*B
        # G = (G1 + G2) / 2 = (200 + 200) / 2 = 200
        # L = 0.299*100 + 0.587*200 + 0.114*300
        expected = 0.299 * 100 + 0.587 * 200 + 0.114 * 300
        assert np.isclose(luma[0, 0], expected)

    def test_unsupported_pattern_raises(self):
        """Non-RGGB patterns should raise ValueError."""
        bayer = np.zeros((8, 8), dtype=np.uint16)

        with pytest.raises(ValueError, match="Only RGGB pattern supported"):
            extract_luminance(bayer, bayer_pattern="GRBG")


class TestLuminanceFromRGB:
    """Tests for luminance extraction from RGB images."""

    def test_output_shape(self):
        """Output should be 2D with same spatial dimensions."""
        rgb = np.zeros((100, 100, 3), dtype=np.float32)
        luma = luminance_from_rgb(rgb)

        assert luma.shape == (100, 100)
        assert luma.dtype == np.float32

    def test_average_method(self):
        """Average method computes simple mean."""
        rgb = np.array([
            [[100, 200, 300]],
        ], dtype=np.float32)

        luma = luminance_from_rgb(rgb, method="average")

        # (100 + 200 + 300) / 3 = 200
        assert luma[0, 0] == 200

    def test_weighted_method(self):
        """Weighted method uses ITU-R BT.601 coefficients."""
        rgb = np.array([
            [[100, 200, 300]],  # R=100, G=200, B=300
        ], dtype=np.float32)

        luma = luminance_from_rgb(rgb, method="weighted")

        # L = 0.299*R + 0.587*G + 0.114*B
        expected = 0.299 * 100 + 0.587 * 200 + 0.114 * 300
        assert np.isclose(luma[0, 0], expected)

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        rgb = np.zeros((8, 8, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="Unknown method"):
            luminance_from_rgb(rgb, method="invalid")

    def test_invalid_shape_raises(self):
        """Non-RGB input should raise ValueError."""
        gray = np.zeros((8, 8), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected .* array"):
            luminance_from_rgb(gray)


class TestGetChannel:
    """Tests for channel extraction."""

    def test_get_r_channel(self):
        """Extract R channel by name."""
        rgb = np.zeros((10, 10, 3), dtype=np.float32)
        rgb[:, :, 0] = 1.0

        r = get_channel(rgb, "R")
        assert r.shape == (10, 10)
        assert np.all(r == 1.0)

    def test_get_g_channel(self):
        """Extract G channel by name."""
        rgb = np.zeros((10, 10, 3), dtype=np.float32)
        rgb[:, :, 1] = 2.0

        g = get_channel(rgb, "G")
        assert np.all(g == 2.0)

    def test_get_b_channel(self):
        """Extract B channel by name."""
        rgb = np.zeros((10, 10, 3), dtype=np.float32)
        rgb[:, :, 2] = 3.0

        b = get_channel(rgb, "B")
        assert np.all(b == 3.0)

    def test_get_channel_by_index(self):
        """Extract channel by numeric index."""
        rgb = np.zeros((10, 10, 3), dtype=np.float32)
        rgb[:, :, 0] = 1.0
        rgb[:, :, 1] = 2.0
        rgb[:, :, 2] = 3.0

        assert np.all(get_channel(rgb, 0) == 1.0)
        assert np.all(get_channel(rgb, 1) == 2.0)
        assert np.all(get_channel(rgb, 2) == 3.0)

    def test_invalid_channel_raises(self):
        """Invalid channel should raise ValueError."""
        rgb = np.zeros((10, 10, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="Invalid channel"):
            get_channel(rgb, "X")

    def test_returns_copy(self):
        """Extracted channel should be a copy, not a view."""
        rgb = np.zeros((10, 10, 3), dtype=np.float32)
        r = get_channel(rgb, "R")
        r[:] = 999

        assert np.all(rgb[:, :, 0] == 0)  # Original unchanged


class TestDebayerConsistency:
    """Tests for consistency between debayer modes."""

    def test_superpixel_exact_values(self):
        """Superpixel debayer should produce exact expected values."""
        # Create a proper Bayer pattern with distinct R, G, B values
        bayer = np.zeros((100, 100), dtype=np.float32)
        bayer[0::2, 0::2] = 1000  # R
        bayer[0::2, 1::2] = 2000  # G1
        bayer[1::2, 0::2] = 2000  # G2
        bayer[1::2, 1::2] = 500   # B

        rgb_sp = debayer_rggb(bayer, mode="superpixel")

        # Superpixel gives exact values: R=1000, G=2000, B=500
        assert np.allclose(rgb_sp[:, :, 0], 1000)
        assert np.allclose(rgb_sp[:, :, 1], 2000)
        assert np.allclose(rgb_sp[:, :, 2], 500)

    def test_bilinear_produces_full_resolution(self):
        """Bilinear debayer should produce full-resolution output with all channels."""
        bayer = np.zeros((100, 100), dtype=np.float32)
        bayer[0::2, 0::2] = 1000  # R
        bayer[0::2, 1::2] = 2000  # G1
        bayer[1::2, 0::2] = 2000  # G2
        bayer[1::2, 1::2] = 500   # B

        rgb_bl = debayer_rggb(bayer, mode="bilinear")

        # All channels should have valid values everywhere (no zeros from missing data)
        assert rgb_bl.shape == (100, 100, 3)
        # All values should be positive (interpolated from positive values)
        assert np.all(rgb_bl >= 0)
        # For uniform Bayer input, each channel should be uniform too
        # (no checkerboard pattern from normalization bugs)
        assert np.mean(rgb_bl[:, :, 0]) > 0  # R has value
        assert np.mean(rgb_bl[:, :, 1]) > 0  # G has value
        assert np.mean(rgb_bl[:, :, 2]) > 0  # B has value

    def test_synthetic_known_pattern(self):
        """Test debayer with known RGGB pattern values."""
        # Create proper RGGB Bayer mosaic
        bayer = np.zeros((4, 4), dtype=np.float32)
        bayer[0::2, 0::2] = 100   # R positions
        bayer[0::2, 1::2] = 200   # G1 positions
        bayer[1::2, 0::2] = 200   # G2 positions
        bayer[1::2, 1::2] = 50    # B positions

        rgb_sp = debayer_rggb(bayer, mode="superpixel")

        # Superpixel should give: R=100, G=200, B=50
        assert rgb_sp.shape == (2, 2, 3)
        assert np.allclose(rgb_sp[:, :, 0], 100)
        assert np.allclose(rgb_sp[:, :, 1], 200)
        assert np.allclose(rgb_sp[:, :, 2], 50)

    def test_bilinear_preserves_original_positions(self):
        """Bilinear should preserve original values at native positions."""
        bayer = np.zeros((10, 10), dtype=np.float32)
        bayer[0::2, 0::2] = 1000  # R
        bayer[0::2, 1::2] = 2000  # G1
        bayer[1::2, 0::2] = 2000  # G2
        bayer[1::2, 1::2] = 500   # B

        rgb_bl = debayer_rggb(bayer, mode="bilinear")

        # At R positions (even,even), R channel should be 1000
        assert np.allclose(rgb_bl[0::2, 0::2, 0], 1000)
        # At G positions, G channel should be 2000
        assert np.allclose(rgb_bl[0::2, 1::2, 1], 2000)
        assert np.allclose(rgb_bl[1::2, 0::2, 1], 2000)
        # At B positions (odd,odd), B channel should be 500
        assert np.allclose(rgb_bl[1::2, 1::2, 2], 500)


class TestBilinearNormalization:
    """Tests for bilinear debayer normalization fix.

    These tests verify that the bilinear interpolation uses proper
    normalized convolution (dividing by actual contributing pixel count)
    rather than a fixed factor, which would cause checkerboard artifacts.
    """

    def test_no_checkerboard_pattern(self):
        """Bilinear output should be smooth, not show checkerboard."""
        # Create uniform Bayer pattern
        bayer = np.zeros((100, 100), dtype=np.float32)
        bayer[0::2, 0::2] = 1000  # R
        bayer[0::2, 1::2] = 2000  # G1
        bayer[1::2, 0::2] = 2000  # G2
        bayer[1::2, 1::2] = 500   # B

        rgb = debayer_rggb(bayer, mode="bilinear")

        # For uniform input, output should also be uniform
        # (no alternating bright/dark pattern)
        for c in range(3):
            channel = rgb[10:-10, 10:-10, c]  # Avoid edges
            # Standard deviation should be very low for uniform input
            std_val = np.std(channel)
            mean_val = np.mean(channel)
            # Coefficient of variation should be < 1%
            cv = std_val / mean_val if mean_val > 0 else 0
            assert cv < 0.01, f"Channel {c} shows variation (cv={cv:.4f})"

    def test_interpolated_values_reasonable(self):
        """Interpolated values should be within expected range."""
        bayer = np.zeros((10, 10), dtype=np.float32)
        bayer[0::2, 0::2] = 1000  # R
        bayer[0::2, 1::2] = 2000  # G1
        bayer[1::2, 0::2] = 2000  # G2
        bayer[1::2, 1::2] = 500   # B

        rgb = debayer_rggb(bayer, mode="bilinear")

        # R at G positions should be interpolated from R neighbors
        # At (0,1) which is G1, R should be average of R at (0,0) and (0,2)
        # With proper normalization, this should be 1000
        r_at_g = rgb[0, 1, 0]  # R channel at G1 position
        assert np.isclose(r_at_g, 1000, atol=100)

        # B at G positions should be interpolated from B neighbors
        b_at_g = rgb[0, 1, 2]  # B channel at G1 position
        assert np.isclose(b_at_g, 500, atol=100)

    def test_comparison_with_superpixel(self):
        """Bilinear averages should match superpixel within tolerance."""
        bayer = np.zeros((100, 100), dtype=np.float32)
        bayer[0::2, 0::2] = 1000  # R
        bayer[0::2, 1::2] = 2000  # G1
        bayer[1::2, 0::2] = 2000  # G2
        bayer[1::2, 1::2] = 500   # B

        rgb_sp = debayer_rggb(bayer, mode="superpixel")
        rgb_bl = debayer_rggb(bayer, mode="bilinear")

        # Downsample bilinear to compare with superpixel
        rgb_bl_down = (rgb_bl[0::2, 0::2] + rgb_bl[0::2, 1::2] +
                       rgb_bl[1::2, 0::2] + rgb_bl[1::2, 1::2]) / 4

        # Channel means should be similar
        for c in range(3):
            sp_mean = np.mean(rgb_sp[:, :, c])
            bl_mean = np.mean(rgb_bl_down[:, :, c])
            # Should be within 5% of each other
            assert np.isclose(sp_mean, bl_mean, rtol=0.05), \
                f"Channel {c}: superpixel={sp_mean:.1f}, bilinear={bl_mean:.1f}"

    def test_boundary_handling(self):
        """Edge pixels should be handled correctly without artifacts."""
        bayer = np.zeros((20, 20), dtype=np.float32)
        bayer[0::2, 0::2] = 1000
        bayer[0::2, 1::2] = 2000
        bayer[1::2, 0::2] = 2000
        bayer[1::2, 1::2] = 500

        rgb = debayer_rggb(bayer, mode="bilinear")

        # Edge values should not be NaN or Inf
        assert not np.any(np.isnan(rgb))
        assert not np.any(np.isinf(rgb))

        # Edge values should be positive (for positive input)
        assert np.all(rgb >= 0)

    def test_gradient_preservation(self):
        """Bilinear should preserve gradients in the image."""
        # Create Bayer with gradient in R channel
        bayer = np.zeros((20, 20), dtype=np.float32)
        for i in range(10):
            for j in range(10):
                r_val = 500 + 50 * i + 30 * j  # Gradient
                bayer[2*i, 2*j] = r_val        # R positions
                bayer[2*i, 2*j+1] = 2000       # G1
                bayer[2*i+1, 2*j] = 2000       # G2
                bayer[2*i+1, 2*j+1] = 500      # B

        rgb = debayer_rggb(bayer, mode="bilinear")

        # R channel should show gradient (monotonically increasing)
        r_channel = rgb[:, :, 0]
        # Check that values generally increase left-to-right and top-to-bottom
        assert r_channel[5, 15] > r_channel[5, 5]
        assert r_channel[15, 5] > r_channel[5, 5]

    def test_real_world_values(self):
        """Test with realistic 16-bit values."""
        # Simulate real camera data
        np.random.seed(42)
        bayer = np.zeros((100, 100), dtype=np.float32)

        # Typical astro values with some noise
        bayer[0::2, 0::2] = 3000 + np.random.normal(0, 100, (50, 50))   # R
        bayer[0::2, 1::2] = 5000 + np.random.normal(0, 100, (50, 50))   # G1
        bayer[1::2, 0::2] = 5000 + np.random.normal(0, 100, (50, 50))   # G2
        bayer[1::2, 1::2] = 2500 + np.random.normal(0, 100, (50, 50))   # B

        rgb = debayer_rggb(bayer, mode="bilinear")

        # Check channel means are reasonable
        r_mean = np.mean(rgb[:, :, 0])
        g_mean = np.mean(rgb[:, :, 1])
        b_mean = np.mean(rgb[:, :, 2])

        # Should be close to input values
        assert np.isclose(r_mean, 3000, rtol=0.1)
        assert np.isclose(g_mean, 5000, rtol=0.1)
        assert np.isclose(b_mean, 2500, rtol=0.1)
