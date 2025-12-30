"""
Pytest configuration and fixtures.

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def synthetic_bayer_rggb():
    """Create a synthetic RGGB Bayer pattern image."""
    def _create(height=100, width=100, r_value=1000, g_value=2000, b_value=500):
        """
        Create a synthetic RGGB Bayer mosaic.

        RGGB pattern:
            R  G  R  G  ...  (even rows)
            G  B  G  B  ...  (odd rows)
        """
        bayer = np.zeros((height, width), dtype=np.float32)

        # R at (even row, even col)
        bayer[0::2, 0::2] = r_value
        # G at (even row, odd col)
        bayer[0::2, 1::2] = g_value
        # G at (odd row, even col)
        bayer[1::2, 0::2] = g_value
        # B at (odd row, odd col)
        bayer[1::2, 1::2] = b_value

        return bayer

    return _create


@pytest.fixture
def synthetic_rgb():
    """Create a synthetic RGB image."""
    def _create(height=100, width=100, r_value=1000, g_value=2000, b_value=500):
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        rgb[:, :, 0] = r_value
        rgb[:, :, 1] = g_value
        rgb[:, :, 2] = b_value
        return rgb

    return _create


@pytest.fixture
def synthetic_star_field():
    """Create a synthetic star field with Gaussian stars."""
    def _create(height=200, width=200, n_stars=20, background=1000, seed=42):
        np.random.seed(seed)

        # Background
        image = np.full((height, width), background, dtype=np.float32)

        # Add Gaussian noise
        image += np.random.normal(0, 50, (height, width)).astype(np.float32)

        # Add stars (2D Gaussians)
        yy, xx = np.mgrid[0:height, 0:width]
        for _ in range(n_stars):
            y0 = np.random.uniform(10, height - 10)
            x0 = np.random.uniform(10, width - 10)
            sigma = np.random.uniform(2, 5)
            amplitude = np.random.uniform(5000, 50000)

            star = amplitude * np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * sigma**2))
            image += star

        return np.clip(image, 0, 65535).astype(np.float32)

    return _create


@pytest.fixture
def raw_data_path():
    """Path to rawData directory if available."""
    path = Path("/Data/dwarf3/rawData")
    if path.exists():
        return path
    return None


@pytest.fixture
def sample_session_path(raw_data_path):
    """Path to a sample session if available."""
    if raw_data_path is None:
        return None

    sessions = list(raw_data_path.glob("DWARF_RAW_TELE_*"))
    if sessions:
        return sessions[0]
    return None
