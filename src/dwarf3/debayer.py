"""
Debayering operations for DWARF 3 RGGB Bayer pattern.

Provides multiple debayer algorithms for converting raw Bayer mosaic
images to RGB or luminance.

RGGB pattern layout (standard for DWARF 3):
    Row 0: R  G  R  G  ...
    Row 1: G  B  G  B  ...
    Row 2: R  G  R  G  ...
    ...

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy.ndimage import convolve

logger = logging.getLogger(__name__)


def debayer_rggb(
    data: np.ndarray,
    mode: Literal["superpixel", "bilinear"] = "superpixel",
) -> np.ndarray:
    """
    Convert RGGB Bayer mosaic to RGB.

    Parameters
    ----------
    data : np.ndarray
        2D Bayer mosaic image (H, W).
    mode : {"superpixel", "bilinear"}, default "superpixel"
        Debayer algorithm:
        - "superpixel": 2x2 binning (half resolution, robust, fast)
        - "bilinear": Bilinear interpolation (full resolution)

    Returns
    -------
    np.ndarray
        3D RGB image:
        - superpixel: shape (H/2, W/2, 3)
        - bilinear: shape (H, W, 3)

    Notes
    -----
    RGGB pattern layout:
        R  G
        G  B

    For DWARF 3 (3840x2160), superpixel mode produces 1920x1080 RGB.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    if mode == "superpixel":
        return _debayer_superpixel_rggb(data)
    elif mode == "bilinear":
        return _debayer_bilinear_rggb(data)
    else:
        raise ValueError(f"Unknown debayer mode: {mode}")


def _debayer_superpixel_rggb(data: np.ndarray) -> np.ndarray:
    """
    2x2 superpixel debayer for RGGB pattern.

    Each 2x2 Bayer block produces one RGB pixel:
        R  G1      ->  R = R
        G2 B           G = (G1 + G2) / 2
                       B = B

    This is the most robust method with best SNR but half resolution.

    Parameters
    ----------
    data : np.ndarray
        2D Bayer mosaic image.

    Returns
    -------
    np.ndarray
        RGB image with shape (H/2, W/2, 3), dtype float32.
    """
    h, w = data.shape
    # Ensure even dimensions
    h2, w2 = h // 2, w // 2

    # Convert to float32 for processing
    data_f = data.astype(np.float32)

    # Extract channels from 2x2 blocks
    # RGGB layout: R at (0,0), G at (0,1) and (1,0), B at (1,1)
    r = data_f[0:h2*2:2, 0:w2*2:2]      # Even rows, even cols
    g1 = data_f[0:h2*2:2, 1:w2*2:2]     # Even rows, odd cols
    g2 = data_f[1:h2*2:2, 0:w2*2:2]     # Odd rows, even cols
    b = data_f[1:h2*2:2, 1:w2*2:2]      # Odd rows, odd cols

    # Combine into RGB
    rgb = np.zeros((h2, w2, 3), dtype=np.float32)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = (g1 + g2) * 0.5
    rgb[:, :, 2] = b

    logger.debug("Superpixel debayer: %s -> %s", data.shape, rgb.shape)
    return rgb


def _debayer_bilinear_rggb(data: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation debayer for RGGB pattern.

    Full resolution output using bilinear interpolation of missing
    color values at each pixel position.

    Parameters
    ----------
    data : np.ndarray
        2D Bayer mosaic image.

    Returns
    -------
    np.ndarray
        RGB image with shape (H, W, 3), dtype float32.

    Notes
    -----
    Uses convolution kernels with proper normalization by counting
    actual contributing pixels at each position.
    """
    h, w = data.shape
    data_f = data.astype(np.float32)

    # Initialize output channels
    r = np.zeros((h, w), dtype=np.float32)
    g = np.zeros((h, w), dtype=np.float32)
    b = np.zeros((h, w), dtype=np.float32)

    # Create masks for each color position in RGGB pattern
    # R at even row, even col
    # G at even row, odd col AND odd row, even col
    # B at odd row, odd col
    r_mask = np.zeros((h, w), dtype=bool)
    g_mask = np.zeros((h, w), dtype=bool)
    b_mask = np.zeros((h, w), dtype=bool)

    r_mask[0::2, 0::2] = True
    g_mask[0::2, 1::2] = True
    g_mask[1::2, 0::2] = True
    b_mask[1::2, 1::2] = True

    # Interpolation kernels (unnormalized - we'll normalize by actual counts)
    kernel_cross = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.float32)

    kernel_diag = np.array([
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1],
    ], dtype=np.float32)

    # Extract raw color planes (zeros where color is not present)
    r_raw = np.where(r_mask, data_f, 0)
    g_raw = np.where(g_mask, data_f, 0)
    b_raw = np.where(b_mask, data_f, 0)

    # Create count masks (1 where color is present, 0 elsewhere)
    r_count_mask = r_mask.astype(np.float32)
    g_count_mask = g_mask.astype(np.float32)
    b_count_mask = b_mask.astype(np.float32)

    # Helper function for normalized convolution
    def interp_normalized(data_plane, count_mask, kernel):
        """Convolve and normalize by actual contributing pixel count."""
        sum_vals = convolve(data_plane, kernel, mode='nearest')
        sum_counts = convolve(count_mask, kernel, mode='nearest')
        # Avoid division by zero
        return np.divide(sum_vals, sum_counts, out=np.zeros_like(sum_vals),
                        where=sum_counts > 0)

    # Interpolate R channel
    # R at R positions: direct copy
    r[r_mask] = data_f[r_mask]
    # R at G positions (cross pattern from R neighbors)
    r_interp_cross = interp_normalized(r_raw, r_count_mask, kernel_cross)
    # R at B positions (diagonal from R neighbors)
    r_interp_diag = interp_normalized(r_raw, r_count_mask, kernel_diag)
    r[g_mask] = r_interp_cross[g_mask]
    r[b_mask] = r_interp_diag[b_mask]

    # Interpolate G channel
    # G at G positions: direct copy
    g[g_mask] = data_f[g_mask]
    # G at R and B positions (cross pattern from G neighbors)
    g_interp = interp_normalized(g_raw, g_count_mask, kernel_cross)
    g[r_mask] = g_interp[r_mask]
    g[b_mask] = g_interp[b_mask]

    # Interpolate B channel
    # B at B positions: direct copy
    b[b_mask] = data_f[b_mask]
    # B at G positions (cross pattern from B neighbors)
    b_interp_cross = interp_normalized(b_raw, b_count_mask, kernel_cross)
    # B at R positions (diagonal from B neighbors)
    b_interp_diag = interp_normalized(b_raw, b_count_mask, kernel_diag)
    b[g_mask] = b_interp_cross[g_mask]
    b[r_mask] = b_interp_diag[r_mask]

    # Stack into RGB
    rgb = np.stack([r, g, b], axis=-1)

    logger.debug("Bilinear debayer: %s -> %s", data.shape, rgb.shape)
    return rgb


def extract_luminance(
    data: np.ndarray,
    bayer_pattern: str = "RGGB",
    method: Literal["superpixel", "weighted"] = "superpixel",
) -> np.ndarray:
    """
    Extract luminance from Bayer mosaic without full debayer.

    This is faster than full debayer when only luminance is needed
    (e.g., for registration).

    Parameters
    ----------
    data : np.ndarray
        2D Bayer mosaic image.
    bayer_pattern : str, default "RGGB"
        Bayer pattern (only RGGB supported currently).
    method : {"superpixel", "weighted"}, default "superpixel"
        - "superpixel": Average 2x2 blocks (half resolution)
        - "weighted": ITU-R BT.601 weighted sum (half resolution)
          L = 0.299*R + 0.587*G + 0.114*B

    Returns
    -------
    np.ndarray
        Luminance image with shape (H/2, W/2), dtype float32.
    """
    if bayer_pattern != "RGGB":
        raise ValueError(f"Only RGGB pattern supported, got {bayer_pattern}")

    h, w = data.shape
    h2, w2 = h // 2, w // 2

    data_f = data.astype(np.float32)

    # Extract 2x2 block components
    r = data_f[0:h2*2:2, 0:w2*2:2]
    g1 = data_f[0:h2*2:2, 1:w2*2:2]
    g2 = data_f[1:h2*2:2, 0:w2*2:2]
    b = data_f[1:h2*2:2, 1:w2*2:2]

    if method == "superpixel":
        # Simple average of all 4 pixels
        luma = (r + g1 + g2 + b) * 0.25
    elif method == "weighted":
        # ITU-R BT.601 luminance coefficients
        # L = 0.299*R + 0.587*G + 0.114*B
        # G is average of G1 and G2
        g_avg = (g1 + g2) * 0.5
        luma = 0.299 * r + 0.587 * g_avg + 0.114 * b
    else:
        raise ValueError(f"Unknown luminance method: {method}")

    logger.debug("Luminance extraction (%s): %s -> %s", method, data.shape, luma.shape)
    return luma


def luminance_from_rgb(
    rgb: np.ndarray,
    method: Literal["average", "weighted"] = "weighted",
) -> np.ndarray:
    """
    Convert RGB image to luminance.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image with shape (H, W, 3).
    method : {"average", "weighted"}, default "weighted"
        - "average": Simple mean of R, G, B
        - "weighted": ITU-R BT.601 weighted sum

    Returns
    -------
    np.ndarray
        Luminance image with shape (H, W), dtype float32.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) array, got shape {rgb.shape}")

    if method == "average":
        return np.mean(rgb, axis=2).astype(np.float32)
    elif method == "weighted":
        # ITU-R BT.601
        return (0.299 * rgb[:, :, 0] +
                0.587 * rgb[:, :, 1] +
                0.114 * rgb[:, :, 2]).astype(np.float32)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_channel(rgb: np.ndarray, channel: Literal["R", "G", "B", 0, 1, 2]) -> np.ndarray:
    """
    Extract a single channel from RGB image.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image with shape (H, W, 3).
    channel : {"R", "G", "B"} or {0, 1, 2}
        Channel to extract.

    Returns
    -------
    np.ndarray
        Single channel with shape (H, W).
    """
    channel_map = {"R": 0, "G": 1, "B": 2, 0: 0, 1: 1, 2: 2}
    if channel not in channel_map:
        raise ValueError(f"Invalid channel: {channel}")
    return rgb[:, :, channel_map[channel]].copy()


def extract_bayer_planes(bayer: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the four color planes from an RGGB Bayer mosaic.

    Each plane is half-resolution (H/2, W/2). This is the building block
    for plane-based stacking which avoids double interpolation.

    Parameters
    ----------
    bayer : np.ndarray
        2D raw Bayer mosaic image (H, W) with RGGB pattern.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (R, G1, G2, B) planes, each with shape (H/2, W/2), dtype float32.

    Notes
    -----
    RGGB pattern layout:
        R   G1
        G2  B

    For DWARF 3 (3840×2160), each plane is 1920×1080.

    Example
    -------
    >>> bayer = read_fits("frame.fits")  # 2160 x 3840
    >>> r, g1, g2, b = extract_bayer_planes(bayer)
    >>> # Each plane is 1080 x 1920
    """
    if bayer.ndim != 2:
        raise ValueError(f"Expected 2D Bayer array, got shape {bayer.shape}")

    h, w = bayer.shape
    h2, w2 = h // 2, w // 2

    bayer_f = bayer.astype(np.float32)

    # Extract channels from 2x2 blocks
    # RGGB layout: R at (0,0), G1 at (0,1), G2 at (1,0), B at (1,1)
    r = bayer_f[0:h2*2:2, 0:w2*2:2]      # Even rows, even cols
    g1 = bayer_f[0:h2*2:2, 1:w2*2:2]     # Even rows, odd cols
    g2 = bayer_f[1:h2*2:2, 0:w2*2:2]     # Odd rows, even cols
    b = bayer_f[1:h2*2:2, 1:w2*2:2]      # Odd rows, odd cols

    logger.debug("Bayer plane extraction: %s -> 4x%s", bayer.shape, r.shape)
    return r, g1, g2, b


def rebuild_bayer_from_planes(
    r: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Rebuild an RGGB Bayer mosaic from four color planes.

    This is the inverse of extract_bayer_planes().

    Parameters
    ----------
    r, g1, g2, b : np.ndarray
        Color planes, each with shape (H/2, W/2).

    Returns
    -------
    np.ndarray
        Reconstructed Bayer mosaic with shape (H, W), dtype float32.
    """
    h2, w2 = r.shape
    h, w = h2 * 2, w2 * 2

    bayer = np.zeros((h, w), dtype=np.float32)
    bayer[0::2, 0::2] = r    # R at even rows, even cols
    bayer[0::2, 1::2] = g1   # G1 at even rows, odd cols
    bayer[1::2, 0::2] = g2   # G2 at odd rows, even cols
    bayer[1::2, 1::2] = b    # B at odd rows, odd cols

    logger.debug("Bayer reconstruction: 4x%s -> %s", r.shape, bayer.shape)
    return bayer


def bayer_luma_rggb(bayer: np.ndarray) -> np.ndarray:
    """
    Extract green-based luminance proxy from raw Bayer mosaic.

    This function extracts a half-resolution luminance image using only
    the green pixels (best SNR) from the RGGB Bayer pattern. This is
    specifically designed for alignment: transforms computed on this
    luminance can be applied to the full-resolution Bayer mosaic.

    Parameters
    ----------
    bayer : np.ndarray
        2D raw Bayer mosaic image (H, W) with RGGB pattern.

    Returns
    -------
    np.ndarray
        Luminance image with shape (H/2, W/2), dtype float32.

    Notes
    -----
    RGGB pattern layout:
        R   G1
        G2  B

    This function averages G1 and G2 to produce a half-resolution
    luminance proxy. Using only green pixels provides:
    - Best SNR (2 green pixels per 2x2 block)
    - Consistent geometry for alignment
    - Transforms computed here can be applied to full Bayer

    The key advantage over full superpixel debayer is that this
    maintains geometric consistency: the transform is computed in
    half-resolution space and can be directly applied to Bayer
    (which has the same spatial relationship).

    Example
    -------
    >>> bayer = read_fits("frame.fits")  # 2160 x 3840
    >>> luma = bayer_luma_rggb(bayer)    # 1080 x 1920
    >>> # Compute transform in half-res space
    >>> transform = find_transform(luma, ref_luma)
    >>> # Apply to full Bayer - translation scaled by 2
    >>> aligned_bayer = apply_transform_to_bayer(bayer, transform.matrix)
    """
    if bayer.ndim != 2:
        raise ValueError(f"Expected 2D Bayer array, got shape {bayer.shape}")

    h, w = bayer.shape
    h2, w2 = h // 2, w // 2

    bayer_f = bayer.astype(np.float32)

    # Extract green pixels from RGGB pattern
    # G1 at even rows, odd cols
    # G2 at odd rows, even cols
    g1 = bayer_f[0:h2*2:2, 1:w2*2:2]
    g2 = bayer_f[1:h2*2:2, 0:w2*2:2]

    # Average the two green channels for luminance proxy
    luma = 0.5 * (g1 + g2)

    logger.debug("Bayer luminance extraction: %s -> %s", bayer.shape, luma.shape)
    return luma
