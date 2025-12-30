"""
Pseudo-color mapping for luminance images.

Provides honest colorization of grayscale astrophotography images using
scientific colormaps. These are visualization aids, not fake RGB from
broadband data.

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Available palette names
PALETTE_NAMES = Literal[
    "grayscale",
    "viridis",
    "inferno",
    "plasma",
    "magma",
    "cividis",
    "cool",
    "warm",
    "hot",
    "copper",
    "bone",
    "blue_gold",
    "h_alpha",
    "oiii",
]


def _linear_interpolate(x: np.ndarray, colors: list[tuple[float, float, float]]) -> np.ndarray:
    """
    Linearly interpolate between colors based on input values.

    Parameters
    ----------
    x : np.ndarray
        Input values in [0, 1] range.
    colors : list of tuples
        List of (R, G, B) colors to interpolate between.

    Returns
    -------
    np.ndarray
        RGB array with shape (*x.shape, 3).
    """
    n_colors = len(colors)
    if n_colors < 2:
        raise ValueError("Need at least 2 colors for interpolation")

    x = np.clip(x, 0, 1)

    # Determine which segment each value falls into
    segments = np.minimum((x * (n_colors - 1)).astype(int), n_colors - 2)
    t = x * (n_colors - 1) - segments  # Local parameter within segment

    # Build color arrays
    colors_array = np.array(colors)

    # Get colors for each segment
    c0 = colors_array[segments]
    c1 = colors_array[np.minimum(segments + 1, n_colors - 1)]

    # Interpolate
    result = c0 + t[..., np.newaxis] * (c1 - c0)

    return result.astype(np.float32)


# Custom astronomy palettes as RGB tuples (0-1 range)
CUSTOM_PALETTES = {
    "grayscale": [(0, 0, 0), (1, 1, 1)],
    "cool": [
        (0, 0, 0.1),      # Dark blue-black
        (0.1, 0.1, 0.4),  # Deep blue
        (0.2, 0.4, 0.6),  # Cool blue
        (0.4, 0.6, 0.8),  # Light blue
        (0.7, 0.85, 1.0), # Very light blue
        (1.0, 1.0, 1.0),  # White
    ],
    "warm": [
        (0.05, 0, 0),     # Very dark red
        (0.3, 0.05, 0),   # Dark red-brown
        (0.6, 0.15, 0),   # Orange-red
        (0.8, 0.4, 0.1),  # Orange
        (1.0, 0.7, 0.3),  # Yellow-orange
        (1.0, 0.95, 0.8), # Cream
    ],
    "blue_gold": [
        (0, 0, 0.15),     # Dark blue
        (0.1, 0.15, 0.4), # Deep blue
        (0.4, 0.3, 0.2),  # Transition
        (0.7, 0.5, 0.1),  # Gold
        (1.0, 0.85, 0.4), # Bright gold
        (1.0, 1.0, 0.9),  # Cream
    ],
    "h_alpha": [
        (0, 0, 0),        # Black
        (0.2, 0, 0.05),   # Very dark red
        (0.5, 0.05, 0.1), # Deep red
        (0.8, 0.15, 0.2), # Red
        (1.0, 0.4, 0.4),  # Light red
        (1.0, 0.8, 0.8),  # Pink-white
    ],
    "oiii": [
        (0, 0, 0),        # Black
        (0, 0.15, 0.1),   # Very dark teal
        (0, 0.35, 0.25),  # Dark teal
        (0.1, 0.6, 0.5),  # Teal
        (0.4, 0.85, 0.7), # Light teal
        (0.8, 1.0, 0.95), # Cyan-white
    ],
}


def get_matplotlib_colormap(name: str) -> np.ndarray:
    """
    Get a matplotlib colormap as an RGB lookup table.

    Parameters
    ----------
    name : str
        Name of the matplotlib colormap.

    Returns
    -------
    np.ndarray
        256x3 RGB lookup table.
    """
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(name)
        lut = cmap(np.linspace(0, 1, 256))[:, :3]  # RGB only, no alpha
        return lut.astype(np.float32)
    except ImportError:
        raise ImportError("matplotlib required for standard colormaps")


def apply_colormap(
    data: np.ndarray,
    palette: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    percentile_clip: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Apply a colormap to a luminance image.

    Parameters
    ----------
    data : np.ndarray
        2D luminance image with values typically in [0, 1].
    palette : str, default "viridis"
        Colormap name. Can be:
        - Standard matplotlib: "viridis", "inferno", "plasma", "magma", "cividis",
          "hot", "copper", "bone"
        - Custom astronomy: "grayscale", "cool", "warm", "blue_gold", "h_alpha", "oiii"
    vmin : float, optional
        Minimum value for normalization. If None, uses data.min().
    vmax : float, optional
        Maximum value for normalization. If None, uses data.max().
    percentile_clip : tuple of floats, optional
        If provided, use these percentiles instead of vmin/vmax.
        Example: (1, 99) clips to 1st and 99th percentiles.

    Returns
    -------
    np.ndarray
        RGB image with shape (H, W, 3), values in [0, 1].

    Notes
    -----
    This provides honest colorization - the color represents intensity
    in a single broadband channel, not reconstructed RGB. Labels should
    indicate "pseudo-color" or "false color" when publishing.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    # Normalize data
    if percentile_clip is not None:
        vmin = np.percentile(data, percentile_clip[0])
        vmax = np.percentile(data, percentile_clip[1])
    else:
        if vmin is None:
            vmin = float(data.min())
        if vmax is None:
            vmax = float(data.max())

    # Avoid division by zero
    if vmax - vmin < 1e-10:
        normalized = np.zeros_like(data)
    else:
        normalized = (data - vmin) / (vmax - vmin)

    normalized = np.clip(normalized, 0, 1)

    # Apply colormap
    if palette in CUSTOM_PALETTES:
        # Use custom palette
        rgb = _linear_interpolate(normalized, CUSTOM_PALETTES[palette])
        logger.debug("Applied custom palette: %s", palette)
    else:
        # Use matplotlib colormap
        try:
            lut = get_matplotlib_colormap(palette)
            # Index into LUT
            indices = (normalized * 255).astype(np.uint8)
            rgb = lut[indices]
            logger.debug("Applied matplotlib colormap: %s", palette)
        except ImportError:
            logger.warning(
                "matplotlib not available, falling back to grayscale for palette %s",
                palette,
            )
            rgb = _linear_interpolate(normalized, CUSTOM_PALETTES["grayscale"])

    return rgb.astype(np.float32)


def list_palettes() -> list[str]:
    """
    List all available palette names.

    Returns
    -------
    list of str
        Available palette names.
    """
    custom = list(CUSTOM_PALETTES.keys())
    try:
        import matplotlib.pyplot as plt
        matplotlib_palettes = [
            "viridis", "inferno", "plasma", "magma", "cividis",
            "hot", "copper", "bone", "twilight", "turbo",
        ]
    except ImportError:
        matplotlib_palettes = []

    return sorted(set(custom + matplotlib_palettes))


def create_colorbar(
    palette: str,
    height: int = 20,
    width: int = 256,
    horizontal: bool = True,
) -> np.ndarray:
    """
    Create a colorbar image for a palette.

    Parameters
    ----------
    palette : str
        Colormap name.
    height : int, default 20
        Height of the colorbar in pixels.
    width : int, default 256
        Width of the colorbar in pixels.
    horizontal : bool, default True
        If True, gradient runs left-to-right. If False, bottom-to-top.

    Returns
    -------
    np.ndarray
        RGB colorbar image with shape (height, width, 3) or (width, height, 3).
    """
    if horizontal:
        gradient = np.linspace(0, 1, width)[np.newaxis, :]
        gradient = np.repeat(gradient, height, axis=0)
    else:
        gradient = np.linspace(0, 1, height)[::-1, np.newaxis]
        gradient = np.repeat(gradient, width, axis=1)

    return apply_colormap(gradient, palette=palette)


def blend_with_luminance(
    colored: np.ndarray,
    luminance: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """
    Blend a colored image with its luminance for a more subtle effect.

    Parameters
    ----------
    colored : np.ndarray
        RGB image from colormap application.
    luminance : np.ndarray
        Original 2D luminance image (normalized to [0, 1]).
    strength : float, default 0.5
        Blend strength. 0 = pure grayscale, 1 = full color.

    Returns
    -------
    np.ndarray
        Blended RGB image.
    """
    # Ensure luminance is broadcastable
    if luminance.ndim == 2:
        lum_rgb = np.stack([luminance, luminance, luminance], axis=-1)
    else:
        lum_rgb = luminance

    # Blend
    blended = (1 - strength) * lum_rgb + strength * colored
    return np.clip(blended, 0, 1).astype(np.float32)


def apply_bicolor(
    data: np.ndarray,
    color1: tuple[float, float, float] = (0.8, 0.2, 0.2),  # Red
    color2: tuple[float, float, float] = (0.2, 0.6, 1.0),  # Teal
    threshold: float = 0.5,
    softness: float = 0.2,
) -> np.ndarray:
    """
    Apply a bicolor scheme (common in narrowband imaging).

    Maps values below threshold to color1, above to color2, with a
    smooth transition in between.

    Parameters
    ----------
    data : np.ndarray
        2D luminance image, normalized to [0, 1].
    color1 : tuple, default (0.8, 0.2, 0.2)
        RGB color for low values.
    color2 : tuple, default (0.2, 0.6, 1.0)
        RGB color for high values.
    threshold : float, default 0.5
        Crossover point between colors.
    softness : float, default 0.2
        Width of the transition zone.

    Returns
    -------
    np.ndarray
        RGB image with shape (H, W, 3).

    Notes
    -----
    This is useful for simulating narrowband imaging results
    (e.g., H-alpha + OIII) from broadband data, but should be
    clearly labeled as synthetic.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    data = np.clip(data, 0, 1)

    # Create transition mask
    if softness > 0:
        mask = (data - threshold) / (softness * 2) + 0.5
        mask = np.clip(mask, 0, 1)
    else:
        mask = (data > threshold).astype(np.float32)

    # Apply colors
    c1 = np.array(color1)
    c2 = np.array(color2)

    # Base brightness from original data
    brightness = data[..., np.newaxis]

    # Mix colors based on mask
    color_mix = (1 - mask[..., np.newaxis]) * c1 + mask[..., np.newaxis] * c2

    # Apply brightness
    result = brightness * color_mix

    return np.clip(result, 0, 1).astype(np.float32)
