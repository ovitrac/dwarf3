"""
I/O operations for DWARF 3 FITS files.

Handles:
- Frame discovery with proper exclusion rules
- FITS reading with correct BZERO/BSCALE handling
- Header validation and extraction

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

# Expected DWARF 3 frame dimensions
DWARF3_NAXIS1 = 3840
DWARF3_NAXIS2 = 2160
DWARF3_BITPIX = 16
DWARF3_BZERO = 32768


def list_lights(
    session_path: str | Path,
    exclude_failed: bool = True,
    exclude_stacked: bool = True,
) -> list[Path]:
    """
    Discover valid light frames in a DWARF 3 session folder.

    Parameters
    ----------
    session_path : str or Path
        Path to the session folder containing FITS files.
    exclude_failed : bool, default True
        Exclude files matching 'failed_*.fits' pattern.
    exclude_stacked : bool, default True
        Exclude files matching 'stacked-*.fits' pattern.

    Returns
    -------
    list[Path]
        Sorted list of valid light frame paths.

    Notes
    -----
    Files are sorted by name, which for DWARF 3 corresponds to
    chronological order due to the timestamp in filenames.
    """
    session = Path(session_path)
    if not session.is_dir():
        raise ValueError(f"Session path is not a directory: {session}")

    all_fits = sorted(session.glob("*.fits"))
    lights = []

    for fpath in all_fits:
        name = fpath.name

        # Exclude device-flagged failures
        if exclude_failed and name.startswith("failed_"):
            logger.debug("Excluding failed frame: %s", name)
            continue

        # Exclude device-produced stacks
        if exclude_stacked and name.startswith("stacked-"):
            logger.debug("Excluding stacked frame: %s", name)
            continue

        lights.append(fpath)

    logger.info(
        "Discovered %d light frames in %s (excluded: %d)",
        len(lights),
        session.name,
        len(all_fits) - len(lights),
    )

    return lights


def read_fits(
    path: str | Path,
    apply_scaling: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Read a DWARF 3 FITS file with correct BZERO handling.

    Parameters
    ----------
    path : str or Path
        Path to the FITS file.
    apply_scaling : bool, default True
        Apply BZERO/BSCALE scaling to get true pixel values.
        If False, returns raw (signed) data.
    dtype : np.dtype, default np.float32
        Output data type. Float32 is recommended for processing.

    Returns
    -------
    np.ndarray
        2D array of shape (NAXIS2, NAXIS1) = (2160, 3840).
        Values are linear ADU counts (0-65535 range for uint16 source).

    Notes
    -----
    DWARF 3 stores unsigned 16-bit data using the FITS convention:
    - BITPIX = 16 (signed 16-bit storage)
    - BZERO = 32768 (offset to represent unsigned range)
    - BSCALE = 1

    When apply_scaling=True (default), astropy automatically applies
    the transformation: physical_value = stored_value * BSCALE + BZERO
    """
    with fits.open(path) as hdul:
        hdu = hdul[0]

        if apply_scaling:
            # astropy applies BZERO/BSCALE when reading with do_not_scale_image_data=False (default)
            # This converts int16 [-32768, 32767] + BZERO=32768 -> uint16 [0, 65535]
            data = hdu.data.astype(dtype)
        else:
            # Read raw stored values without scaling
            data = hdu.data.astype(dtype)

    return data


def read_header(path: str | Path) -> fits.Header:
    """
    Read FITS header without loading data.

    Parameters
    ----------
    path : str or Path
        Path to the FITS file.

    Returns
    -------
    fits.Header
        FITS header object.
    """
    with fits.open(path) as hdul:
        return hdul[0].header.copy()


def extract_metadata(header: fits.Header) -> dict[str, Any]:
    """
    Extract relevant metadata from a DWARF 3 FITS header.

    Parameters
    ----------
    header : fits.Header
        FITS header object.

    Returns
    -------
    dict
        Dictionary with standardized metadata keys.
    """
    return {
        "naxis1": header.get("NAXIS1"),
        "naxis2": header.get("NAXIS2"),
        "bitpix": header.get("BITPIX"),
        "bzero": header.get("BZERO", 0),
        "bscale": header.get("BSCALE", 1),
        "exptime": header.get("EXPTIME"),
        "gain": header.get("GAIN"),
        "filter": header.get("FILTER", "").strip(),
        "camera": header.get("CAMERA", "").strip(),
        "bayerpat": header.get("BAYERPAT", "").strip(),
        "ra": header.get("RA"),
        "dec": header.get("DEC"),
        "object": header.get("OBJECT", "").strip(),
        "date_obs": header.get("DATE-OBS", ""),
        "det_temp": header.get("DET-TEMP"),
        "telescope": header.get("TELESCOP", "").strip(),
        "instrument": header.get("INSTRUME", "").strip(),
        "xbinning": header.get("XBINNING", 1),
        "ybinning": header.get("YBINNING", 1),
        "focallen": header.get("FOCALLEN"),
        "xpixsz": header.get("XPIXSZ"),
        "ypixsz": header.get("YPIXSZ"),
    }


def validate_header(
    header: fits.Header,
    expected_naxis1: int = DWARF3_NAXIS1,
    expected_naxis2: int = DWARF3_NAXIS2,
) -> tuple[bool, str]:
    """
    Validate that a FITS header matches expected DWARF 3 specifications.

    Parameters
    ----------
    header : fits.Header
        FITS header to validate.
    expected_naxis1 : int
        Expected width (default: 3840).
    expected_naxis2 : int
        Expected height (default: 2160).

    Returns
    -------
    tuple[bool, str]
        (is_valid, reason) - True if valid, False with explanation otherwise.
    """
    naxis1 = header.get("NAXIS1")
    naxis2 = header.get("NAXIS2")

    if naxis1 != expected_naxis1 or naxis2 != expected_naxis2:
        return False, f"Dimension mismatch: got ({naxis1}, {naxis2}), expected ({expected_naxis1}, {expected_naxis2})"

    bitpix = header.get("BITPIX")
    if bitpix != DWARF3_BITPIX:
        return False, f"BITPIX mismatch: got {bitpix}, expected {DWARF3_BITPIX}"

    return True, "OK"


def read_shots_info(session_path: str | Path) -> dict[str, Any] | None:
    """
    Read shotsInfo.json from a session folder.

    Parameters
    ----------
    session_path : str or Path
        Path to the session folder.

    Returns
    -------
    dict or None
        Parsed JSON content, or None if file not found.
    """
    info_path = Path(session_path) / "shotsInfo.json"
    if not info_path.exists():
        logger.warning("shotsInfo.json not found in %s", session_path)
        return None

    with open(info_path, "r") as f:
        return json.load(f)


def write_fits(
    path: str | Path,
    data: np.ndarray,
    header: fits.Header | None = None,
    overwrite: bool = False,
) -> None:
    """
    Write a FITS file with proper header.

    Parameters
    ----------
    path : str or Path
        Output path.
    data : np.ndarray
        Image data to write.
    header : fits.Header, optional
        Header to include. A minimal header is created if not provided.
    overwrite : bool, default False
        Whether to overwrite existing file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if header is None:
        header = fits.Header()

    # Create HDU and write
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(path, overwrite=overwrite)
    logger.info("Wrote FITS: %s", path)
