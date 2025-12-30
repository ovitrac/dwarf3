"""
Cache management for dwarf3 stacking pipeline.

Provides persistent caching of intermediate results to enable
resumption at any pipeline stage without recomputation.

Cached artifacts:
- Frame scores (quality assessment)
- Alignment transforms (registration)
- Stacked intermediates (per-channel or mono)
- Calibration parameters (white balance, background)

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .config import FrameScore, StackConfig
from .utils import __version__

logger = logging.getLogger(__name__)


# Cache file names
CACHE_DIR = "cache"
SCORES_FILE = "scores.json"
TRANSFORMS_FILE = "transforms.json"
STACK_MONO_FILE = "stack_mono.fits"
STACK_R_FILE = "stack_r.fits"
STACK_G_FILE = "stack_g.fits"
STACK_B_FILE = "stack_b.fits"
STACK_RGB_FILE = "stack_rgb_linear.fits"
CALIBRATION_FILE = "calibration.json"
MANIFEST_FILE = "cache_manifest.json"
ALIGNED_DIR = "aligned"
ALIGNED_MANIFEST = "aligned_manifest.json"


@dataclass
class CacheManifest:
    """Metadata about cached artifacts."""

    version: str = ""
    created: str = ""
    session_id: str = ""
    config_hash: str = ""

    # What is cached
    has_scores: bool = False
    has_transforms: bool = False
    has_stack_mono: bool = False
    has_stack_rgb: bool = False
    has_calibration: bool = False

    # Frame lists (for validation)
    n_input_frames: int = 0
    n_selected_frames: int = 0
    n_aligned_frames: int = 0

    # Source hash (for invalidation)
    input_frames_hash: str = ""

    # Extra metadata
    metadata: dict = field(default_factory=dict)


def compute_config_hash(config: StackConfig) -> str:
    """
    Compute a hash of configuration that affects caching.

    Parameters
    ----------
    config : StackConfig
        Configuration to hash.

    Returns
    -------
    str
        Short hash string.
    """
    # Only hash parameters that affect cached results
    relevant = {
        "keep_fraction": config.keep_fraction,
        "sigma": config.sigma,
        "maxiters": config.maxiters,
        "reference": config.reference,
        "debayer": config.debayer,
        "backend_align": config.backend_align,
    }
    content = json.dumps(relevant, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()[:12]


def compute_frames_hash(frame_paths: list[Path]) -> str:
    """
    Compute a hash of input frame list (names only, not content).

    Parameters
    ----------
    frame_paths : list[Path]
        List of frame paths.

    Returns
    -------
    str
        Short hash string.
    """
    names = sorted([p.name for p in frame_paths])
    content = "\n".join(names)
    return hashlib.md5(content.encode()).hexdigest()[:12]


class PipelineCache:
    """
    Manages caching of intermediate pipeline results.

    The cache enables resumption at any pipeline stage:
    1. Score frames (skip if scores.json exists)
    2. Align frames (skip if transforms.json exists)
    3. Stack frames (skip if stack_*.fits exists)
    4. Calibrate colors (skip if calibration.json exists)

    Cache invalidation:
    - Config change → invalidate everything after scores
    - Frame list change → invalidate everything
    - Manual: delete cache directory or specific files

    Parameters
    ----------
    cache_dir : Path
        Directory to store cache files.
    session_id : str
        Session identifier for manifest.
    """

    def __init__(self, cache_dir: Path, session_id: str = ""):
        self.cache_dir = Path(cache_dir)
        self.session_id = session_id
        self.manifest: CacheManifest | None = None

        # Create cache directory if needed
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing manifest
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load existing manifest if present."""
        manifest_path = self.cache_dir / MANIFEST_FILE
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    data = json.load(f)
                self.manifest = CacheManifest(**data)
                logger.debug("Loaded cache manifest from %s", manifest_path)
            except Exception as e:
                logger.warning("Failed to load cache manifest: %s", e)
                self.manifest = None

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        if self.manifest is None:
            return

        manifest_path = self.cache_dir / MANIFEST_FILE
        with open(manifest_path, "w") as f:
            json.dump(asdict(self.manifest), f, indent=2)
        logger.debug("Saved cache manifest to %s", manifest_path)

    def init_manifest(
        self,
        config: StackConfig,
        frame_paths: list[Path],
    ) -> CacheManifest:
        """
        Initialize or update cache manifest.

        Parameters
        ----------
        config : StackConfig
            Current configuration.
        frame_paths : list[Path]
            Input frame list.

        Returns
        -------
        CacheManifest
            The manifest (new or validated existing).
        """
        config_hash = compute_config_hash(config)
        frames_hash = compute_frames_hash(frame_paths)

        # Check if existing manifest is still valid
        if self.manifest is not None:
            if self.manifest.input_frames_hash != frames_hash:
                logger.info("Input frames changed, invalidating cache")
                self.clear()
            elif self.manifest.config_hash != config_hash:
                logger.info("Config changed, invalidating post-score cache")
                self.clear(keep_scores=True)

        if self.manifest is None:
            self.manifest = CacheManifest(
                version=__version__,
                created=datetime.now().isoformat(),
                session_id=self.session_id,
                config_hash=config_hash,
                input_frames_hash=frames_hash,
                n_input_frames=len(frame_paths),
            )
            self._save_manifest()

        return self.manifest

    def clear(self, keep_scores: bool = False) -> None:
        """
        Clear cached artifacts.

        Parameters
        ----------
        keep_scores : bool, default False
            If True, preserve scores cache (useful when only config changed).
        """
        files_to_remove = [
            TRANSFORMS_FILE,
            STACK_MONO_FILE,
            STACK_R_FILE,
            STACK_G_FILE,
            STACK_B_FILE,
            STACK_RGB_FILE,
            CALIBRATION_FILE,
            MANIFEST_FILE,
        ]

        if not keep_scores:
            files_to_remove.append(SCORES_FILE)

        for fname in files_to_remove:
            fpath = self.cache_dir / fname
            if fpath.exists():
                fpath.unlink()
                logger.debug("Removed cached file: %s", fname)

        self.manifest = None
        logger.info("Cache cleared")

    # --- Scores ---

    def has_scores(self) -> bool:
        """Check if scores are cached."""
        return (self.cache_dir / SCORES_FILE).exists()

    def save_scores(self, scores: list[FrameScore]) -> None:
        """
        Save frame scores to cache.

        Parameters
        ----------
        scores : list[FrameScore]
            Scores to cache.
        """
        scores_path = self.cache_dir / SCORES_FILE

        data = {
            "version": __version__,
            "n_scores": len(scores),
            "scores": [asdict(s) for s in scores],
        }

        with open(scores_path, "w") as f:
            json.dump(data, f, indent=2)

        if self.manifest:
            self.manifest.has_scores = True
            self._save_manifest()

        logger.info("Saved %d scores to cache", len(scores))

    def load_scores(self) -> list[FrameScore] | None:
        """
        Load cached scores.

        Returns
        -------
        list[FrameScore] or None
            Cached scores, or None if not available.
        """
        scores_path = self.cache_dir / SCORES_FILE
        if not scores_path.exists():
            return None

        try:
            with open(scores_path) as f:
                data = json.load(f)

            scores = [FrameScore(**s) for s in data["scores"]]
            logger.info("Loaded %d cached scores", len(scores))
            return scores

        except Exception as e:
            logger.warning("Failed to load cached scores: %s", e)
            return None

    # --- Transforms ---

    def has_transforms(self) -> bool:
        """Check if transforms are cached."""
        return (self.cache_dir / TRANSFORMS_FILE).exists()

    def get_transforms_path(self) -> Path:
        """Get path to transforms cache file."""
        return self.cache_dir / TRANSFORMS_FILE

    def update_transforms_status(self, n_aligned: int) -> None:
        """Update manifest after transforms are saved."""
        if self.manifest:
            self.manifest.has_transforms = True
            self.manifest.n_aligned_frames = n_aligned
            self._save_manifest()

    # --- Stacked images ---

    def has_stack_mono(self) -> bool:
        """Check if mono stack is cached."""
        return (self.cache_dir / STACK_MONO_FILE).exists()

    def has_stack_rgb(self) -> bool:
        """Check if RGB stack is cached."""
        return (self.cache_dir / STACK_RGB_FILE).exists()

    def has_stack_channels(self) -> bool:
        """Check if per-channel stacks are cached."""
        return all([
            (self.cache_dir / STACK_R_FILE).exists(),
            (self.cache_dir / STACK_G_FILE).exists(),
            (self.cache_dir / STACK_B_FILE).exists(),
        ])

    def save_stack_mono(
        self,
        data: np.ndarray,
        mask_count: np.ndarray,
        header: dict | None = None,
    ) -> Path:
        """
        Save mono stack to cache.

        Parameters
        ----------
        data : np.ndarray
            Stacked image data.
        mask_count : np.ndarray
            Contributing frame count per pixel.
        header : dict, optional
            FITS header metadata.

        Returns
        -------
        Path
            Path to saved file.
        """
        stack_path = self.cache_dir / STACK_MONO_FILE
        self._save_fits_stack(stack_path, data, mask_count, header)

        if self.manifest:
            self.manifest.has_stack_mono = True
            self._save_manifest()

        return stack_path

    def load_stack_mono(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Load cached mono stack.

        Returns
        -------
        tuple or None
            (data, mask_count) or None if not available.
        """
        stack_path = self.cache_dir / STACK_MONO_FILE
        return self._load_fits_stack(stack_path)

    def save_stack_rgb(
        self,
        data: np.ndarray,
        mask_count: np.ndarray,
        header: dict | None = None,
    ) -> Path:
        """
        Save RGB stack to cache.

        Parameters
        ----------
        data : np.ndarray
            RGB stacked image (H, W, 3).
        mask_count : np.ndarray
            Contributing frame count per pixel (H, W).
        header : dict, optional
            FITS header metadata.

        Returns
        -------
        Path
            Path to saved file.
        """
        stack_path = self.cache_dir / STACK_RGB_FILE
        self._save_fits_rgb(stack_path, data, mask_count, header)

        if self.manifest:
            self.manifest.has_stack_rgb = True
            self._save_manifest()

        return stack_path

    def load_stack_rgb(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Load cached RGB stack.

        Returns
        -------
        tuple or None
            (rgb_data, mask_count) or None if not available.
        """
        stack_path = self.cache_dir / STACK_RGB_FILE
        return self._load_fits_rgb(stack_path)

    def save_stack_channel(
        self,
        channel: str,
        data: np.ndarray,
        mask_count: np.ndarray,
    ) -> Path:
        """
        Save individual channel stack.

        Parameters
        ----------
        channel : str
            Channel name: 'R', 'G', or 'B'.
        data : np.ndarray
            Channel data.
        mask_count : np.ndarray
            Mask count for channel.

        Returns
        -------
        Path
            Path to saved file.
        """
        file_map = {"R": STACK_R_FILE, "G": STACK_G_FILE, "B": STACK_B_FILE}
        if channel not in file_map:
            raise ValueError(f"Invalid channel: {channel}")

        stack_path = self.cache_dir / file_map[channel]
        self._save_fits_stack(stack_path, data, mask_count)
        return stack_path

    def load_stack_channel(self, channel: str) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Load individual channel stack.

        Parameters
        ----------
        channel : str
            Channel name: 'R', 'G', or 'B'.

        Returns
        -------
        tuple or None
            (data, mask_count) or None if not available.
        """
        file_map = {"R": STACK_R_FILE, "G": STACK_G_FILE, "B": STACK_B_FILE}
        if channel not in file_map:
            raise ValueError(f"Invalid channel: {channel}")

        stack_path = self.cache_dir / file_map[channel]
        return self._load_fits_stack(stack_path)

    # --- Calibration ---

    def has_calibration(self) -> bool:
        """Check if calibration is cached."""
        return (self.cache_dir / CALIBRATION_FILE).exists()

    def save_calibration(self, calibration: dict) -> None:
        """
        Save calibration parameters.

        Parameters
        ----------
        calibration : dict
            Calibration data (gains, offsets, etc.).
        """
        calib_path = self.cache_dir / CALIBRATION_FILE

        # Convert numpy arrays to lists for JSON
        serializable = {}
        for k, v in calibration.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            else:
                serializable[k] = v

        with open(calib_path, "w") as f:
            json.dump(serializable, f, indent=2)

        if self.manifest:
            self.manifest.has_calibration = True
            self._save_manifest()

        logger.info("Saved calibration to cache")

    def load_calibration(self) -> dict | None:
        """
        Load cached calibration.

        Returns
        -------
        dict or None
            Calibration data or None if not available.
        """
        calib_path = self.cache_dir / CALIBRATION_FILE
        if not calib_path.exists():
            return None

        try:
            with open(calib_path) as f:
                data = json.load(f)
            logger.info("Loaded cached calibration")
            return data
        except Exception as e:
            logger.warning("Failed to load cached calibration: %s", e)
            return None

    # --- Aligned frames ---

    def has_aligned_frames(self) -> bool:
        """Check if aligned frames are cached."""
        aligned_dir = self.cache_dir / ALIGNED_DIR
        manifest_path = aligned_dir / ALIGNED_MANIFEST
        return manifest_path.exists()

    def get_aligned_dir(self) -> Path:
        """Get directory for aligned frame cache."""
        aligned_dir = self.cache_dir / ALIGNED_DIR
        aligned_dir.mkdir(parents=True, exist_ok=True)
        return aligned_dir

    def save_aligned_frames(
        self,
        aligned_frames: list[np.ndarray],
        validity_masks: list[np.ndarray],
        frame_paths: list[Path],
        reference_path: str,
    ) -> Path:
        """
        Save aligned frames to disk cache using memory-mapped files.

        Parameters
        ----------
        aligned_frames : list[np.ndarray]
            List of aligned RGB frames (H, W, 3).
        validity_masks : list[np.ndarray]
            List of validity masks (H, W).
        frame_paths : list[Path]
            Original frame paths (for validation).
        reference_path : str
            Path to reference frame.

        Returns
        -------
        Path
            Path to aligned frames directory.
        """
        aligned_dir = self.get_aligned_dir()

        n_frames = len(aligned_frames)
        if n_frames == 0:
            logger.warning("No aligned frames to save")
            return aligned_dir

        h, w, c = aligned_frames[0].shape

        # Save as numpy memmap for fast loading
        rgb_path = aligned_dir / "aligned_rgb.npy"
        mask_path = aligned_dir / "aligned_masks.npy"

        # Create memmap files
        rgb_shape = (n_frames, h, w, c)
        mask_shape = (n_frames, h, w)

        rgb_mmap = np.memmap(rgb_path, dtype=np.float32, mode='w+', shape=rgb_shape)
        mask_mmap = np.memmap(mask_path, dtype=np.float32, mode='w+', shape=mask_shape)

        # Write frames
        for i, (rgb, mask) in enumerate(zip(aligned_frames, validity_masks)):
            rgb_mmap[i] = rgb.astype(np.float32)
            mask_mmap[i] = mask.astype(np.float32)

        rgb_mmap.flush()
        mask_mmap.flush()
        del rgb_mmap, mask_mmap

        # Save manifest
        manifest = {
            "version": __version__,
            "created": datetime.now().isoformat(),
            "n_frames": n_frames,
            "shape": [h, w, c],
            "reference": reference_path,
            "frame_names": [p.name for p in frame_paths],
            "rgb_file": "aligned_rgb.npy",
            "mask_file": "aligned_masks.npy",
            "total_size_gb": (n_frames * h * w * (c + 1) * 4) / 1e9,
        }

        manifest_path = aligned_dir / ALIGNED_MANIFEST
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "Saved %d aligned frames to cache (%.1f GB)",
            n_frames, manifest["total_size_gb"]
        )

        if self.manifest:
            self.manifest.n_aligned_frames = n_frames
            self._save_manifest()

        return aligned_dir

    def load_aligned_frames(
        self,
        expected_frames: list[Path] | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], dict] | None:
        """
        Load aligned frames from cache.

        Parameters
        ----------
        expected_frames : list[Path], optional
            If provided, validate that cached frames match this list.

        Returns
        -------
        tuple or None
            (aligned_frames, validity_masks, manifest) or None if not available.
        """
        aligned_dir = self.cache_dir / ALIGNED_DIR
        manifest_path = aligned_dir / ALIGNED_MANIFEST

        if not manifest_path.exists():
            return None

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            # Validate frame list if provided
            if expected_frames is not None:
                expected_names = [p.name for p in expected_frames]
                if manifest["frame_names"] != expected_names:
                    logger.info("Aligned frame cache invalidated: frame list changed")
                    return None

            # Load memmap files
            rgb_path = aligned_dir / manifest["rgb_file"]
            mask_path = aligned_dir / manifest["mask_file"]

            if not rgb_path.exists() or not mask_path.exists():
                logger.warning("Aligned cache files missing")
                return None

            n_frames = manifest["n_frames"]
            h, w, c = manifest["shape"]

            rgb_mmap = np.memmap(rgb_path, dtype=np.float32, mode='r',
                                 shape=(n_frames, h, w, c))
            mask_mmap = np.memmap(mask_path, dtype=np.float32, mode='r',
                                  shape=(n_frames, h, w))

            # Load into memory (we need them for stacking anyway)
            aligned_frames = [rgb_mmap[i].copy() for i in range(n_frames)]
            validity_masks = [mask_mmap[i].copy() for i in range(n_frames)]

            del rgb_mmap, mask_mmap

            logger.info(
                "Loaded %d aligned frames from cache (%.1f GB)",
                n_frames, manifest["total_size_gb"]
            )

            return aligned_frames, validity_masks, manifest

        except Exception as e:
            logger.warning("Failed to load aligned frames cache: %s", e)
            return None

    def clear_aligned_frames(self) -> None:
        """Clear aligned frames cache."""
        aligned_dir = self.cache_dir / ALIGNED_DIR
        if aligned_dir.exists():
            import shutil
            shutil.rmtree(aligned_dir)
            logger.info("Cleared aligned frames cache")

    def get_aligned_info(self) -> dict | None:
        """Get info about cached aligned frames without loading them."""
        aligned_dir = self.cache_dir / ALIGNED_DIR
        manifest_path = aligned_dir / ALIGNED_MANIFEST

        if not manifest_path.exists():
            return None

        try:
            with open(manifest_path) as f:
                return json.load(f)
        except Exception:
            return None

    # --- Internal helpers ---

    def _save_fits_stack(
        self,
        path: Path,
        data: np.ndarray,
        mask_count: np.ndarray,
        header_dict: dict | None = None,
    ) -> None:
        """Save 2D stack to FITS with mask as extension."""
        hdr = fits.Header()
        hdr["DWARF3"] = (__version__, "dwarf3 version")
        hdr["CACHTYP"] = ("STACK_MONO", "Cache type")
        hdr["CREATED"] = (datetime.now().isoformat(), "Creation timestamp")

        if header_dict:
            for k, v in header_dict.items():
                if len(k) <= 8:  # FITS key length limit
                    hdr[k] = v

        primary = fits.PrimaryHDU(data.astype(np.float32), header=hdr)
        mask_hdu = fits.ImageHDU(mask_count.astype(np.int16), name="MASKCOUNT")

        hdul = fits.HDUList([primary, mask_hdu])
        hdul.writeto(path, overwrite=True)
        logger.debug("Saved stack cache: %s", path)

    def _load_fits_stack(self, path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        """Load 2D stack from FITS."""
        if not path.exists():
            return None

        try:
            with fits.open(path) as hdul:
                data = hdul[0].data.astype(np.float32)
                mask_count = hdul["MASKCOUNT"].data.astype(np.int16)
            return data, mask_count
        except Exception as e:
            logger.warning("Failed to load stack cache %s: %s", path, e)
            return None

    def _save_fits_rgb(
        self,
        path: Path,
        rgb_data: np.ndarray,
        mask_count: np.ndarray,
        header_dict: dict | None = None,
    ) -> None:
        """Save RGB stack to FITS (channels as extensions)."""
        hdr = fits.Header()
        hdr["DWARF3"] = (__version__, "dwarf3 version")
        hdr["CACHTYP"] = ("STACK_RGB", "Cache type")
        hdr["CREATED"] = (datetime.now().isoformat(), "Creation timestamp")
        hdr["NAXIS3"] = (3, "RGB channels")

        if header_dict:
            for k, v in header_dict.items():
                if len(k) <= 8:
                    hdr[k] = v

        # Store as (3, H, W) for FITS convention
        rgb_fits = np.transpose(rgb_data, (2, 0, 1)).astype(np.float32)

        primary = fits.PrimaryHDU(rgb_fits, header=hdr)
        mask_hdu = fits.ImageHDU(mask_count.astype(np.int16), name="MASKCOUNT")

        hdul = fits.HDUList([primary, mask_hdu])
        hdul.writeto(path, overwrite=True)
        logger.debug("Saved RGB stack cache: %s", path)

    def _load_fits_rgb(self, path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        """Load RGB stack from FITS."""
        if not path.exists():
            return None

        try:
            with fits.open(path) as hdul:
                rgb_fits = hdul[0].data.astype(np.float32)
                mask_count = hdul["MASKCOUNT"].data.astype(np.int16)

            # Convert from (3, H, W) to (H, W, 3)
            rgb_data = np.transpose(rgb_fits, (1, 2, 0))
            return rgb_data, mask_count
        except Exception as e:
            logger.warning("Failed to load RGB stack cache %s: %s", path, e)
            return None

    def get_status(self) -> dict:
        """
        Get cache status summary.

        Returns
        -------
        dict
            Summary of what is cached.
        """
        status = {
            "cache_dir": str(self.cache_dir),
            "has_manifest": self.manifest is not None,
            "has_scores": self.has_scores(),
            "has_transforms": self.has_transforms(),
            "has_aligned_frames": self.has_aligned_frames(),
            "has_stack_mono": self.has_stack_mono(),
            "has_stack_rgb": self.has_stack_rgb(),
            "has_stack_channels": self.has_stack_channels(),
            "has_calibration": self.has_calibration(),
        }

        # Add aligned frame info if available
        aligned_info = self.get_aligned_info()
        if aligned_info:
            status["aligned_n_frames"] = aligned_info.get("n_frames", 0)
            status["aligned_size_gb"] = aligned_info.get("total_size_gb", 0)

        return status


def get_session_cache(
    output_dir: Path,
    session_id: str,
) -> PipelineCache:
    """
    Get or create a cache manager for a session.

    Parameters
    ----------
    output_dir : Path
        Base output directory (processedData/<session>).
    session_id : str
        Session identifier.

    Returns
    -------
    PipelineCache
        Cache manager instance.
    """
    cache_dir = output_dir / CACHE_DIR
    return PipelineCache(cache_dir, session_id)
