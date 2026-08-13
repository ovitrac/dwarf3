#!/usr/bin/env python3
"""
Solar eclipse session processor for DWARF 3 raw data (2026-08-12 partial eclipse).

Processes each DWARF_RAW_TELE_Sun_* session into a registered, sigma-clipped
master (linear FITS + display PNG) with adaptive stacking depth:

- clear sky      : stack all frames (device default 20)
- tree-occluded  : stack only the first N (default 5) frames, because the
  mount tracks the Sun while trees are fixed in alt-az (and move with wind);
  deep stacks smear the foliage silhouette.

Occlusion is detected from the temporal instability of the lit-pixel mask
(mean XOR fraction between consecutive frames): a lunar crescent is
geometrically stable over a 30 s burst, wind-blown foliage is not.

Registration is translation-only on the lit-mask centroid (sub-pixel,
center of mass). Solar disks have no stars: astroalign does not apply.

Photometric normalization: ADU / EXPTIME (electrons-proxy per second), so the
exposure ladder (0.002 -> 0.077 s) maps to a continuous brightness scale.
The single GAIN=0 session is rescaled by an empirical factor derived from
disk surface-brightness continuity with its GAIN=10 neighbours.

Each master carries sim-interpolated circumstances (alt, az, mag, obsc, pa)
from the Viaduc des Fauvettes simulation JSON, plus a measured lit-disc
fraction for the observed-vs-predicted agreement panel.

Outputs (per session) under processedData/eclipse_2026-08-12/<session>/:
    stack_rgb.fits   linear float32 RGB master (3 HDUs: R, G, B)
    stack.png        16-bit sRGB display render (global fixed stretch)
    meta.json        all decisions, metrics and sim circumstances

Plus a global manifest.json at the output root.

Author: Olivier Vitrac — dwarf3 project. MIT license.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

# dwarf3 package (installed editable from src/)
from dwarf3.debayer import debayer_rggb
from dwarf3.io import list_lights, read_fits, read_header
from dwarf3.stack import sigma_clip_mean_rgb

log = logging.getLogger("dwarf3.eclipse")

LOCAL_UTC_OFFSET_H = 2.0  # Europe/Paris, CEST on 2026-08-12

# Display-only white balance, anchored so the 19:16 pre-contact disc matches
# the DWARF device rendering (stacked.jpg) of the same session. Constant for
# all sessions so the real atmospheric reddening evolves freely.
# Derivation: linearized device disc RGB ratios / linear master disc ratios.
DISPLAY_GAIN_R = 9.044
DISPLAY_GAIN_B = 0.650

# ----------------------------------------------------------------------------
# session discovery & metadata
# ----------------------------------------------------------------------------

SESSION_RE = re.compile(
    r"DWARF_RAW_TELE_Sun_EXP_(?P<exp>[0-9.]+)_GAIN_(?P<gain>\d+)_"
    r"(?P<ts>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3})$"
)

FRAME_TS_RE = re.compile(r"_(\d{8})-(\d{6})(\d{3})?_")


@dataclass
class SessionInfo:
    path: Path
    name: str
    exptime_s: float
    gain: int
    start_local: datetime  # from folder name (local clock)


def discover_sessions(raw_root: Path) -> list[SessionInfo]:
    """Find Sun sessions under raw_root, sorted by start time."""
    out: list[SessionInfo] = []
    for p in sorted(raw_root.glob("DWARF_RAW_TELE_Sun_*")):
        m = SESSION_RE.match(p.name)
        if not m or not p.is_dir():
            continue
        ts = datetime.strptime(m.group("ts")[:19], "%Y-%m-%d-%H-%M-%S")
        out.append(
            SessionInfo(
                path=p,
                name=p.name,
                exptime_s=float(m.group("exp")),
                gain=int(m.group("gain")),
                start_local=ts,
            )
        )
    out.sort(key=lambda s: s.start_local)
    return out


def frame_time_local(path: Path) -> datetime | None:
    """Parse the local timestamp embedded in a DWARF frame filename."""
    m = FRAME_TS_RE.search(path.name)
    if not m:
        return None
    base = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    if m.group(3):
        base += timedelta(milliseconds=int(m.group(3)))
    return base


# ----------------------------------------------------------------------------
# simulation circumstances
# ----------------------------------------------------------------------------


class SimCircumstances:
    """Interpolate simulated frames (utc_h -> alt, az, mag, obsc, pa, sd, hz)."""

    KEYS = ("alt", "az", "mag", "obsc", "pa", "sd")

    def __init__(self, sim_json: Path, canopy: str = "veg20"):
        d = json.loads(sim_json.read_text())
        fr = d["frames"]
        self.t = np.array([f["utc_h"] for f in fr])
        self.cols = {k: np.array([float(f.get(k, np.nan)) for f in fr]) for k in self.KEYS}
        # hz: horizon elevation at the Sun's azimuth, per canopy model
        canopy = d.get("canopy_default", canopy)
        self.cols["hz"] = np.array(
            [float(f["hz"].get(canopy, np.nan)) if isinstance(f.get("hz"), dict)
             else float(f.get("hz", np.nan)) for f in fr]
        )
        self.site = d.get("site", {})

    def at(self, utc_h: float) -> dict[str, float]:
        return {k: float(np.interp(utc_h, self.t, v)) for k, v in self.cols.items()}


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------


def lit_mask(luma: np.ndarray, frac: float = 0.25, min_signal: float = 30.0) -> np.ndarray:
    """
    Binary mask of sun-lit pixels.

    Threshold sits `frac` of the way between the sky background (median: sky
    dominates every frame) and the 99.9th percentile. The DWARF 3 sensor has
    a ~200 ADU pedestal, and near sunset the crescent peaks at only ~750 ADU,
    so an absolute fraction-of-peak threshold would swallow the whole frame.
    """
    bg = float(np.median(luma))
    hi = float(np.percentile(luma, 99.9))
    if hi - bg < min_signal:  # no significant lit region
        return np.zeros(luma.shape, dtype=bool)
    return luma > bg + frac * (hi - bg)


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """(x, y) center of mass of a boolean mask; None if empty."""
    n = int(mask.sum())
    if n < 50:
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Sub-pixel translation (bilinear), edges replicated."""
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img, m, (img.shape[1], img.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def fit_disk_center(
    mask: np.ndarray, radius_px: float, n_iter: int = 300, tol: float = 3.0,
    rng_seed: int = 12345,
) -> tuple[float, float, float] | None:
    """
    RANSAC circle fit with fixed radius on the mask outline.

    The outer arc of the crescent lies on the solar limb; fitting a circle of
    known solar radius recovers the disk center even at magnitude 0.93.

    Both the solar and the lunar limb are circles of nearly the same radius,
    so candidates farther than 1.05 R from the lit-region centroid (the Moon's
    center always is; the Sun's never is) are rejected.
    Returns (cx, cy, inlier_fraction) or None.
    """
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150)
    ys, xs = np.nonzero(edges)
    if len(xs) < 30:
        return None
    cen = mask_centroid(mask)
    if cen is None:
        return None
    cen = np.array(cen)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    rng = np.random.default_rng(rng_seed)  # fixed seed: deterministic
    best: tuple[float, float, int] | None = None
    r = float(radius_px)
    for _ in range(n_iter):
        i, j = rng.choice(len(pts), 2, replace=False)
        p, q = pts[i], pts[j]
        d = np.linalg.norm(q - p)
        if d < 10 or d > 2 * r - 2:
            continue
        mid = (p + q) / 2
        h = np.sqrt(max(r * r - (d / 2) ** 2, 0.0))
        perp = np.array([-(q - p)[1], (q - p)[0]]) / d
        for sgn in (+1, -1):
            c = mid + sgn * h * perp
            if np.linalg.norm(c - cen) > 1.05 * r:  # lunar-limb candidate
                continue
            resid = np.abs(np.linalg.norm(pts - c, axis=1) - r)
            n_in = int((resid < tol).sum())
            if best is None or n_in > best[2]:
                best = (float(c[0]), float(c[1]), n_in)
    if best is None:
        return None
    return best[0], best[1], best[2] / len(pts)


def measure_disk_radius(mask: np.ndarray) -> float | None:
    """Radius from a (near-)full disk via minimum enclosing circle."""
    m8 = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (_, _), r = cv2.minEnclosingCircle(c)
    area = cv2.contourArea(c)
    if area / (np.pi * r * r) < 0.90:  # not a full disk
        return None
    return float(r)


# ----------------------------------------------------------------------------
# per-session processing
# ----------------------------------------------------------------------------


@dataclass
class SessionResult:
    session: str
    time_local_mid: str
    utc_h_mid: float
    exptime_s: float
    gain: int
    n_frames_total: int
    n_frames_used: int
    occluded: bool
    transient: bool
    mask_xor_metric: float
    disk_center_xy: tuple[float, float] | None
    disk_radius_px: float | None
    ransac_inlier_frac: float | None
    lit_fraction_measured: float | None
    norm_factor: float
    display_black: float = 0.0
    display_white: float = 0.0
    sim: dict[str, float] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def process_session(
    info: SessionInfo,
    out_dir: Path,
    sim: SimCircumstances,
    disk_radius_px: float | None,
    *,
    occluded_frames: int = 5,
    xor_threshold: float = 0.02,
    debayer_mode: str = "superpixel",
    gain0_factor: float = 1.0,
    sigma: float = 3.0,
    overwrite: bool = False,
) -> SessionResult:
    """Process one session. Returns metrics; writes FITS/PNG/JSON to out_dir."""
    lights = [Path(p) for p in list_lights(info.path)]
    if not lights:
        raise RuntimeError(f"no light frames in {info.path}")

    sdir = out_dir / info.name
    sdir.mkdir(parents=True, exist_ok=True)
    meta_path = sdir / "meta.json"
    if meta_path.exists() and not overwrite:
        d = json.loads(meta_path.read_text())
        log.info("%s: cached", info.name)
        return SessionResult(**d)

    # -- read + debayer all frames (superpixel: 1920x1080 RGB) ---------------
    rgb_frames: list[np.ndarray] = []
    lumas: list[np.ndarray] = []
    for p in lights:
        raw = read_fits(p)  # float32, BZERO applied
        rgb = debayer_rggb(raw, mode=debayer_mode)
        luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
        rgb_frames.append(rgb)
        lumas.append(luma)

    masks = [lit_mask(l) for l in lumas]

    # -- occlusion metric: mean XOR fraction between consecutive lit masks ---
    # A thin crescent flickers a little too (glow near threshold), so tree
    # occlusion additionally requires the Sun near the simulated horizon.
    xors = []
    for a, b in zip(masks[:-1], masks[1:]):
        union = int((a | b).sum())
        xors.append(int((a ^ b).sum()) / union if union else 0.0)
    xor_metric = float(np.mean(xors)) if xors else 0.0

    t_start = frame_time_local(lights[0]) or info.start_local
    utc_h0 = t_start.hour + t_start.minute / 60 + t_start.second / 3600 - LOCAL_UTC_OFFSET_H
    circ0 = sim.at(utc_h0)
    near_horizon = (circ0["alt"] - circ0["hz"]) < 1.5  # deg above tree line
    masks_empty = all(m.sum() < 500 for m in masks)  # Sun swallowed by trees
    unstable = xor_metric > xor_threshold
    occluded = (unstable and near_horizon) or masks_empty
    transient = unstable and not occluded  # clouds/haze flicker: keep depth

    n_use = min(occluded_frames, len(rgb_frames)) if occluded else len(rgb_frames)
    use = slice(0, n_use)

    # -- registration: lit-mask centroid, translation to first frame.
    # With no usable mask (Sun fully behind trees) fall back to identity:
    # the mount keeps tracking, and trees are what remains to be seen.
    refc = mask_centroid(masks[0])
    aligned: list[np.ndarray] = []
    n_fail = 0
    for rgb, m in zip(rgb_frames[use], masks[use]):
        c = mask_centroid(m)
        if refc is None or c is None:
            aligned.append(rgb)
            n_fail += 1
            continue
        aligned.append(shift_image(rgb, refc[0] - c[0], refc[1] - c[1]))

    # -- stack ----------------------------------------------------------------
    if len(aligned) >= 3:
        stacked, _ = sigma_clip_mean_rgb(aligned, sigma=sigma)
    else:
        stacked = np.mean(np.stack(aligned), axis=0)

    # -- photometric normalization: ADU/s (+ empirical gain-0 rescale) -------
    norm = 1.0 / info.exptime_s
    if info.gain == 0:
        norm *= gain0_factor
    stacked_n = stacked * norm

    # -- disk geometry on the stacked master ----------------------------------
    sluma = stacked_n[..., 0] * 0.2126 + stacked_n[..., 1] * 0.7152 + stacked_n[..., 2] * 0.0722
    smask = lit_mask(sluma)
    radius = measure_disk_radius(smask)  # full disk only (pre/post contacts)
    r_used = radius or disk_radius_px
    center = None
    inlier = None
    if r_used:
        fit = fit_disk_center(smask, r_used)
        if fit:
            center = (round(fit[0], 2), round(fit[1], 2))
            inlier = round(fit[2], 3)
    lit_frac = None
    if r_used and center and not occluded:
        lit_frac = float(smask.sum() / (np.pi * r_used**2))

    # -- timing + sim circumstances -------------------------------------------
    t0 = frame_time_local(lights[0]) or info.start_local
    t1 = frame_time_local(lights[: n_use][-1]) or t0
    tmid = t0 + (t1 - t0) / 2
    utc_h = tmid.hour + tmid.minute / 60 + tmid.second / 3600 - LOCAL_UTC_OFFSET_H
    circ = {k: round(v, 5) for k, v in sim.at(utc_h).items()}

    # -- outputs ---------------------------------------------------------------
    from astropy.io import fits as pyfits

    hdr0 = read_header(lights[0])
    hdus = [pyfits.PrimaryHDU()]
    for i, ch in enumerate("RGB"):
        h = pyfits.ImageHDU(stacked_n[..., i].astype(np.float32), name=ch)
        h.header["BUNIT"] = "ADU/s"
        hdus.append(h)
    ph = hdus[0].header
    ph["ORIGIN"] = "dwarf3 process_eclipse.py"
    ph["DATE-OBS"] = tmid.isoformat()
    ph["EXPTIME"] = info.exptime_s
    ph["GAIN"] = info.gain
    ph["NFRAMES"] = len(aligned)
    ph["OCCLUDED"] = occluded
    ph["NORMFACT"] = norm
    for k in ("TELESCOP", "INSTRUME", "BAYERPAT"):
        if k in hdr0:
            ph[k] = hdr0[k]
    ph.add_history("Registered on lit-mask centroid (translation only)")
    ph.add_history(f"Sigma-clipped mean of {len(aligned)}/{len(lights)} frames")
    fits_path = sdir / "stack_rgb.fits"
    pyfits.HDUList(hdus).writeto(fits_path, overwrite=True)

    # display PNG: per-channel sky/pedestal subtraction FIRST (so WB gains do
    # not tint the background), then device-anchored WB, then white point at
    # the disk, gamma 0.45. Radiometry lives in the FITS.
    bp = np.percentile(stacked_n, 15.0, axis=(0, 1)).astype(np.float32)
    sig = np.clip(stacked_n - bp, 0, None)
    disp_rgb = sig * np.array([DISPLAY_GAIN_R, 1.0, DISPLAY_GAIN_B], dtype=np.float32)
    dluma = disp_rgb[..., 0] * 0.2126 + disp_rgb[..., 1] * 0.7152 + disp_rgb[..., 2] * 0.0722
    black = float(bp.mean())  # logged for reference
    white = float(np.percentile(dluma, 99.5))
    if white < 1e-9:
        white = 1.0
    disp = np.clip(disp_rgb / white, 0, 1) ** 0.45
    png = (disp * 65535).astype(np.uint16)
    png_path = sdir / "stack.png"
    cv2.imwrite(str(png_path), cv2.cvtColor(png, cv2.COLOR_RGB2BGR))

    res = SessionResult(
        session=info.name,
        time_local_mid=tmid.isoformat(),
        utc_h_mid=round(utc_h, 5),
        exptime_s=info.exptime_s,
        gain=info.gain,
        n_frames_total=len(lights),
        n_frames_used=len(aligned),
        occluded=occluded,
        transient=transient,
        mask_xor_metric=round(xor_metric, 4),
        disk_center_xy=center,
        disk_radius_px=round(r_used, 1) if r_used else None,
        ransac_inlier_frac=inlier,
        lit_fraction_measured=round(lit_frac, 4) if lit_frac is not None else None,
        norm_factor=norm,
        display_black=black,
        display_white=white,
        sim=circ,
        outputs={"fits": str(fits_path), "png": str(png_path)},
        notes=(
            ([f"{n_fail} frame(s) stacked unregistered (no lit mask)"] if n_fail else [])
            + (["transient mask instability (clouds/haze), full depth kept"] if transient else [])
        ),
    )
    meta_path.write_text(json.dumps(asdict(res), indent=2))
    return res


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--raw-root", type=Path, default=Path("rawData"))
    ap.add_argument("--out", type=Path, default=Path("processedData/eclipse_2026-08-12"))
    ap.add_argument("--sim", type=Path, required=True, help="sim_results_*.json")
    ap.add_argument("--occluded-frames", type=int, default=5)
    ap.add_argument("--xor-threshold", type=float, default=0.02)
    ap.add_argument("--debayer", choices=["superpixel", "bilinear"], default="superpixel")
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--gain0-factor", type=float, default=1.0,
                    help="empirical flux rescale for GAIN=0 sessions")
    ap.add_argument("--only", type=str, default=None, help="substring filter on session name")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    sessions = discover_sessions(args.raw_root)
    if args.only:
        sessions = [s for s in sessions if args.only in s.name]
    if not sessions:
        log.error("no sessions found under %s", args.raw_root)
        return 1
    log.info("%d session(s) to process", len(sessions))
    if args.dry_run:
        for s in sessions:
            print(f"  {s.name}  exp={s.exptime_s}s gain={s.gain}")
        return 0

    sim = SimCircumstances(args.sim)
    args.out.mkdir(parents=True, exist_ok=True)

    results: list[SessionResult] = []
    disk_radius: float | None = None
    for i, s in enumerate(sessions):
        try:
            r = process_session(
                s, args.out, sim, disk_radius,
                occluded_frames=args.occluded_frames,
                xor_threshold=args.xor_threshold,
                debayer_mode=args.debayer,
                gain0_factor=args.gain0_factor,
                sigma=args.sigma,
                overwrite=args.overwrite,
            )
        except Exception as e:  # keep batch alive, report at the end
            log.error("%s: FAILED: %s", s.name, e)
            continue
        if disk_radius is None and r.disk_radius_px:
            disk_radius = r.disk_radius_px
            log.info("solar disk radius locked: %.1f px", disk_radius)
        log.info(
            "%s  used=%d/%d occluded=%s xor=%.3f lit=%s sim_obsc=%.3f",
            s.name, r.n_frames_used, r.n_frames_total, r.occluded,
            r.mask_xor_metric, str(r.lit_fraction_measured), r.sim.get("obsc", -1),
        )
        results.append(r)

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(args.raw_root),
        "sim": str(args.sim),
        "params": {
            "occluded_frames": args.occluded_frames,
            "xor_threshold": args.xor_threshold,
            "debayer": args.debayer,
            "sigma": args.sigma,
            "gain0_factor": args.gain0_factor,
        },
        "n_sessions": len(results),
        "sessions": [asdict(r) for r in results],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("manifest written: %s", args.out / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
