#!/usr/bin/env python3
"""
Trajectory montage: the observed eclipse discs placed along their true sky
path over the Viaduc des Fauvettes photograph (IMG_3846, 20:18 local —
within 30 s of maximum eclipse).

Geometry
--------
The backdrop photo contains the (overexposed) Sun; its saturated-blob
centroid is the calibration anchor. With the EXIF 35 mm-equivalent focal
length (24 mm -> f_px = diag_px / (2 tan(atan(43.27/48)))) and the simulated
Sun altitude/azimuth at the EXIF capture time, the camera pose (yaw, pitch;
roll assumed 0 — handheld but level horizon) is solved so the Sun maps to
its observed pixel. Every other (az, alt) then projects rectilinearly.

Disc sprites are cut from the processed session masters (stack.png) around
the fitted disk center and composited in 'lighten' mode. At true angular
scale the Sun is ~36 px across on this photo, so discs are enlarged by
--disc-scale (default 4x): trajectory is exact, size is exaggerated —
the caption should say so.

Author: Olivier Vitrac — dwarf3 project. MIT license.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

LOCAL_UTC_OFFSET_H = 2.0

# IMG_3846 EXIF (iPhone 17 Pro Max, main camera)
EXIF_TIME_LOCAL = "2026-08-12T20:18:00"
EQUIV_FOCAL_MM = 24.0
FULLFRAME_DIAG_MM = 43.266


def sun_pixel(im: np.ndarray) -> tuple[float, float]:
    """Centroid of the largest saturated blob (the Sun's core)."""
    luma = im @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    m = (luma > 0.98 * luma.max()).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m)
    if n < 2:
        raise RuntimeError("no saturated blob found in backdrop")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return float(cents[largest][0]), float(cents[largest][1])


def altaz_to_vec(az_deg: float, alt_deg: float) -> np.ndarray:
    """Unit vector, x=East, y=North, z=Up."""
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    return np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])


class CameraModel:
    """Rectilinear camera, roll = 0, solved from one anchor direction."""

    def __init__(self, w: int, h: int, f_px: float,
                 anchor_azalt: tuple[float, float], anchor_px: tuple[float, float]):
        self.w, self.h, self.f = w, h, f_px
        az0, alt0 = anchor_azalt
        u0, v0 = anchor_px
        # offsets of the anchor from the image center, in camera angles
        dx = (u0 - w / 2) / f_px   # tan of yaw offset
        dy = (v0 - h / 2) / f_px   # tan of pitch offset (down positive)
        self.az_c = az0 - np.degrees(np.arctan(dx)) / np.cos(np.radians(alt0))
        self.alt_c = alt0 + np.degrees(np.arctan(dy))

    def project(self, az: float, alt: float) -> tuple[float, float]:
        """(az, alt) in degrees -> pixel (u, v). Small-field rectilinear."""
        daz = np.radians(az - self.az_c) * np.cos(np.radians(alt))
        dalt = np.radians(alt - self.alt_c)
        u = self.w / 2 + self.f * np.tan(daz)
        v = self.h / 2 - self.f * np.tan(dalt)
        return float(u), float(v)


def cut_disc(root: Path, sess: dict, pad: float = 1.6) -> np.ndarray | None:
    """Cut a square sprite around the fitted disk center from stack.png."""
    if not sess.get("disk_center_xy") or not sess.get("disk_radius_px"):
        return None
    png = cv2.imread(str(root / sess["session"] / "stack.png"), cv2.IMREAD_UNCHANGED)
    if png is None:
        return None
    img = (png.astype(np.float32) / 257.0).astype(np.uint8) if png.dtype == np.uint16 else png
    cx, cy = sess["disk_center_xy"]
    r = sess["disk_radius_px"] * pad
    x0, x1 = int(cx - r), int(cx + r)
    y0, y1 = int(cy - r), int(cy + r)
    h, w = img.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        # clamp with black padding
        out = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(w, x1), min(h, y1)
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        return out
    return img[y0:y1, x0:x1]


def viewport_paste(canvas: np.ndarray, sprite: np.ndarray, cx: float, cy: float,
                   dark: float = 0.72):
    """
    Composite sprite (BGR) centered at (cx, cy): first darken a feathered
    circular 'smoked-glass viewport' (the sky is far brighter than the
    filtered telescope discs), then lighten-paste the brightness-boosted
    sprite inside it. In place.
    """
    sh, sw = sprite.shape[:2]
    x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
    x1, y1 = x0 + sw, y0 + sh
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas.shape[1], x1), min(canvas.shape[0], y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sub = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    spr = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0].astype(np.float32)

    # boost sprite so the crescent reaches full brightness; measure the
    # reference level on lit pixels only (tree-occluded discs are mostly
    # dark and would otherwise be blown into red noise)
    lit = spr[spr.max(axis=2) > 20]
    p99 = np.percentile(lit, 99.0) if len(lit) else 255.0
    spr = np.clip(spr * (255.0 / max(p99, 60.0)), 0, 255)

    # feathered radial mask over the *sprite* extent
    yy, xx = np.mgrid[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0].astype(np.float32)
    r = np.hypot(xx - sw / 2, yy - sh / 2) / (0.5 * min(sw, sh))
    g = np.clip((1.0 - r) / 0.35, 0, 1)  # 1 inside, feather over outer 35 %
    g = (g * g * (3 - 2 * g))[..., None]  # smoothstep

    darkened = sub * (1.0 - dark * g)
    out = np.maximum(darkened, spr * g)
    canvas[cy0:cy1, cx0:cx1] = np.clip(out, 0, 255).astype(np.uint8)


def pick_sessions(sessions: list[dict], n_target: int = 13) -> list[dict]:
    """
    Evenly-spaced-in-time selection, always keeping first, maximum and the
    last usable disc; prefers clean (non-occluded, non-transient) sessions.
    """
    def sane(s):
        c = s.get("disk_center_xy")
        return c and abs(c[0] - 480) < 60 and abs(c[1] - 270) < 45

    usable = [s for s in sessions if sane(s)]
    if not usable:
        return []
    t = np.array([s["utc_h_mid"] for s in usable])
    i_max = int(np.argmax([s["sim"]["obsc"] for s in usable]))
    targets = np.linspace(t[0], t[-1], n_target)
    chosen: dict[int, dict] = {}
    for tt in targets:
        order = np.argsort(np.abs(t - tt))
        best = next(
            (int(i) for i in order[:5]
             if not usable[int(i)]["occluded"] and not usable[int(i)].get("transient")),
            int(order[0]),
        )
        chosen[best] = usable[best]
    chosen[i_max] = usable[i_max]
    chosen[len(usable) - 1] = usable[-1]
    picks = [chosen[i] for i in sorted(chosen)]
    # dedupe discs closer than 3 min, preferring maximum then earlier
    deduped: list[dict] = []
    for p in picks:
        if deduped and abs(p["utc_h_mid"] - deduped[-1]["utc_h_mid"]) < 0.05:
            if p["sim"]["obsc"] > deduped[-1]["sim"]["obsc"]:
                deduped[-1] = p
            continue
        deduped.append(p)
    return deduped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", type=Path, default=Path("processedData/eclipse_2026-08-12"))
    ap.add_argument("--backdrop", type=Path, required=True, help="IMG_3846 as JPG/PNG")
    ap.add_argument("--sim", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/montage/")
    ap.add_argument("--n", type=int, default=10, help="target number of discs")
    ap.add_argument("--disc-scale", type=float, default=3.0,
                    help="disc size exaggeration relative to true angular scale")
    ap.add_argument("--path-line", action="store_true", default=True)
    ap.add_argument("--labels", action="store_true", default=True)
    args = ap.parse_args(argv)

    out_dir = args.out or args.root / "montage"
    out_dir.mkdir(parents=True, exist_ok=True)

    # backdrop + camera pose
    im = cv2.imread(str(args.backdrop))  # BGR
    imf = im[..., ::-1].astype(np.float32) / 255.0
    h, w = im.shape[:2]
    diag_px = float(np.hypot(w, h))
    f_px = (diag_px / 2) / np.tan(np.arctan(FULLFRAME_DIAG_MM / (2 * EQUIV_FOCAL_MM)))

    sim = json.loads(args.sim.read_text())
    fr = sim["frames"]
    tarr = np.array([f["utc_h"] for f in fr])
    alt = np.array([f["alt"] for f in fr])
    az = np.array([f["az"] for f in fr])

    texif = datetime.fromisoformat(EXIF_TIME_LOCAL)
    utc_exif = texif.hour + texif.minute / 60 + texif.second / 3600 - LOCAL_UTC_OFFSET_H
    sun_azalt = (float(np.interp(utc_exif, tarr, az)), float(np.interp(utc_exif, tarr, alt)))
    anchor = sun_pixel(imf)
    cam = CameraModel(w, h, f_px, sun_azalt, anchor)
    print(f"anchor px={anchor}, sun az/alt={sun_azalt}, f_px={f_px:.0f}, "
          f"camera center az={cam.az_c:.2f} alt={cam.alt_c:.2f}")

    manifest = json.loads((args.root / "manifest.json").read_text())
    sessions = sorted(manifest["sessions"], key=lambda s: s["utc_h_mid"])
    picks = pick_sessions(sessions, args.n)
    print(f"{len(picks)} discs selected")

    canvas = im.copy()

    # predicted path (thin line across the whole photo time range)
    if args.path_line:
        span = [s["utc_h_mid"] for s in sessions]
        ts = np.linspace(span[0], span[-1], 200)
        pts = np.array([
            cam.project(float(np.interp(t, tarr, az)), float(np.interp(t, tarr, alt)))
            for t in ts
        ], dtype=np.int32)
        ok = (pts[:, 0] > -w) & (pts[:, 0] < 2 * w) & (pts[:, 1] > -h) & (pts[:, 1] < 2 * h)
        overlay = canvas.copy()
        cv2.polylines(overlay, [pts[ok]], False, (70, 55, 40), 4, cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0)

    # true angular scale on this photo, then exaggeration
    sun_diam_px_true = 2 * np.tan(np.radians(0.263)) * f_px
    for s in picks:
        sprite = cut_disc(args.root, s)
        if sprite is None:
            continue
        # sprite covers pad*2 = 3.2 solar radii -> target size on canvas
        target = int(round(sun_diam_px_true * args.disc_scale * 1.6))
        sprite = cv2.resize(sprite, (target, target), interpolation=cv2.INTER_AREA)
        u, v = cam.project(s["sim"]["az"], s["sim"]["alt"])
        viewport_paste(canvas, sprite, u, v)
        if args.labels:
            # labels sit perpendicular to the descending path (upper right)
            t = s["time_local_mid"][11:16]
            lx, ly = int(u) + target // 2 + 14, int(v) - target // 2 + 8
            cv2.putText(canvas, t, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (10, 10, 10), 7, cv2.LINE_AA)
            cv2.putText(canvas, t, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (240, 240, 240), 2, cv2.LINE_AA)

    full = out_dir / "montage_full.png"
    web = out_dir / "montage_web.jpg"
    cv2.imwrite(str(full), canvas)
    cv2.imwrite(str(web), cv2.resize(canvas, (2400, int(2400 * h / w)),
                                     interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    meta = {
        "backdrop": str(args.backdrop),
        "exif_time_local": EXIF_TIME_LOCAL,
        "anchor_px": anchor,
        "sun_azalt_at_exif": sun_azalt,
        "f_px": f_px,
        "camera_center_azalt": [cam.az_c, cam.alt_c],
        "disc_scale_exaggeration": args.disc_scale,
        "sessions": [s["session"] for s in picks],
    }
    (out_dir / "montage_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"written: {full}\n         {web}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
