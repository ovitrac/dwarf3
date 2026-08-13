#!/usr/bin/env python3
"""
Assemble the eclipse timelapse video from processed session masters.

Reads processedData/eclipse_2026-08-12/manifest.json, re-centers every master
on the (smoothed) solar disk center, renders 1920x1080 frames with a time
stamp and an obscuration gauge (simulated curve + measured lit fraction),
then encodes H.264 (site-friendly) via ffmpeg.

Frame plan: 25 fps, each session held `hold` frames with `fade` crossfade;
an extra pause at maximum eclipse.

Author: Olivier Vitrac — dwarf3 project. MIT license.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

FPS = 25
CANVAS = (1920, 1080)  # w, h
ANCHOR = (960, 470)    # where the disk center sits on canvas (x, y)


def load_manifest(out_root: Path) -> list[dict]:
    m = json.loads((out_root / "manifest.json").read_text())
    return sorted(m["sessions"], key=lambda s: s["utc_h_mid"])


def smooth_centers(sessions: list[dict], max_dev_px: float = 40.0) -> np.ndarray:
    """
    Robust per-session disk centers (in master-PNG coordinates, 960x540).

    Invalid or outlying centers (RANSAC picked the lunar limb, trees, ...)
    are replaced by linear interpolation over time.
    """
    def ok(s):
        return s["disk_center_xy"] and (s.get("ransac_inlier_frac") or 0) >= 0.15

    t = np.array([s["utc_h_mid"] for s in sessions])
    cx = np.array([s["disk_center_xy"][0] if ok(s) else np.nan for s in sessions])
    cy = np.array([s["disk_center_xy"][1] if ok(s) else np.nan for s in sessions])
    for arr in (cx, cy):
        med = np.nanmedian(arr)
        arr[np.abs(arr - med) > max_dev_px] = np.nan
        valid = ~np.isnan(arr)
        arr[~valid] = np.interp(t[~valid], t[valid], arr[valid])
    return np.column_stack([cx, cy])


def render_session_frame(sess: dict, center: np.ndarray, out_root: Path) -> np.ndarray:
    """Load a master PNG, recenter the disk, upscale to canvas (BGR uint8)."""
    png = cv2.imread(str(out_root / sess["session"] / "stack.png"), cv2.IMREAD_UNCHANGED)
    img8 = (png.astype(np.float32) / 257.0).astype(np.uint8) if png.dtype == np.uint16 else png
    h, w = img8.shape[:2]
    # translate disk center to anchor (canvas coords = 2x master coords)
    dx = ANCHOR[0] / 2 - center[0]
    dy = ANCHOR[1] / 2 - center[1]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img8, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return cv2.resize(shifted, CANVAS, interpolation=cv2.INTER_LANCZOS4)


def draw_overlay(frame: np.ndarray, sess: dict) -> np.ndarray:
    """Time stamp (bottom left) + obscuration gauge (bottom right)."""
    f = frame.copy()
    t = sess["time_local_mid"][11:16]
    obsc = sess["sim"]["obsc"]
    lit = sess.get("lit_fraction_measured")

    cv2.putText(f, t, (60, 1020), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(f, t, (60, 1020), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (235, 235, 235), 2, cv2.LINE_AA)

    # gauge: 400 px bar, simulated obscuration fill, measured tick
    x0, y0, w, h = 1440, 1000, 400, 14
    cv2.rectangle(f, (x0 - 2, y0 - 2), (x0 + w + 2, y0 + h + 2), (0, 0, 0), -1)
    cv2.rectangle(f, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), -1)
    cv2.rectangle(f, (x0, y0), (x0 + int(w * obsc), y0 + h), (60, 140, 235), -1)
    if lit is not None:
        xm = x0 + int(w * max(0.0, min(1.0, 1.0 - lit)))
        cv2.line(f, (xm, y0 - 6), (xm, y0 + h + 6), (235, 235, 235), 2)
    label = f"obscuration {obsc*100:.0f}%"
    cv2.putText(f, label, (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(f, label, (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 1, cv2.LINE_AA)
    return f


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", type=Path, default=Path("processedData/eclipse_2026-08-12"))
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/video/eclipse_timelapse.mp4")
    ap.add_argument("--hold", type=int, default=5, help="frames per session")
    ap.add_argument("--fade", type=int, default=3, help="crossfade frames between sessions")
    ap.add_argument("--max-hold", type=int, default=50, help="extra frames at maximum eclipse")
    args = ap.parse_args(argv)

    out = args.out or args.root / "video" / "eclipse_timelapse.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    sessions = load_manifest(args.root)
    centers = smooth_centers(sessions)

    rendered = []
    for s, c in zip(sessions, centers):
        rendered.append(draw_overlay(render_session_frame(s, c, args.root), s))
    i_max = int(np.argmax([s["sim"]["obsc"] for s in sessions]))

    tmp = Path(tempfile.mkdtemp(prefix="eclipse_video_", dir=out.parent))
    n = 0

    def emit(img: np.ndarray, k: int = 1):
        nonlocal n
        for _ in range(k):
            cv2.imwrite(str(tmp / f"f{n:05d}.png"), img)
            n += 1

    for i, img in enumerate(rendered):
        hold = args.hold + (args.max_hold if i == i_max else 0)
        emit(img, hold)
        if i + 1 < len(rendered):
            nxt = rendered[i + 1]
            for j in range(1, args.fade + 1):
                a = j / (args.fade + 1)
                emit(cv2.addWeighted(img, 1 - a, nxt, a, 0))

    # hold last frame a moment
    emit(rendered[-1], FPS)

    cmd = [
        "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(tmp / "f%05d.png"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    shutil.rmtree(tmp)

    # poster = maximum eclipse frame
    poster = out.with_suffix(".poster.jpg")
    cv2.imwrite(str(poster), rendered[i_max], [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"video : {out}\nposter: {poster}\nframes: {n} ({n/FPS:.1f}s at {FPS} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
