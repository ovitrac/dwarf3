#!/usr/bin/env python3
"""
Observed-vs-simulated agreement dataset for the 2026-08-12 eclipse.

Measured obscuration per session = 1 - lit_fraction (lit disc area over
pi R^2, from the processed masters). Compared against the simulated
obscuration interpolated at mid-exposure. Emits a compact JSON consumed by
the web page, plus summary statistics.

Author: Olivier Vitrac — dwarf3 project. MIT license.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", type=Path, default=Path("processedData/eclipse_2026-08-12"))
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/agreement.json")
    args = ap.parse_args(argv)

    manifest = json.loads((args.root / "manifest.json").read_text())
    sessions = sorted(manifest["sessions"], key=lambda s: s["utc_h_mid"])

    rows = []
    for s in sessions:
        lit = s.get("lit_fraction_measured")
        alt, hz = s["sim"]["alt"], s["sim"]["hz"]
        # near the tree line, foliage removes light that is not lunar
        # obscuration: measurable, but not comparable
        contaminated = (alt - hz) < 2.0
        rows.append({
            "t": s["time_local_mid"][11:19],
            "utc_h": s["utc_h_mid"],
            "obsc_sim": s["sim"]["obsc"],
            "mag_sim": s["sim"]["mag"],
            "alt": alt,
            "az": s["sim"]["az"],
            "obsc_meas": round(1.0 - lit, 4) if lit is not None else None,
            "occluded": s["occluded"],
            "contaminated": contaminated,
            "transient": s.get("transient", False),
            "n_used": s["n_frames_used"],
            "exptime_s": s["exptime_s"],
        })

    clean = [r for r in rows if r["obsc_meas"] is not None and not r["contaminated"]]
    d = np.array([r["obsc_meas"] - r["obsc_sim"] for r in clean])
    i_max = int(np.argmax([r["obsc_meas"] for r in clean]))
    summary = {
        "n_sessions": len(rows),
        "n_compared": len(clean),
        "rms_deviation": round(float(np.sqrt((d**2).mean())), 4),
        "mean_deviation": round(float(d.mean()), 4),
        "max_abs_deviation": round(float(np.abs(d).max()), 4),
        "max_obsc_measured": clean[i_max]["obsc_meas"],
        "max_obsc_time_local": clean[i_max]["t"],
        "max_obsc_simulated": max(r["obsc_sim"] for r in rows),
    }
    out = args.out or args.root / "agreement.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
