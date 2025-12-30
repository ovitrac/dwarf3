<p align="center">
  <img src="docs/assets/banner.png" alt="dwarf3 banner" width="1563"><br>
</p>

# dwarf3🔭•🪐•☄️

**Version 0.22.0-beta** | Reproducible stacking pipeline for [DWARF 3](https://dwarflab.com/) smart telescope FITS acquisitions.

A scientific, scriptable, reproducible processing library that replicates and extends DWARF 3 internal stacking logic with full transparency over frame rejection, registration, and stacking statistics.

See [METHODS.md](docs/METHODS.md) for detailed algorithm descriptions.



## 👤 Author

**Olivier Vitrac, PhD, HDR** | Generative Simulation Initiative | olivier.vitrac@gmail.com



## 🌌 Features

- **Deterministic processing**: Same inputs produce identical outputs
- **Full traceability**: Every frame rejection and decision is logged
- **Science-grade outputs**: Linear FITS^*^ masters suitable for further processing
- **Batch/headless operation**: No GUI required, CI-compatible
- **Transparent quality assessment**: Explainable frame scoring (no black-box ML)

^*^<small>FITS = [Flexible Image Transport System](https://fits.gsfc.nasa.gov/fits_viewer.html) </small>



## 📂 Folder Conventions

```
project/
├── rawData/                    # IMMUTABLE — never modified
│   └── DWARF_RAW_TELE_*/       # One folder per acquisition session
│       ├── *.fits              # Raw light frames
│       ├── failed_*.fits       # Device-flagged failures (excluded)
│       └── shotsInfo.json      # Session metadata
│
├── processedData/              # ALL outputs go here
│   └── <session_id>/
│       ├── stacked/
│       │   ├── master_linear.fits
│       │   ├── master_quicklook.png
│       │   └── master_quicklook.tif
│       ├── run_manifest.json   # Complete processing record
│       ├── report.json         # Summary statistics
│       └── report.md           # Human-readable report
```

**Non-negotiable rule**: `rawData/` is never modified — no renaming, no deletion, no metadata changes.



## ⬇️ Installation

### 🛠️ Using conda (recommended)

```bash
# Create and activate the environment
conda env create -f environment.yaml
conda activate dwarf-astro

# Install dwarf3 in development mode
cd /path/to/dwarf3
pip install -e .
```

### 🔧 Using pip (in existing environment)

```bash
# Install dependencies
pip install -r requirements.txt

# Install dwarf3 in development mode
cd /path/to/dwarf3
pip install -e .
```

### ⚙️ Verifying installation

```bash
# Check version
dwarf3 --version

# Show help
dwarf3 --help
dwarf3 stack --help
```



## 🚀 Quickstart

### >_ Command Line

```bash
# Basic stacking with defaults
dwarf3 stack rawData/DWARF_RAW_TELE_M31_EXP_15_GAIN_60_2025-12-27-18-26-56-449

# With custom parameters
dwarf3 stack rawData/DWARF_RAW_TELE_M31_* \
    --keep 0.92 \
    --sigma 3.0 \
    --out processedData/

# Dry run (score frames, generate report, no stacking)
dwarf3 stack rawData/DWARF_RAW_TELE_M31_* --dry-run
```



### 🐍 Python API

```python
from dwarf3 import stack_session, StackConfig

# Configure the pipeline
config = StackConfig(
    keep_fraction=0.92,      # Keep top 92% of frames by quality
    sigma=3.0,               # Sigma-clipping threshold
    maxiters=5,              # Max clipping iterations
    reference="best",        # Use best-quality frame as reference
)

# Run the pipeline
result = stack_session(
    "rawData/DWARF_RAW_TELE_M31_EXP_15_GAIN_60_2025-12-27-18-26-56-449",
    output_root="processedData",
    config=config,
)

# Access results
print(f"Stacked {len(result.kept)} frames")
print(f"Output: {result.outputs['master_linear']}")
print(f"SNR proxy: {result.stats['snr_proxy']:.2f}")
```



## 🆑 CLI Options

### ☰ Stack Command

```
dwarf3 stack <session_path> [options]

Options:
  --out PATH          Output root directory (default: processedData)
  --keep FLOAT        Fraction of frames to keep (default: 0.92)
  --sigma FLOAT       Sigma for sigma-clipped mean (default: 3.0)
  --maxiters INT      Max iterations for sigma clipping (default: 5)
  --reference METHOD  Reference frame selection: best|first (default: best)
  --debayer MODE      Debayer mode: none|rgb|superpixel|bayer-first (default: none)
  --align-mode MODE   Alignment: integer|rgb_affine|auto (default: auto)
  --workers N         Parallel workers for scoring (default: auto)
  --dry-run           Score and report only, no stacking
  --no-quicklook      Skip PNG/TIFF preview generation
  --use-gpu           Use GPU acceleration if available
  -v, --verbose       Enable verbose output
  -q, --quiet         Suppress colored output

Cache Options:
  --no-cache          Disable caching (compute fresh, don't save)
  --cache-refresh     Clear existing cache before starting
  --frame-list FILE   File containing frame names to include (one per line)

Transform Options:
  --save-transforms   Save alignment transforms for reuse
  --load-transforms   Load cached transforms (skip alignment)
```



### ⌞ ⌝ Frames Command

List and manage frames in a session:

```bash
# List all frames
dwarf3 frames rawData/DWARF_RAW_TELE_M31_*

# List with quality scores (from cache or computed)
dwarf3 frames rawData/DWARF_RAW_TELE_M31_* --scored

# Preview which frames would be kept at 90%
dwarf3 frames rawData/DWARF_RAW_TELE_M31_* --keep 0.90

# Save frame list to file for later use
dwarf3 frames rawData/DWARF_RAW_TELE_M31_* --keep 0.92 --save selected_frames.txt

# Use saved frame list for stacking
dwarf3 stack rawData/DWARF_RAW_TELE_M31_* --frame-list selected_frames.txt
```



### 🗃️ Cache Command

Manage pipeline cache for fast restarts:

```bash
# Show cache status
dwarf3 cache rawData/DWARF_RAW_TELE_M31_*

# Clear all cache
dwarf3 cache rawData/DWARF_RAW_TELE_M31_* --clear

# Clear only alignment transforms
dwarf3 cache rawData/DWARF_RAW_TELE_M31_* --clear-transforms

# Clear only stacked images
dwarf3 cache rawData/DWARF_RAW_TELE_M31_* --clear-stack
```

**Cache artifacts:**
- `scores.json` - Frame quality scores (enables fast re-selection)
- `transforms.json` - Alignment transforms (skip re-alignment)
- `stack_*.fits` - Stacked intermediates (skip re-stacking)
- `calibration.json` - Color calibration parameters



## 📥 Outputs

| File | Description |
|------|-------------|
| `master_linear.fits` | Science-grade linear FITS (float32) |
| `master_quicklook.png` | 8-bit asinh-stretched preview |
| `master_quicklook.tif` | 16-bit asinh-stretched preview |
| `run_manifest.json` | Complete processing record (all frames, scores, transforms) |
| `report.json` | Summary statistics |
| `report.md` | Human-readable Markdown report |



## 🔭 Acquisition Modes

DWARF 3 supports two tracking modes with different processing strategies:

### 🧭 Alt-Az Mode (Default)
- Telescope tracks via altitude/azimuth axes
- Field rotation occurs during long sessions (>80° over 2.5h for M31)
- **Strategy:** Debayer first → full affine transforms → mask-aware stacking

```bash
# Alt-Az mode (default)
./scripts/process_M31.sh
```

### 𝆺𝅥⃝🌍 EQ Mode (Equatorial)
- Telescope tracks via polar axis (no field rotation)
- Frames shift by translation only
- **Strategy:** Integer-pixel Bayer-safe alignment → stack → debayer once

```bash
# EQ mode (--eq-mode flag)
./scripts/process_M43.sh
./scripts/process_M45.sh
```

| Mode | Use When | Advantages |
|------|----------|------------|
| Alt-Az | Long sessions, no polar alignment | Handles any rotation |
| EQ | Polar-aligned mount, short sessions | Sharper colors, faster |

See [METHOD.md](METHOD.md) Section 4 for detailed mode descriptions.



## 🌕 Methods

For detailed algorithm descriptions, see [METHODS.md](docs/METHODS.md).

**Summary:**

- **FITS Decoding:** Unsigned 16-bit via BZERO=32768 offset, converted to float32
- **Frame Selection:** Quality scoring (background, noise, stars) → keep top 92%
- **Acquisition Modes:** Alt-Az (full affine) vs EQ (integer-pixel Bayer-safe)
- **Debayering:** Superpixel (half-res, fast) or bilinear (full-res, color-preserving)
- **Registration:** Star-based alignment via astroalign with parallel processing
- **Stacking:** Two-pass sigma-clipped mean with per-pixel coverage weighting
- **Color:** Background-based white balance, SCNR green removal, saturation boost
- **Visualization:** Asinh stretch for faint-to-bright dynamic range



## 🌌 M31 Complete Pipeline Example

For long alt-az sessions with significant field rotation (e.g., M31 over 2.5 hours with ~87° rotation), use the debayer-first workflow with mask-aware stacking:

```python
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from dwarf3 import (
    list_lights,
    read_fits,
    select_frames,
    debayer_rggb,
    align_rgb_debayer_first_parallel,
    sigma_clip_mask_aware_rgb,
    crop_to_coverage,
    apply_bayer_compensation,
    asinh_stretch,
    write_fits,
    StackConfig,
)

# 1. Session path
session = Path("rawData/DWARF_RAW_TELE_M31_EXP_15_GAIN_60_2025-12-27-18-26-56-449")

# 2. Discover and score frames
frame_paths = list_lights(session)
print(f"Found {len(frame_paths)} light frames")

# 3. Select best 92% by quality
config = StackConfig(keep_fraction=0.92, sigma=3.0)
kept_paths, rejected = select_frames(frame_paths, config=config)
print(f"Keeping {len(kept_paths)}, rejected {len(rejected)}")

# 4. Prepare reference (best frame, debayered)
ref_bayer = read_fits(kept_paths[0])
ref_rgb = debayer_rggb(ref_bayer, mode="bilinear")

# 5. Parallel alignment with explicit validity masks
aligned_frames, results, masks = align_rgb_debayer_first_parallel(
    kept_paths,
    ref_rgb,
    str(kept_paths[0]),
    debayer_mode="bilinear",
    workers=None,  # Auto-detect CPU count
)
successes = sum(1 for r in results if r.success)
print(f"Aligned {successes}/{len(kept_paths)} frames")

# 6. Mask-aware sigma-clipped stacking with edge feathering
stacked_rgb, coverage = sigma_clip_mask_aware_rgb(
    aligned_frames,
    masks=masks,          # Explicit validity masks from alignment
    sigma=3.0,
    maxiters=5,
    feather_width=10,     # Smooth 10-pixel edge transition
)
print(f"Coverage: {coverage.min():.0f}-{coverage.max():.0f} frames per pixel")

# 7. Crop to well-covered region (avoids edge artifacts)
cropped_rgb, bounds = crop_to_coverage(
    stacked_rgb, coverage, min_coverage_fraction=0.8
)
print(f"Cropped to {cropped_rgb.shape[1]}x{cropped_rgb.shape[0]} pixels")

# 8. Auto Bayer compensation
balanced = apply_bayer_compensation(cropped_rgb, auto=True)

# 9. Optional: per-channel smoothing for noise reduction
for c in range(3):
    balanced[:, :, c] = gaussian_filter(balanced[:, :, c], sigma=0.7)

# 10. Asinh stretch for visualization
preview = asinh_stretch(balanced, a=5.0)

# 11. Save outputs
output_dir = Path("processedData") / session.name / "stacked"
output_dir.mkdir(parents=True, exist_ok=True)
write_fits(balanced, output_dir / "master_lrgb_galaxy.fits")

# Save preview (PIL or matplotlib)
from PIL import Image
img = Image.fromarray((np.clip(preview, 0, 1) * 255).astype(np.uint8))
img.save(output_dir / "master_lrgb_galaxy.png")

print(f"Saved to {output_dir}")
```

**Key features used:**
- `align_rgb_debayer_first_parallel`: Parallel debayer + alignment with validity masks
- `sigma_clip_mask_aware_rgb(feather_width=10)`: Soft edge blending prevents hard boundaries
- `crop_to_coverage`: Removes low-coverage edges from rotated stacks
- `apply_bayer_compensation(auto=True)`: Automatic channel balancing
- Coverage map for quality assessment

**Geometry-aware stacking features:**
- **Explicit mask warping**: Validity masks are warped with nearest-neighbor interpolation
- **Edge feathering**: Distance-transform-based soft taper at frame boundaries
- **Coverage cropping**: Automatic cropping to well-sampled region



## 📄 DWARF 3 FITS Header Reference

Typical header fields from DWARF 3:

| Keyword | Example | Description |
|---------|---------|-------------|
| `NAXIS1` | 3840 | Image width |
| `NAXIS2` | 2160 | Image height |
| `BITPIX` | 16 | Bits per pixel (signed) |
| `BZERO` | 32768 | Offset for unsigned |
| `EXPTIME` | 15.0 | Exposure time (seconds) |
| `GAIN` | 60 | Camera gain setting |
| `FILTER` | 'Astro' | Filter used |
| `BAYERPAT` | 'RGGB' | Bayer pattern |
| `RA` | 10.68548 | Right ascension (degrees) |
| `DEC` | 41.27235 | Declination (degrees) |
| `OBJECT` | 'M 31' | Target name |



## 📜 License

MIT License. See [LICENSE](LICENSE) for details.



## ✅ Acknowledgments

This library builds on excellent open-source tools:
- [Astropy](https://www.astropy.org/) for FITS I/O and sigma clipping
- [astroalign](https://github.com/toros-astro/astroalign) for star-based registration
- [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) for numerical operations
