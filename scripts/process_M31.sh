#!/bin/bash
# Process M31 (Andromeda Galaxy) - Alt-Az mode
#
# Session: DWARF_RAW_TELE_M 31_EXP_15_GAIN_60_2025-12-27-18-26-56-449
# Frames: 470 discovered, 432 stacked (92%)
# Exposure: 15s x 432 = 1h 48m total (108 minutes)
# Mode: Alt-Az (field rotation - requires full affine transforms)
#
# dwarf3 v0.23.0-rc1 Test Results (2026-01-01)
# --------------------------------------------
# Alt-Az Mode + CCM Color Test: PASSED
#   - Alignment: Full affine transforms (plane-based stacking)
#   - Processing time: 29.0 minutes
#   - SNR proxy: 48.3 (excellent signal quality)
#   - CCM preset: "rich" (default) - corrected color rendering
#   - White balance: R=1.000, G=0.962, B=1.000 (background method)
#   - Color quality: Significantly improved vs camera-native colors
#
# New v0.23.0-rc1 Features Used:
#   --ccm PRESET     Color Correction Matrix for sRGB conversion:
#                    neutral, rich (default), vivid, ha_emission
#                    Transforms DWARF camera RGB to proper sRGB
#   --wb METHOD      White balance calibration (background, stars, gray_world)
#
# Note: Alt-Az tracking causes field rotation, so EQ mode (integer shifts)
# cannot be used. The pipeline automatically uses full affine transforms.
#
# Author: Olivier Vitrac, PhD, HDR
#         Generative Simulation Initiative

set -e

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Process M31 (Andromeda Galaxy) session"
    echo ""
    echo "Options:"
    echo "  --skip-align    Skip alignment if cached transforms exist"
    echo "  --workers N     Number of parallel workers (default: 8)"
    echo "  --saturation F  Saturation boost factor (default: 1.2)"
    echo "  --wb METHOD     White balance: background|stars|gray_world|none"
    echo "  --ccm PRESET    Color correction: neutral|rich|vivid|ha_emission (default: rich)"
    echo "  -v, --verbose   Verbose logging"
    echo "  -h, --help      Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                      # Full processing"
    echo "  $0 --skip-align         # Re-process with cached alignment"
    echo "  $0 --saturation 1.5     # Higher saturation"
    exit 0
}

# Check for help flag
for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        usage
    fi
done

cd "$(dirname "$0")/.." || exit 1

SESSION="rawData/DWARF_RAW_TELE_M 31_EXP_15_GAIN_60_2025-12-27-18-26-56-449"

echo "=============================================="
echo "Processing M31 (Andromeda Galaxy)"
echo "Mode: Alt-Az (full affine transforms, CCM color correction)"
echo "dwarf3 v0.23.0-rc1"
echo "=============================================="

python3 scripts/process_session.py "$SESSION" \
    --out processedData \
    --keep 0.92 \
    --sigma 3.0 \
    --workers 8 \
    --wb background \
    --saturation 1.2 \
    --scnr 0.0 \
    "$@"

echo ""
echo "Outputs in: processedData/DWARF_RAW_TELE_M 31_EXP_15_GAIN_60_2025-12-27-18-26-56-449/stacked/"
