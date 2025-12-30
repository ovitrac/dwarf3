#!/bin/bash
# Process M45 (Pleiades / Seven Sisters) - EQ mode
#
# Session: DWARF_RAW_TELE_M 45_EXP_60_GAIN_60_2025-12-28-19-00-32-147
# Frames: 141 total, ~130 valid
# Exposure: 60s x 130 = 2h 10m total
# Mode: EQ (equatorial mount - integer-pixel shifts preserve Bayer grid)
#
# Note: M45 is a reflection nebula with blue stars. The blue reflection
# nebulosity around the stars is faint and requires good processing.
#
# Author: Olivier Vitrac, PhD, HDR
#         Generative Simulation Initiative

set -e

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Process M45 (Pleiades) session - EQ mode"
    echo ""
    echo "Options:"
    echo "  --skip-align    Skip alignment if cached transforms exist"
    echo "  --workers N     Number of parallel workers (default: 8)"
    echo "  --saturation F  Saturation boost factor (default: 1.3)"
    echo "  --wb METHOD     White balance: background|stars|gray_world|none"
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

SESSION="rawData/DWARF_RAW_TELE_M 45_EXP_60_GAIN_60_2025-12-28-19-00-32-147"

echo "=============================================="
echo "Processing M45 (Pleiades / Seven Sisters)"
echo "Mode: EQ (integer-pixel Bayer-safe alignment)"
echo "=============================================="

python3 scripts/process_session.py "$SESSION" \
    --out processedData \
    --keep 0.92 \
    --sigma 3.0 \
    --workers 8 \
    --eq-mode \
    --wb background \
    --saturation 1.3 \
    --scnr 0.0 \
    "$@"

echo ""
echo "Outputs in: processedData/DWARF_RAW_TELE_M 45_EXP_60_GAIN_60_2025-12-28-19-00-32-147/stacked/"
