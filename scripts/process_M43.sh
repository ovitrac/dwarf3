#!/bin/bash
# Process M43 (De Mairan's Nebula / Orion region) - EQ mode
#
# Session: DWARF_RAW_TELE_M 43_EXP_60_GAIN_60_2025-12-28-21-44-39-627
# Frames: 60 discovered, ~55 stacked
# Exposure: 60s x 55 = 55 min total
# Mode: EQ (equatorial mount - integer-pixel shifts preserve Bayer grid)
#
# Note: M43 is an emission/reflection nebula. Use --wb background to avoid
# overcorrecting the natural red H-alpha signal.
#
# Author: Olivier Vitrac, PhD, HDR
#         Generative Simulation Initiative

set -e

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Process M43 (De Mairan's Nebula) session - EQ mode"
    echo ""
    echo "Options:"
    echo "  --skip-align    Skip alignment if cached transforms exist"
    echo "  --workers N     Number of parallel workers (default: 8)"
    echo "  --saturation F  Saturation boost factor (default: 1.4)"
    echo "  --wb METHOD     White balance: background|stars|gray_world|none"
    echo "  -v, --verbose   Verbose logging"
    echo "  -h, --help      Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                      # Full processing"
    echo "  $0 --skip-align         # Re-process with cached alignment"
    echo "  $0 --saturation 1.6     # Higher saturation for nebula"
    exit 0
}

# Check for help flag
for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        usage
    fi
done

cd "$(dirname "$0")/.." || exit 1

SESSION="rawData/DWARF_RAW_TELE_M 43_EXP_60_GAIN_60_2025-12-28-21-44-39-627"

echo "=============================================="
echo "Processing M43 (De Mairan's Nebula)"
echo "Mode: EQ (integer-pixel Bayer-safe alignment)"
echo "=============================================="

python3 scripts/process_session.py "$SESSION" \
    --out processedData \
    --keep 0.92 \
    --sigma 3.0 \
    --workers 8 \
    --eq-mode \
    --wb background \
    --saturation 1.4 \
    --scnr 0.0 \
    "$@"

echo ""
echo "Outputs in: processedData/DWARF_RAW_TELE_M 43_EXP_60_GAIN_60_2025-12-28-21-44-39-627/stacked/"
