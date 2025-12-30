#!/bin/bash
# Process all DWARF 3 sessions
#
# Author: Olivier Vitrac, PhD, HDR
#         Generative Simulation Initiative

cd "$(dirname "$0")/.." || exit 1

echo "=============================================="
echo "DWARF 3 Batch Processing"
echo "=============================================="
echo ""

# Process M31 (Alt-Az mode)
echo "[1/2] Processing M31..."
./scripts/process_M31.sh "$@"
echo ""

# Process M43 (EQ mode)
echo "[2/2] Processing M43..."
./scripts/process_M43.sh "$@"
echo ""

echo "=============================================="
echo "All sessions processed!"
echo "=============================================="
echo ""
echo "Results:"
echo "  M31: processedData/DWARF_RAW_TELE_M 31_.../stacked/"
echo "  M43: processedData/DWARF_RAW_TELE_M 43_.../stacked/"
