#!/usr/bin/env bash
# Download auxiliary data for er3t example scripts.
# Run from the er3t/examples/ directory.
#
# Prerequisites:
#   Run  er3t/install.sh  first to install core data (REPTRAN + auxiliary files).
#   This script only downloads the LES cloud field needed for examples 02–05.
#
# Data sources:
#   les.nc — Schmidt Lab Google Drive
#
# Previous Google Drive IDs (Hong Chen's, archived):
#   les.nc: 1Oov75VffmuQSljxjoOS6q6egmfT6CmkI

# ── Google Drive file ID ──────────────────────────────────────────────────────
LES_GDRIVE_ID="1cmrZDaCwoQNhaoPGhJ9OhSVEpDU9h-gg"
LES_DEST="data/00_er3t_mca/aux/les.nc"
# ─────────────────────────────────────────────────────────────────────────────

echo "╭────────────────────────────────────────────────╮"
echo "      EaR³T — Example Auxiliary Data Install      "
echo "╰────────────────────────────────────────────────╯"
echo

# Check for gdown
if ! command -v gdown &> /dev/null; then
    echo "[Error] 'gdown' is required to download from Google Drive."
    echo "  Install: pip install gdown"
    echo "  (It is included in the er3t conda environment.)"
    exit 1
fi

# ── LES cloud field ───────────────────────────────────────────────────────────
echo "<1> Downloading LES cloud field (les.nc, ~209 MB) ..."
echo "  Source : Google Drive (Schmidt Lab, ID: $LES_GDRIVE_ID)"
echo "  Target : $LES_DEST"
echo

mkdir -p "$(dirname "$LES_DEST")"
gdown "$LES_GDRIVE_ID" --output "$LES_DEST"

if [ ! -f "$LES_DEST" ]; then
    echo "[Error] LES download failed — '$LES_DEST' not found."
    exit 1
fi
echo "LES cloud field installed."
echo

echo "╭────────────────────────────────────────────────╮"
echo "  All example data installed. Ready to run.       "
echo "╰────────────────────────────────────────────────╯"
