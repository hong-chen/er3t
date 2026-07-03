#!/usr/bin/env bash
# Download and install the er3t core data package and REPTRAN absorption database.
# Run from the er3t/ root directory.
#
# Data sources:
#   er3t core data  — Schmidt Lab Google Drive (temporary; will move to Zenodo for v2)
#   REPTRAN         — libRadtran project: https://www.libradtran.org
#
# For future reference / Zenodo migration:
#   Previous Google Drive ID (Hong Chen's original, archived):
#     1KKpLR7IyqJ4gS6xCxc7f1hwUfUMJksVL

# ── Configuration ────────────────────────────────────────────────────────────
ER3T_DATA_GDRIVE_ID="15YymaUt1i3ad45OZI4kXFDZZlxCNVuGU"
ER3T_DATA_FILE="er3t-data.tar.gz"

REPTRAN_URL="https://www.libradtran.org/lib/exe/fetch.php?media=download:reptran_2024_all.tar.gz"
REPTRAN_FILE="reptran_2024_all.tar.gz"
REPTRAN_DEST="er3t/data/abs/reptran"
# ─────────────────────────────────────────────────────────────────────────────

echo "╭────────────────────────────────────────────────╮"
echo "         EaR³T — Data Package Installer           "
echo "╰────────────────────────────────────────────────╯"
echo

# Check for gdown
if ! command -v gdown &> /dev/null; then
    echo "[Error]: 'gdown' is required but not found."
    echo "  Install: pip install gdown"
    echo "  (It is included in the er3t conda environment.)"
    exit 1
fi

# Check for wget (needed for REPTRAN)
if ! command -v wget &> /dev/null; then
    echo "[Error]: 'wget' is required but not found."
    echo "  macOS:  brew install wget"
    echo "  Ubuntu: sudo apt-get install wget"
    exit 1
fi

# ── Step 1: er3t core data ────────────────────────────────────────────────────
echo "<1.1> Downloading er3t core data package (~39 MB) ..."
echo "  Source : Schmidt Lab Google Drive"
echo
gdown "$ER3T_DATA_GDRIVE_ID" --output "$ER3T_DATA_FILE"

if [ ! -f "$ER3T_DATA_FILE" ]; then
    echo "[Error]: Download failed — '$ER3T_DATA_FILE' not found."
    exit 1
fi

echo "<1.2> Unpacking er3t core data ..."
tar -xzf "$ER3T_DATA_FILE"
rm -f "$ER3T_DATA_FILE"
echo "Done."
echo

# ── Step 2: REPTRAN absorption database ──────────────────────────────────────
echo "<2.1> Downloading REPTRAN absorption database (~142 MB) ..."
echo "  Source : libRadtran project (Gasteiger et al. 2014)"
echo
wget --show-progress -O "$REPTRAN_FILE" "$REPTRAN_URL"

if [ ! -f "$REPTRAN_FILE" ]; then
    echo "[Error]: REPTRAN download failed."
    exit 1
fi

echo "<2.2> Unpacking REPTRAN into $REPTRAN_DEST ..."
mkdir -p "$REPTRAN_DEST"
tar -xzf "$REPTRAN_FILE" -C "$REPTRAN_DEST" --strip-components=1 2>/dev/null || \
tar -xzf "$REPTRAN_FILE" -C "$REPTRAN_DEST"
rm -f "$REPTRAN_FILE"
echo "Done."
echo

echo "╭────────────────────────────────────────────────╮"
echo "   Core data installed.                           "
echo "   Next: run examples/install-examples.sh         "
echo "╰────────────────────────────────────────────────╯"
