#!/usr/bin/env bash
# One-time setup for the EnergyPlus-Agent plugin.
# Run this from the project root after cloning the repo.
#
# Usage:
#   bash scripts/setup_plugin.sh
#
# Requirements:
#   - uv  (https://docs.astral.sh/uv/)
#   - EnergyPlus 25.1.0+  (https://energyplus.net/downloads)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGIN_DIR="$PROJECT_ROOT/plugins/energyplus_agent"

echo "═══════════════════════════════════════════"
echo " EnergyPlus-Agent Plugin Setup"
echo "═══════════════════════════════════════════"

# ── 1. Ensure submodule is present ─────────────────────────────────────────
if [ ! -f "$PLUGIN_DIR/main.py" ]; then
  echo "► Initialising git submodule…"
  git -C "$PROJECT_ROOT" submodule update --init --recursive
else
  echo "✓ Plugin already present at $PLUGIN_DIR"
fi

# ── 2. Install plugin dependencies via uv ──────────────────────────────────
echo "► Installing plugin dependencies (uv sync)…"
cd "$PLUGIN_DIR"
uv sync

# ── 3. Check for EnergyPlus IDD file ────────────────────────────────────────
IDD_FILE="$PLUGIN_DIR/data/dependencies/Energy+.idd"
if [ ! -f "$IDD_FILE" ]; then
  echo ""
  echo "⚠  Missing: $IDD_FILE"
  echo "   Copy Energy+.idd from your EnergyPlus installation:"
  echo "   Linux:   /usr/local/EnergyPlus-25-1-0/Energy+.idd"
  echo "   macOS:   /Applications/EnergyPlus-25-1-0/Energy+.idd"
  echo "   Windows: C:\\EnergyPlusV25-1-0\\Energy+.idd"
  echo ""
  echo "   mkdir -p $PLUGIN_DIR/data/dependencies"
  echo "   cp /path/to/Energy+.idd $PLUGIN_DIR/data/dependencies/"
else
  echo "✓ IDD file found"
fi

# ── 4. Check for weather file ────────────────────────────────────────────────
EPW_FILE="$PLUGIN_DIR/data/weather/Shenzhen.epw"
if [ ! -f "$EPW_FILE" ]; then
  echo ""
  echo "⚠  Missing weather file: $EPW_FILE"
  echo "   Download from EnergyPlus weather database:"
  echo "   https://energyplus.net/weather"
  echo "   Place the .epw file in: $PLUGIN_DIR/data/weather/"
  echo "   Or set ENERGYPLUS_WEATHER_FILE in .env to a different path."
else
  echo "✓ Default weather file found (Shenzhen.epw)"
fi

echo ""
echo "═══════════════════════════════════════════"
echo " Setup complete!"
echo " Run a simulation with:"
echo "   python main.py -d \"...building description...\" --simulate"
echo "═══════════════════════════════════════════"
