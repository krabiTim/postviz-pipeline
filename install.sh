#!/usr/bin/env bash
# PostViz Pipeline — Team Install Script
#
# Prerequisites:
#   - WSL2 Ubuntu 24.04
#   - NVIDIA GPU with CUDA passthrough enabled
#   - ComfyUI running on port 8188
#
# Usage:
#   git clone https://github.com/krabiTim/postviz-pipeline.git ~/postviz-pipeline
#   cd ~/postviz-pipeline
#   bash install.sh

set -euo pipefail

VENV="${HOME}/comfyui-postviz/venv"
FFMPEG_BIN="${HOME}/bin/ffmpeg"
FFPROBE_BIN="${HOME}/bin/ffprobe"

echo ""
echo "================================================"
echo "  PostViz Pipeline — Install"
echo "================================================"
echo ""

# ── 1. System dependencies ────────────────────────────────────────────────────
echo "[1/6] Checking system dependencies..."

missing=()

if ! command -v colmap &>/dev/null; then
    missing+=("colmap")
fi

if ! command -v python3 &>/dev/null; then
    missing+=("python3")
fi

if ! command -v git &>/dev/null; then
    missing+=("git")
fi

if [ ${#missing[@]} -gt 0 ]; then
    echo "    Missing packages: ${missing[*]}"
    echo "    Run: sudo apt update && sudo apt install -y ${missing[*]}"
    echo "    Then re-run this script."
    exit 1
fi

echo "    OK — colmap, python3, git found"

# ── 2. FFmpeg static binary ───────────────────────────────────────────────────
echo "[2/6] Checking FFmpeg..."
mkdir -p "${HOME}/bin"

if [ ! -f "${FFMPEG_BIN}" ]; then
    echo "    Downloading FFmpeg static binary..."
    FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    TMP=$(mktemp -d)
    curl -L "${FFMPEG_URL}" -o "${TMP}/ffmpeg.tar.xz"
    tar -xf "${TMP}/ffmpeg.tar.xz" -C "${TMP}" --strip-components=1
    cp "${TMP}/ffmpeg" "${FFMPEG_BIN}"
    cp "${TMP}/ffprobe" "${FFPROBE_BIN}"
    chmod +x "${FFMPEG_BIN}" "${FFPROBE_BIN}"
    rm -rf "${TMP}"
    echo "    FFmpeg installed at ${FFMPEG_BIN}"
else
    echo "    OK — FFmpeg already at ${FFMPEG_BIN}"
fi

# ── 3. Python venv ────────────────────────────────────────────────────────────
echo "[3/6] Setting up Python virtual environment..."

if [ ! -d "${VENV}" ]; then
    echo "    Creating venv at ${VENV}..."
    mkdir -p "$(dirname "${VENV}")"
    python3 -m venv "${VENV}"
fi

# shellcheck disable=SC1090
source "${VENV}/bin/activate"
echo "    Upgrading pip..."
pip install --quiet --upgrade pip

# ── 4. Python dependencies ────────────────────────────────────────────────────
echo "[4/6] Installing Python dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install --quiet -r "${SCRIPT_DIR}/requirements.txt"
echo "    Dependencies installed"

# ── 5. Blender (optional, for FBX camera import) ─────────────────────────────
echo "[5/6] Checking Blender (for FBX/USD camera import)..."

if command -v blender &>/dev/null; then
    BLENDER_VERSION=$(blender --version 2>/dev/null | head -1)
    echo "    OK — ${BLENDER_VERSION}"
elif command -v snap &>/dev/null; then
    echo "    Blender not found. Install with:"
    echo "      sudo snap install blender --classic"
    echo "    (optional — only needed for FBX/USD camera import)"
else
    echo "    Blender not found (optional — needed for FBX/USD camera import)"
    echo "    Install: sudo apt install blender  OR  sudo snap install blender --classic"
fi

# ── 6. Project directories ────────────────────────────────────────────────────
echo "[6/6] Creating project directories..."
mkdir -p "${SCRIPT_DIR}/projects"
mkdir -p "${SCRIPT_DIR}/delivery"
mkdir -p "${SCRIPT_DIR}/workflows"
echo "    OK"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "  Install complete!"
echo ""
echo "  Start the app:"
echo "    source ${VENV}/bin/activate"
echo "    cd ${SCRIPT_DIR}"
echo "    python main.py"
echo ""
echo "  Then open: http://localhost:7860"
echo "================================================"
echo ""
