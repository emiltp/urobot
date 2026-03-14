#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# build.sh – Build the UR Robot Controller macOS .app + .dmg
#
# Usage:
#   conda activate ur-test
#   ./build.sh          # full clean build
#   ./build.sh --quick  # skip clean (faster incremental rebuild)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPEC_FILE="$SCRIPT_DIR/urobot.spec"
APP_NAME="URobotController"
VERSION=$(python -c "from version import __version__; print(__version__)")
SVG_ICON="$SCRIPT_DIR/src/urlogo.svg"
ICNS_ICON="$SCRIPT_DIR/src/urlogo.icns"
INSTALLER_DIR="$SCRIPT_DIR/installer"
DMG_OUTPUT="$INSTALLER_DIR/${APP_NAME}-${VERSION}.dmg"

# ── Colours ──────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Colour

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Pre-flight checks ───────────────────────────────────────

# Detect the active conda environment
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    error "No conda environment is active. Please activate one first:\n       conda activate <env-name>"
fi
CONDA_ENV="$CONDA_DEFAULT_ENV"

# Verify pyinstaller is installed in the active environment
if ! python -c "import PyInstaller" 2>/dev/null; then
    warn "PyInstaller not found in '$CONDA_ENV'. Installing..."
    pip install pyinstaller
fi

# ── Parse arguments ──────────────────────────────────────────
CLEAN_FLAG="--clean"
if [[ "${1:-}" == "--quick" ]]; then
    CLEAN_FLAG=""
    info "Quick build (skipping clean)"
fi

# ── Generate .icns icon from SVG ─────────────────────────────
generate_icns() {
    if [[ ! -f "$SVG_ICON" ]]; then
        warn "SVG icon not found at $SVG_ICON – building without app icon"
        return 1
    fi

    # Skip if .icns is already newer than the .svg
    if [[ -f "$ICNS_ICON" && "$ICNS_ICON" -nt "$SVG_ICON" ]]; then
        info "Icon $ICNS_ICON is up to date"
        return 0
    fi

    info "Generating .icns icon from $SVG_ICON ..."

    local ICONSET_DIR
    ICONSET_DIR=$(mktemp -d)/urlogo.iconset
    mkdir -p "$ICONSET_DIR"

    # Render SVG → 1024×1024 PNG using PyQt6 (already in the env)
    local MASTER_PNG="$ICONSET_DIR/master_1024.png"
    python - "$SVG_ICON" "$MASTER_PNG" <<'PYEOF'
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QImage, QPainter

app = QApplication.instance() or QApplication(sys.argv[:1])
svg_path, out_path = sys.argv[1], sys.argv[2]
renderer = QSvgRenderer(svg_path)
size = 1024
image = QImage(QSize(size, size), QImage.Format.Format_ARGB32_Premultiplied)
image.fill(0)
painter = QPainter(image)
renderer.render(painter)
painter.end()
image.save(out_path)
PYEOF

    if [[ ! -f "$MASTER_PNG" ]]; then
        warn "Failed to render SVG to PNG – building without app icon"
        rm -rf "$(dirname "$ICONSET_DIR")"
        return 1
    fi

    # Create all required icon sizes using sips
    local sizes=(16 32 64 128 256 512 1024)
    for sz in "${sizes[@]}"; do
        sips -z "$sz" "$sz" "$MASTER_PNG" --out "$ICONSET_DIR/icon_${sz}x${sz}.png" >/dev/null 2>&1
    done

    # Rename to Apple's expected iconset filenames
    mv "$ICONSET_DIR/icon_16x16.png"     "$ICONSET_DIR/icon_16x16.png"        2>/dev/null || true
    cp "$ICONSET_DIR/icon_32x32.png"     "$ICONSET_DIR/icon_16x16@2x.png"     2>/dev/null || true
    # 32
    cp "$ICONSET_DIR/icon_64x64.png"     "$ICONSET_DIR/icon_32x32@2x.png"     2>/dev/null || true
    # 128
    cp "$ICONSET_DIR/icon_256x256.png"   "$ICONSET_DIR/icon_128x128@2x.png"   2>/dev/null || true
    # 256
    cp "$ICONSET_DIR/icon_512x512.png"   "$ICONSET_DIR/icon_256x256@2x.png"   2>/dev/null || true
    # 512
    cp "$ICONSET_DIR/icon_1024x1024.png" "$ICONSET_DIR/icon_512x512@2x.png"   2>/dev/null || true

    # Remove non-standard sizes
    rm -f "$ICONSET_DIR/icon_64x64.png" "$ICONSET_DIR/icon_1024x1024.png" "$ICONSET_DIR/master_1024.png"

    # Convert iconset → .icns
    iconutil -c icns "$ICONSET_DIR" -o "$ICNS_ICON"
    rm -rf "$(dirname "$ICONSET_DIR")"

    if [[ -f "$ICNS_ICON" ]]; then
        info "Icon created: $ICNS_ICON"
        return 0
    else
        warn "iconutil failed – building without app icon"
        return 1
    fi
}

# ── Create DMG installer ─────────────────────────────────────
create_dmg() {
    local APP_PATH="$SCRIPT_DIR/dist/${APP_NAME}.app"

    if [[ ! -d "$APP_PATH" ]]; then
        error "App bundle not found at $APP_PATH – cannot create DMG"
    fi

    info "Creating DMG installer..."

    # Ensure installer directory exists
    mkdir -p "$INSTALLER_DIR"

    # Remove old DMG if present
    rm -f "$DMG_OUTPUT"

    # Create a temporary directory for DMG contents
    local DMG_STAGE
    DMG_STAGE=$(mktemp -d)

    # Copy .app into staging area
    cp -R "$APP_PATH" "$DMG_STAGE/"

    # Create a symlink to /Applications for drag-and-drop install
    ln -s /Applications "$DMG_STAGE/Applications"

    # Create the DMG
    hdiutil create \
        -volname "${APP_NAME} ${VERSION}" \
        -srcfolder "$DMG_STAGE" \
        -ov \
        -format UDZO \
        "$DMG_OUTPUT"

    rm -rf "$DMG_STAGE"

    if [[ -f "$DMG_OUTPUT" ]]; then
        info "DMG created: $DMG_OUTPUT"
    else
        error "Failed to create DMG"
    fi
}

# ── Build ────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

info "Building UR Robot Controller v${VERSION}..."
info "  Version   : $VERSION"
info "  Conda env : $CONDA_ENV"
info "  Spec file : $SPEC_FILE"
info "  Project   : $SCRIPT_DIR"
echo ""

# Step 1: Generate icon
generate_icns || true

# Step 2: Run PyInstaller
pyinstaller "$SPEC_FILE" \
    $CLEAN_FLAG \
    --noconfirm \
    --distpath "$SCRIPT_DIR/dist" \
    --workpath "$SCRIPT_DIR/build"

# Step 3: Create DMG
echo ""
if [[ -d "$SCRIPT_DIR/dist/${APP_NAME}.app" ]]; then
    info "PyInstaller build successful!"
    echo ""
    create_dmg
    echo ""
    info "──────────────────────────────────────────────"
    info "Build complete!"
    info ""
    info "  App bundle : dist/${APP_NAME}.app"
    info "  DMG        : installer/${APP_NAME}-${VERSION}.dmg"
    info ""
    info "To run directly:"
    info "  open dist/${APP_NAME}.app"
    info ""
    info "To install:"
    info "  open installer/${APP_NAME}-${VERSION}.dmg"
    info "  → drag ${APP_NAME} to Applications"
    info "──────────────────────────────────────────────"
else
    error "Build completed but .app bundle not found. Check the output above for errors."
fi
