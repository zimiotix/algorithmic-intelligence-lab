#!/usr/bin/env bash
# Add "Algorithmic Intelligence Lab" to the desktop app launcher (GNOME, KDE, ...).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UV="$(command -v uv)"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
mkdir -p "$APPS" "$ICONS"
cp "$ROOT/ailab/app/icon.svg" "$ICONS/ailab.svg"
cat > "$APPS/ailab.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Algorithmic Intelligence Lab
Comment=Learn deterministic intelligent algorithms by playing with them
Exec=$UV run --project "$ROOT" ailab
Path=$ROOT
Icon=ailab
Terminal=false
Categories=Education;Science;
StartupWMClass=ailab
DESKTOP
update-desktop-database "$APPS" 2>/dev/null || true
echo "Installed. Look for 'Algorithmic Intelligence Lab' in your app launcher."
